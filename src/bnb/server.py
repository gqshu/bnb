"""FastAPI service for the single live binaural-beat stream.

One stream, one fixed port. The service owns a single :class:`StreamEngine`; the
endpoints start/stop it, read/update its spec, and expose the audio itself at
``/stream.wav`` for a browser ``<audio>`` element. The demo portal is served at ``/``.

Endpoints:
    GET   /                     the demo portal
    GET   /api/stream           current stream state (running, beat, background)
    POST  /api/stream/start     start the stream (beat and background both optional)
    POST  /api/stream/stop      stop the stream
    PATCH /api/stream/spec      live-update beat / background / volumes
    GET   /api/backgrounds      catalog of background tracks (rendered ones are playable)
    GET   /stream.wav           the live audio (open-ended WAV, paced to real time)
"""

from __future__ import annotations

import random
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import lameenc

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from .background import SPECIAL_GROUPS
from .catalog import CategoryManager
from .stream import Beat, StreamEngine, to_int16_bytes, wav_stream_header
from .tone import (
    CARRIER_HZ,
    ISOCHRONIC_MAX_CARRIER_HZ,
    ISOCHRONIC_MIN_CARRIER_HZ,
    MAX_AM_BEAT_HZ,
    MAX_BEAT_HZ,
    MAX_CARRIER_HZ,
    MAX_DEPTH,
    MAX_DUTY,
    MAX_RAMP_MS,
    MIN_CARRIER_HZ,
    MIN_DEPTH,
    MIN_DUTY,
    MIN_RAMP_MS,
    Waveform,
)

PORT = 8000
CHUNK_SECONDS = 0.2  # render granularity; also the live-edit latency floor
# Burst the first few seconds as fast as the socket accepts them, then pace to real
# time. Without this, an open-ended stream delivered at exactly 1× realtime never lets
# the player build a jitter buffer, so mobile players (WeChat InnerAudioContext) sit in
# "waiting" and never start. The burst gives them something to start on; edits then lag
# by ~this much, which is fine for background music.
PREBUFFER_SECONDS = 3.0
# Android's audio player refuses an endless chunked stream with no length — it wants
# Range support and a Content-Length. The stream is truly endless, so we advertise a
# huge fake length the player never reaches within a session, answer its opening
# ``Range: bytes=0-`` with a 206, and just keep streaming live audio. iOS is happy
# with this too. ~10h at any sane bitrate; a session ends long before it matters.
FAKE_STREAM_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
MP3_BITRATE_KBPS = 128  # stereo; ample for a soundscape + sine beat
MP3_QUALITY = 5  # lameenc: 0=best/slowest … 9=fastest; 5 keeps per-stream CPU low
NATURAL_SOUNDS_GROUP = "natural_sounds"
FOCUS_GOAL = "focus"

# Named focus presets (control.md's "named target states, not raw parameters" — the same
# rule relax already follows — extended to the am_music/Brain.fm-competing mode). Hz/depth
# are internal; the client only ever sees the preset name. "Deep Work" is the slower,
# gentler pulse for a longer session; "Quick Focus" a brisker one for a short session.
# Values are starting guesses to tune on real sessions, not derived constants.
FOCUS_PRESETS: dict[str, dict[str, Any]] = {
    "deep_work": {"beat_hz": 14.0, "depth": 0.35, "modulator": "sine"},
    "quick_focus": {"beat_hz": 18.0, "depth": 0.5, "modulator": "sine"},
}

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title="bnb — binaural beat stream")
engine = StreamEngine()
categories = CategoryManager()

# Idle sessions are reaped this long after their last control call. A client that
# leaves without calling DELETE (app killed, network drop) is cleaned up lazily on
# the next `POST /api/session`. Proper per-connection reaping (on stream disconnect)
# is a later refinement.
SESSION_TTL_SECONDS = 30 * 60


class SessionManager:
    """A registry of independent :class:`StreamEngine` instances keyed by a
    server-minted session id, so many clients can each drive their own stream
    concurrently. Guarded by a lock; idle sessions are swept lazily on create."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._engines: dict[str, StreamEngine] = {}
        self._touched: dict[str, float] = {}

    def _sweep_locked(self) -> None:
        now = time.monotonic()
        stale = [sid for sid, t in self._touched.items() if now - t > SESSION_TTL_SECONDS]
        for sid in stale:
            eng = self._engines.pop(sid, None)
            self._touched.pop(sid, None)
            if eng is not None:
                eng.stop()

    def create(self) -> str:
        sid = uuid.uuid4().hex
        with self._lock:
            self._sweep_locked()
            self._engines[sid] = StreamEngine()
            self._touched[sid] = time.monotonic()
        return sid

    def get(self, sid: str) -> StreamEngine:
        with self._lock:
            eng = self._engines.get(sid)
            if eng is None:
                raise HTTPException(status_code=404, detail=f"session {sid!r} not found")
            self._touched[sid] = time.monotonic()
            return eng

    def remove(self, sid: str) -> None:
        with self._lock:
            eng = self._engines.pop(sid, None)
            self._touched.pop(sid, None)
        if eng is not None:
            eng.stop()

    def count(self) -> int:
        with self._lock:
            return len(self._engines)


sessions = SessionManager()


class BeatSpec(BaseModel):
    """A binaural/monaural/isochronic/am_music beat. Bounds come straight from the
    audio-design constraints.

    ``beat_hz`` may be exactly 0: both ears then get the same carrier and no beat
    exists, which is the acoustically-matched **sham** condition (carrier audible,
    Δ absent) the EEG pilot compares against. The engine renders it natively —
    identical phase on both channels. Note this is deliberately looser than
    ``tone.render_binaural``, which still rejects 0 because a 0 Hz *binaural* WAV
    is not a binaural tone at all; the sham only exists as a live stream state.

    ``carrier_hz``'s valid range depends on ``mode`` — isochronic's usable carrier
    range (100-500 Hz) sits lower than the other modes' (200-900 Hz), so it's
    checked in ``_mode_bounds`` rather than as a static ``Field`` bound.
    ``depth``/``duty``/``ramp_ms`` apply to ``isochronic`` and ``am_music``.
    ``modulator`` only applies to ``am_music``. ``beat_hz``'s upper bound is also
    mode-dependent: am_music has no binaural-fusion ceiling, so it's allowed up to
    ``tone.MAX_AM_BEAT_HZ`` (60 Hz) rather than ``tone.MAX_BEAT_HZ`` (40 Hz).
    """

    carrier_hz: float = Field(CARRIER_HZ)
    beat_hz: float = Field(10.0, ge=0.0)
    volume: float = Field(0.5, ge=0.0, le=1.0)
    waveform: Waveform = "sine"
    mode: Literal["dichotic", "diotic", "monaural", "isochronic", "am_music"] = "dichotic"
    depth: float = Field(1.0, ge=MIN_DEPTH, le=MAX_DEPTH)
    duty: float = Field(0.5, ge=MIN_DUTY, le=MAX_DUTY)
    ramp_ms: float = Field(5.0, ge=MIN_RAMP_MS, le=MAX_RAMP_MS)
    modulator: Literal["sine", "gate"] = "sine"

    @model_validator(mode="after")
    def _mode_bounds(self) -> "BeatSpec":
        if self.mode == "isochronic":
            lo, hi = ISOCHRONIC_MIN_CARRIER_HZ, ISOCHRONIC_MAX_CARRIER_HZ
        else:
            lo, hi = MIN_CARRIER_HZ, MAX_CARRIER_HZ
        if not lo <= self.carrier_hz <= hi:
            raise ValueError(
                f"carrier {self.carrier_hz} Hz outside usable range {lo}-{hi} Hz for mode {self.mode!r}"
            )
        beat_hi = MAX_AM_BEAT_HZ if self.mode == "am_music" else MAX_BEAT_HZ
        if not self.beat_hz <= beat_hi:
            raise ValueError(f"beat {self.beat_hz} Hz outside 0-{beat_hi} Hz for mode {self.mode!r}")
        return self

    def to_engine(self) -> Beat:
        return Beat(
            self.carrier_hz,
            self.beat_hz,
            self.volume,
            self.waveform,
            self.mode,
            self.depth,
            self.duty,
            self.ramp_ms,
            self.modulator,
        )


class StartRequest(BaseModel):
    beat: BeatSpec | None = None
    background_id: str | None = None
    background_volume: float = Field(1.0, ge=0.0, le=1.0)
    preset: Literal["deep_work", "quick_focus"] | None = None


class SpecUpdate(BaseModel):
    """A partial live update. Only the fields present in the body are applied;
    ``beat: null`` removes the beat, ``background_id: null`` removes the background."""

    beat: BeatSpec | None = None
    background_id: str | None = None
    background_volume: float | None = Field(None, ge=0.0, le=1.0)
    preset: Literal["deep_work", "quick_focus"] | None = None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB_DIR / "index.html").read_text()


@app.get("/api/stream")
def get_state() -> dict:
    return engine.snapshot()


# ── Shared engine operations (used by both the legacy single-stream API and the
#    per-session API) ─────────────────────────────────────────────────────────
def _start_engine(eng: StreamEngine, req: StartRequest) -> dict:
    try:
        eng.start(
            beat=req.beat.to_engine() if req.beat else None,
            background_id=req.background_id,
            background_volume=req.background_volume,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return eng.snapshot()


def _update_engine(eng: StreamEngine, update: SpecUpdate) -> dict:
    """Applies whichever fields are present as one atomic update (see
    :meth:`StreamEngine.update`) — a request that changes ``beat`` and
    ``background_id`` together is validated by what they end up as, not by
    which field happens to be applied first."""
    update = _resolve_preset_update(update)
    fields = update.model_fields_set
    kwargs: dict[str, Any] = {}
    if "beat" in fields:
        kwargs["beat"] = update.beat.to_engine() if update.beat else None
    if "background_id" in fields:
        kwargs["background_id"] = update.background_id
    if "background_volume" in fields and update.background_volume is not None:
        kwargs["background_volume"] = update.background_volume
    try:
        eng.update(**kwargs)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return eng.snapshot()


def _audio_stream(gen, media_type: str, range_header: str | None) -> StreamingResponse:
    """Wrap a live-audio generator with headers Android's player needs: Range support
    and a (fake, huge) Content-Length. An opening ``Range: bytes=START-`` gets a 206;
    START is otherwise ignored because the stream is live, not seekable."""
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-cache, no-store"}
    if range_header:
        try:
            start = int(range_header.split("=", 1)[1].split("-", 1)[0] or "0")
        except (ValueError, IndexError):
            start = 0
        headers["Content-Range"] = f"bytes {start}-{FAKE_STREAM_BYTES - 1}/{FAKE_STREAM_BYTES}"
        headers["Content-Length"] = str(max(1, FAKE_STREAM_BYTES - start))
        status = 206
    else:
        headers["Content-Length"] = str(FAKE_STREAM_BYTES)
        status = 200
    return StreamingResponse(gen, status_code=status, media_type=media_type, headers=headers)


def _wav_response(eng: StreamEngine, range_header: str | None = None) -> StreamingResponse:
    """Open-ended WAV of one engine's live mix, paced to real time."""

    def generate():
        yield wav_stream_header(eng.sample_rate)
        n = int(eng.sample_rate * CHUNK_SECONDS)
        prebuffer = int(PREBUFFER_SECONDS / CHUNK_SECONDS)
        sent = 0
        deadline: float | None = None
        while eng.running:
            yield to_int16_bytes(eng.read(n))
            sent += 1
            if sent < prebuffer:
                continue  # burst the prebuffer as fast as the socket accepts
            if deadline is None:
                deadline = time.monotonic()  # start the realtime clock after the burst
            deadline += CHUNK_SECONDS
            behind = deadline - time.monotonic()
            if behind > 0:
                time.sleep(behind)

    return _audio_stream(generate(), "audio/wav", range_header)


def _mp3_response(eng: StreamEngine, range_header: str | None = None) -> StreamingResponse:
    """Open-ended MP3 of one engine's live mix, paced to real time. WeChat's
    ``InnerAudioContext`` (and iOS ``<audio>``) play a chunked MP3 reliably where a
    long open-ended WAV is flaky, so this is the mobile-facing stream. Each engine
    chunk is LAME-encoded incrementally (lameenc buffers frames as needed)."""

    def generate():
        encoder = lameenc.Encoder()
        encoder.set_bit_rate(MP3_BITRATE_KBPS)
        encoder.set_in_sample_rate(eng.sample_rate)
        encoder.set_channels(2)
        encoder.set_quality(MP3_QUALITY)
        n = int(eng.sample_rate * CHUNK_SECONDS)
        prebuffer = int(PREBUFFER_SECONDS / CHUNK_SECONDS)
        sent = 0
        deadline: float | None = None
        encoded_any = False
        while eng.running:
            data = encoder.encode(to_int16_bytes(eng.read(n)))
            encoded_any = True
            if data:
                yield bytes(data)
            sent += 1
            if sent < prebuffer:
                continue  # burst the prebuffer as fast as the socket accepts
            if deadline is None:
                deadline = time.monotonic()  # start the realtime clock after the burst
            deadline += CHUNK_SECONDS
            behind = deadline - time.monotonic()
            if behind > 0:
                time.sleep(behind)
        if encoded_any:  # lameenc errors if flush() runs before any encode()
            tail = encoder.flush()
            if tail:
                yield bytes(tail)

    return _audio_stream(generate(), "audio/mpeg", range_header)


def _compatible_keywords(group: str, goal: str) -> set[str] | None:
    """Keywords in a special group whose ``KeywordEntry.goals`` allow ``goal``.

    Special-group specs carry no per-track ``goal`` (it's metadata on the taxonomy
    definition, not the render — see ``background.KeywordEntry.goals``), so filtering
    happens here, cross-referencing a catalog entry's ``keyword`` against the group's
    declared allow-list. Returns ``None`` for a group with no keyword-goal metadata
    (nothing to filter on), which callers treat as "everything is compatible."
    """
    spec_group = SPECIAL_GROUPS.get(group)
    if spec_group is None:
        return None
    return {kw for kw, entry in spec_group.keywords.items() if goal in entry.goals}


def _random_special_background(group: str = NATURAL_SOUNDS_GROUP, goal: str = "relax") -> str | None:
    """A random rendered background from a special group, compatible with ``goal``
    (§ :func:`_compatible_keywords`), or None if nothing rendered matches."""
    candidates = categories.search(group=group, rendered=True)
    allowed = _compatible_keywords(group, goal)
    if allowed is not None:
        candidates = [e for e in candidates if e["keyword"] in allowed]
    return random.choice(candidates)["track_id"] if candidates else None


def _random_natural_background(goal: str = "relax") -> str | None:
    """A random rendered ``natural_sounds`` track_id compatible with ``goal`` — the
    session blank-start fallback (§ ``session_start``)."""
    return _random_special_background(NATURAL_SOUNDS_GROUP, goal)


def _random_focus_background() -> str | None:
    """A random rendered ``goal=focus`` grid track_id, or None if none are rendered yet —
    the bed a focus preset's am_music beat modulates."""
    entry = categories.pick(goal=FOCUS_GOAL, rendered=True)
    return entry["track_id"] if entry else None


def _beat_from_preset(name: str) -> BeatSpec:
    cfg = FOCUS_PRESETS[name]
    return BeatSpec(mode="am_music", beat_hz=cfg["beat_hz"], depth=cfg["depth"], modulator=cfg["modulator"])


def _resolve_preset(req: StartRequest) -> StartRequest:
    """A named focus preset expands into the am_music ``BeatSpec`` plus a goal=focus
    background (never exposing Hz to the client — control.md's "named target states, not
    raw parameters" rule). A ``background_id`` the caller already set wins over the
    auto-pick, same precedence as the existing blank-request natural_sounds auto-pick."""
    if req.preset is None:
        return req
    if req.preset not in FOCUS_PRESETS:
        raise HTTPException(status_code=400, detail=f"unknown preset {req.preset!r}, expected one of {list(FOCUS_PRESETS)}")
    return req.model_copy(
        update={
            "beat": _beat_from_preset(req.preset),
            "background_id": req.background_id or _random_focus_background(),
        }
    )


def _resolve_preset_update(update: SpecUpdate) -> SpecUpdate:
    """:func:`_resolve_preset`'s counterpart for a partial live update — only fills
    ``background_id`` from the auto-pick if the caller didn't explicitly set one."""
    if update.preset is None:
        return update
    if update.preset not in FOCUS_PRESETS:
        raise HTTPException(status_code=400, detail=f"unknown preset {update.preset!r}, expected one of {list(FOCUS_PRESETS)}")
    fields: dict[str, Any] = {"beat": _beat_from_preset(update.preset)}
    if "background_id" not in update.model_fields_set:
        fields["background_id"] = _random_focus_background()
    return update.model_copy(update=fields)


# ── Legacy single-stream API (the demo portal + existing tests) ───────────────
@app.post("/api/stream/start")
def start(req: StartRequest) -> dict:
    return _start_engine(engine, _resolve_preset(req))


@app.post("/api/stream/stop")
def stop() -> dict:
    engine.stop()
    return engine.snapshot()


@app.patch("/api/stream/spec")
def update_spec(update: SpecUpdate) -> dict:
    return _update_engine(engine, update)


_debug_mp3_path: str | None = None


@app.get("/debug/sample.mp3")
def debug_sample_mp3():
    """DIAGNOSTIC: a *complete, finite* MP3 file (real Content-Length, seekable via
    FileResponse) of a LOUD, unmistakable 440 Hz test tone. Isolates device audio
    output from content: if this is silent on a device, the problem is the device's
    audio (volume/focus/routing), not our streaming or the (quiet) natural beds."""
    global _debug_mp3_path
    import tempfile

    import numpy as np
    from fastapi.responses import FileResponse

    if _debug_mp3_path is None or not Path(_debug_mp3_path).exists():
        sr = 44100
        t = np.arange(int(sr * 6)) / sr
        tone = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        stereo = np.stack([tone, tone], axis=1)
        enc = lameenc.Encoder()
        enc.set_bit_rate(MP3_BITRATE_KBPS)
        enc.set_in_sample_rate(sr)
        enc.set_channels(2)
        enc.set_quality(MP3_QUALITY)
        mp3 = bytes(enc.encode(to_int16_bytes(stereo))) + bytes(enc.flush())
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.write(mp3)
        tmp.close()
        _debug_mp3_path = tmp.name
    return FileResponse(_debug_mp3_path, media_type="audio/mpeg")


_bg_mp3_cache: dict[str, str] = {}
_BG_NAMES = {
    "rain": "雨声",
    "ocean": "海浪",
    "wind": "风声",
    "stream": "溪流",
    "forest": "森林",
    "night": "夜",
    "chimes": "风铃",
}


def _bg_display_name(track_id: str) -> str:
    parts = track_id.split("_")  # natural_sounds_<keyword>_seed...
    kw = parts[2] if len(parts) >= 3 else ""
    return _BG_NAMES.get(kw, "自然音")


def _background_mp3_path(track_id: str) -> str:
    """Encode a rendered background wav to a complete MP3 file once, cached by track_id.
    The client downloads this and decodes it into a WebAudio buffer (no live stream)."""
    cached = _bg_mp3_cache.get(track_id)
    if cached and Path(cached).exists():
        return cached
    import tempfile

    import numpy as np
    import soundfile as sf

    from . import assets

    src = assets.find_track(track_id)
    if src is None:
        raise HTTPException(status_code=404, detail=f"no rendered audio for {track_id!r}")
    data, sr = sf.read(str(src), dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    enc = lameenc.Encoder()
    enc.set_bit_rate(MP3_BITRATE_KBPS)
    enc.set_in_sample_rate(sr)
    enc.set_channels(2)
    enc.set_quality(MP3_QUALITY)
    mp3 = bytes(enc.encode(to_int16_bytes(data))) + bytes(enc.flush())
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.write(mp3)
    tmp.close()
    _bg_mp3_cache[track_id] = tmp.name
    return tmp.name


@app.get("/api/backgrounds/random")
def random_background(group: str = NATURAL_SOUNDS_GROUP, goal: Literal["relax", "focus"] = "relax") -> dict:
    """Pick a random rendered background compatible with ``goal``; returns its id, display
    name, and file URL. ``goal`` only restricts groups with keyword-goal metadata (today,
    ``natural_sounds`` — see ``KeywordEntry.goals``); a group with none is unfiltered."""
    tid = _random_special_background(group, goal)
    if tid is None:
        raise HTTPException(status_code=404, detail="no rendered background track")
    return {"track_id": tid, "name": _bg_display_name(tid), "url": f"/background/{tid}.mp3"}


@app.get("/background/{track_id}.mp3")
def background_mp3(track_id: str):
    """A complete, downloadable MP3 of one background bed (seekable file, not a stream)."""
    from fastapi.responses import FileResponse

    return FileResponse(_background_mp3_path(track_id), media_type="audio/mpeg")


@app.get("/api/backgrounds")
def backgrounds() -> list[dict]:
    return [
        {"track_id": e["track_id"], "summary": e["summary"], "rendered": e["rendered"]}
        for e in categories.search()
    ]


@app.get("/stream.wav")
def stream_wav(request: Request) -> StreamingResponse:
    return _wav_response(engine, request.headers.get("range"))


@app.get("/stream.mp3")
def stream_mp3(request: Request) -> StreamingResponse:
    return _mp3_response(engine, request.headers.get("range"))


# ── Per-session API (concurrent streams, one engine per session id) ───────────
@app.post("/api/session")
def create_session() -> dict:
    """Mint a new session with its own stream engine. Returns its ``session_id``,
    which the client passes on every subsequent control/stream call."""
    return {"session_id": sessions.create(), "active_sessions": sessions.count()}


@app.delete("/api/session/{sid}")
def delete_session(sid: str) -> dict:
    """Stop and drop a session. Idempotent — deleting an unknown id is a no-op."""
    sessions.remove(sid)
    return {"ok": True}


@app.get("/api/session/{sid}")
def session_state(sid: str) -> dict:
    return sessions.get(sid).snapshot()


@app.post("/api/session/{sid}/start")
def session_start(sid: str, req: StartRequest) -> dict:
    eng = sessions.get(sid)
    req = _resolve_preset(req)
    # Blank spec (no beat, no background, no preset) → the backend picks a random
    # rendered natural_sounds bed, so the client can just "start" and hear something.
    # An explicit background_id, beat, or preset is respected as-is.
    if req.beat is None and req.background_id is None:
        req = req.model_copy(update={"background_id": _random_natural_background("relax")})
    return _start_engine(eng, req)


@app.post("/api/session/{sid}/stop")
def session_stop(sid: str) -> dict:
    eng = sessions.get(sid)
    eng.stop()
    return eng.snapshot()


@app.patch("/api/session/{sid}/spec")
def session_spec(sid: str, update: SpecUpdate) -> dict:
    return _update_engine(sessions.get(sid), update)


@app.get("/stream/{sid}.wav")
def session_stream_wav(sid: str, request: Request) -> StreamingResponse:
    return _wav_response(sessions.get(sid), request.headers.get("range"))


@app.get("/stream/{sid}.mp3")
def session_stream_mp3(sid: str, request: Request) -> StreamingResponse:
    return _mp3_response(sessions.get(sid), request.headers.get("range"))


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT)


if __name__ == "__main__":
    main()
