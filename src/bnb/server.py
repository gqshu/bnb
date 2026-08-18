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
    GET   /api/profiles         mode-profile catalog for the app's selection grid
    GET   /api/catalog          full catalog + filter facets (the catalog-management page)
    POST  /api/catalog/tags     tag tracks           (only when tagging is enabled)
    DELETE /api/catalog/tags    untag tracks         (only when tagging is enabled)
    GET   /stream.wav           the live audio (open-ended WAV, paced to real time)
"""

from __future__ import annotations

import os
import random
import sys
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import lameenc

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from .background import SPECIAL_GROUPS, SUBSTRATES
from .catalog import CategoryManager
from .profiles import list_profiles, profiles_image_dir
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
HOST = os.environ.get("BNB_HOST", "127.0.0.1")
"""Interface :func:`main` binds, overridable with ``--host``.

Loopback by default: the service has no authentication and (with tagging on) writes
to the asset repo, so it should not land on a shared network unless someone asks for
it. ``0.0.0.0`` binds every address — what the mini program needs to reach the host
over the LAN, and what a container needs, where loopback means "unreachable from
anywhere" (docker/Dockerfile passes it to uvicorn directly for that reason).
"""
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

FALSEY = {"0", "false", "no", "off", ""}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() not in FALSEY


# Tagging is the one API that *writes* to the asset repo, so it's a switch decided at
# service start rather than always-on: the same code can serve streams to clients with
# catalog mutation closed off. On by default (the local dashboard is the main caller);
# ``BNB_ENABLE_TAGGING=0`` in the environment or ``--no-tagging`` on the command line
# turns it off, and the endpoints then 403 while ``GET /api/catalog`` reports the state
# so the dashboard can grey its tagging controls out instead of failing on submit.
TAGGING_ENABLED = _env_flag("BNB_ENABLE_TAGGING", True)
MAX_TAG_LENGTH = 64

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


def _special_pool(group: str, goal: str) -> list[dict]:
    """Rendered tracks in a special group compatible with ``goal``.

    Special-group specs carry no per-track goal, so compatibility comes from the
    keyword's :class:`~bnb.background.KeywordEntry` goal allow-list, cross-referenced
    here (§ :func:`_compatible_keywords`)."""
    candidates = categories.search(group=group, rendered=True)
    allowed = _compatible_keywords(group, goal)
    if allowed is not None:
        candidates = [e for e in candidates if e["keyword"] in allowed]
    return candidates


def _random_special_entry(group: str, goal: str) -> dict | None:
    """A random rendered entry from a special group compatible with ``goal``, or None."""
    pool = _special_pool(group, goal)
    return random.choice(pool) if pool else None


def _random_special_background(group: str = NATURAL_SOUNDS_GROUP, goal: str = "relax") -> str | None:
    """The :func:`_random_special_entry` track_id, or None."""
    entry = _random_special_entry(group, goal)
    return entry["track_id"] if entry else None


def _typed_pool(category: str, goal: str) -> list[dict]:
    """Rendered backgrounds of one ``type`` compatible with ``goal``.

    ``category`` names either a grid **substrate** — the physical kind of bed (``drone``,
    ``noise_texture``, ``percussive_with_tail``…), matched on the track's per-track ``goal``
    field — or a special **group** (``natural_sounds``), matched on the keyword goal
    allow-list (§ :func:`_special_pool`). Substrate rather than cultural style is the axis
    a listener actually picks a background by. Raises 400 for a type in neither."""
    if category in SUBSTRATES:
        return categories.search(substrate=category, goal=goal, rendered=True)
    if category in SPECIAL_GROUPS:
        return _special_pool(category, goal)
    raise HTTPException(
        status_code=400,
        detail=f"unknown background type {category!r}, expected one of {sorted([*SUBSTRATES, *SPECIAL_GROUPS])}",
    )


def _goal_compatible_pool(goal: str) -> list[dict]:
    """Every rendered background compatible with ``goal``, across all types.

    The union of the grid tracks whose per-track ``goal`` field matches and every special
    group's goal-compatible tracks. ``search(goal=…)`` naturally excludes the special
    tracks (their goal field is None), so the two pools don't overlap."""
    pool = categories.search(goal=goal, rendered=True)
    for group in SPECIAL_GROUPS:
        pool.extend(_special_pool(group, goal))
    return pool


def _random_background_entry(goal: str, category: str | None = None, exclude: str | None = None) -> dict | None:
    """A random rendered background compatible with ``goal``, or None if nothing matches.

    ``category`` is the optional ``type`` filter: given, the pool is that one type
    (§ :func:`_typed_pool`); omitted, it's every compatible type (§ :func:`_goal_compatible_pool`).
    ``exclude`` drops one track_id (the one already playing) so the switch button lands on
    a different bed, falling back to the full pool if that's the only compatible track."""
    pool = _goal_compatible_pool(goal) if category is None else _typed_pool(category, goal)
    if exclude is not None:
        pool = [e for e in pool if e["track_id"] != exclude] or pool
    return random.choice(pool) if pool else None


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
_STYLE_NAMES = {
    "buddhist_meditative": "梵音冥想",
    "lofi": "Lo-Fi",
    "neoclassical": "新古典",
    "nature_ambient": "自然氛围",
    "neutral": "中性背景",
}


def _bg_display_name(entry: dict) -> str:
    """A client-facing label for a background, from its structured catalog fields.

    Grid tracks read off ``style``, special tracks off ``keyword`` — parsing the
    track_id string would be fragile (styles like ``nature_ambient`` themselves
    contain underscores). Falls back to the raw taxonomy value for a type with no
    curated name yet."""
    if entry.get("kind") == "special":
        kw = entry.get("keyword") or ""
        return _BG_NAMES.get(kw, "自然音")
    style = entry.get("style") or ""
    return _STYLE_NAMES.get(style, style or "背景音乐")


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
def random_background(
    goal: Literal["relax", "focus"] = "relax",
    type: str | None = None,
    exclude: str | None = None,
) -> dict:
    """Pick a random rendered background compatible with ``goal``; returns its id, display
    name, and file URL.

    ``type`` is an optional filter — a background **substrate** (``drone``,
    ``noise_texture``, ``percussive_with_tail``, ``field_recording``, ``melodic_instrument``)
    or the special **group** (``natural_sounds``). Omit it and the pick spans every type
    compatible with the goal; pass it to pin one type. Unknown types 400. ``exclude`` is
    the track_id already playing, so the switch button lands on a different bed while still
    respecting ``goal`` + ``type``. 404 if nothing rendered matches."""
    entry = _random_background_entry(goal, category=type, exclude=exclude)
    if entry is None:
        detail = f"no rendered background for goal={goal!r}" + (f", type={type!r}" if type else "")
        raise HTTPException(status_code=404, detail=detail)
    tid = entry["track_id"]
    return {"track_id": tid, "name": _bg_display_name(entry), "url": f"/background/{tid}.mp3"}


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


@app.get("/api/profiles")
def profiles() -> dict:
    """The mode-profile catalog for the app's selection grid (§ :mod:`bnb.profiles`).

    Authored presets read from ``assets/profiles.json`` on each request, each a card the
    client renders and, on tap, expands into a play spec (``goal`` + client-side beat
    params). ``{"profiles": [...]}`` mirrors the other list endpoints' envelope. A broken
    or missing config file 500s with the reason (rather than serving an empty grid), so a
    bad hand-edit is obvious."""
    try:
        return {"profiles": list_profiles()}
    except (OSError, ValueError) as exc:  # ValueError covers json.JSONDecodeError + validation
        raise HTTPException(status_code=500, detail=f"profiles config error: {exc}")


@app.get("/profile/{filename}")
def profile_image(filename: str):
    """A profile card's background image from ``assets/profiles/``. Filenames only — no
    path segments or ``..`` — so a request can't escape the image directory; 404 if
    absent."""
    from fastapi.responses import FileResponse

    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=404, detail="no such profile image")
    path = profiles_image_dir() / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="no such profile image")
    return FileResponse(str(path))


# Dimensions the catalog page offers as filters. Each is a flat, low-cardinality field
# on a catalog entry (§ ``bnb.assets._entry``), so the facet values are just the
# distinct values present — no separate taxonomy call needed, and a new value shows up
# in the UI as soon as one track carries it. ``tags`` is faceted separately: a track
# carries a list of them, not one value.
FACET_FIELDS = ("goal", "style", "substrate", "group", "provider")


@app.get("/api/catalog")
def catalog() -> dict:
    """The full catalog plus the distinct value of every filterable dimension.

    ``/api/backgrounds`` is the stream's track picker (id + summary only); this is the
    catalog-management view, so entries come through whole and the ``facets`` block
    lets the client build its filter controls without hardcoding taxonomy values.
    ``tagging_enabled`` rides along so the client knows up front whether the tagging
    operations are open on this service (§ :data:`TAGGING_ENABLED`).
    """
    tracks = categories.catalog()["tracks"]
    facets = {
        field: sorted({e[field] for e in tracks if e.get(field) is not None})
        for field in FACET_FIELDS
    }
    facets["tags"] = sorted({tag for e in tracks for tag in e.get("tags", [])})
    return {
        "count": len(tracks),
        "tracks": tracks,
        "facets": facets,
        "tagging_enabled": TAGGING_ENABLED,
    }


class TagRequest(BaseModel):
    """One tag applied to (or removed from) a batch of tracks — the dashboard's
    operation panel acts on a multi-track selection, so the batch is the unit."""

    track_ids: list[str] = Field(min_length=1)
    tag: str = Field(min_length=1, max_length=MAX_TAG_LENGTH)

    @field_validator("tag")
    @classmethod
    def _normalize(cls, value: str) -> str:
        """Collapse surrounding/internal whitespace, so " warm  bed " and "warm bed"
        are the same tag rather than two that look identical in the UI."""
        tag = " ".join(value.split())
        if not tag:
            raise ValueError("tag must not be blank")
        return tag


def _require_tagging() -> None:
    if not TAGGING_ENABLED:
        raise HTTPException(status_code=403, detail="tagging is disabled on this service")


def _edit_tags(req: TagRequest, *, add: bool) -> dict:
    edit = categories.add_tag if add else categories.remove_tag
    try:
        tags = edit(req.track_ids, req.tag)
    except FileNotFoundError as exc:
        # Nothing was written — the manager resolves every id before it edits any spec.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"tag": req.tag, "tags": tags}


@app.post("/api/catalog/tags")
def add_tags(req: TagRequest) -> dict:
    """Add one tag to every listed track. Idempotent — a tag a track already carries
    changes nothing. Returns each track's resulting tag list."""
    _require_tagging()
    return _edit_tags(req, add=True)


@app.delete("/api/catalog/tags")
def remove_tags(req: TagRequest) -> dict:
    """Remove one tag from every listed track (a tag it doesn't carry is a no-op), so a
    mistyped tag isn't permanent. Returns each track's resulting tag list."""
    _require_tagging()
    return _edit_tags(req, add=False)


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
    global TAGGING_ENABLED
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="bnb — binaural beat stream service")
    parser.add_argument(
        "--tagging",
        action=argparse.BooleanOptionalAction,
        default=TAGGING_ENABLED,
        help="allow the catalog tagging endpoints to write to the asset repo "
        f"(default: {'on' if TAGGING_ENABLED else 'off'}, from ${'BNB_ENABLE_TAGGING'})",
    )
    parser.add_argument(
        "--host",
        default=HOST,
        help="interface to bind; 0.0.0.0 serves on every address, so a phone on the "
        f"same LAN can reach it (default: {HOST}, from $BNB_HOST)",
    )
    args = parser.parse_args()
    TAGGING_ENABLED = args.tagging
    # Binding beyond loopback exposes an unauthenticated service; say so once rather
    # than letting an open port be a surprise.
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"bnb: binding {args.host}:{PORT} — this service has no authentication and "
            f"tagging writes are {'ON' if TAGGING_ENABLED else 'off'}. Trusted networks only.",
            file=sys.stderr,
        )
    uvicorn.run(app, host=args.host, port=PORT)


if __name__ == "__main__":
    main()
