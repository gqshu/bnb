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

import time
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator

from .catalog import CategoryManager
from .stream import Beat, StreamEngine, to_int16_bytes, wav_stream_header
from .tone import (
    CARRIER_HZ,
    ISOCHRONIC_MAX_CARRIER_HZ,
    ISOCHRONIC_MIN_CARRIER_HZ,
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

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title="bnb — binaural beat stream")
engine = StreamEngine()
categories = CategoryManager()


class BeatSpec(BaseModel):
    """A binaural/monaural/isochronic beat. Bounds come straight from the
    audio-design constraints.

    ``beat_hz`` may be exactly 0: both ears then get the same carrier and no beat
    exists, which is the acoustically-matched **sham** condition (carrier audible,
    Δ absent) the EEG pilot compares against. The engine renders it natively —
    identical phase on both channels. Note this is deliberately looser than
    ``tone.render_binaural``, which still rejects 0 because a 0 Hz *binaural* WAV
    is not a binaural tone at all; the sham only exists as a live stream state.

    ``carrier_hz``'s valid range depends on ``mode`` — isochronic's usable carrier
    range (100-500 Hz) sits lower than the other modes' (200-900 Hz), so it's
    checked in ``_carrier_in_range`` rather than as a static ``Field`` bound.
    ``depth``/``duty``/``ramp_ms`` only apply to ``isochronic``.
    """

    carrier_hz: float = Field(CARRIER_HZ)
    beat_hz: float = Field(10.0, ge=0.0, le=MAX_BEAT_HZ)
    volume: float = Field(0.5, ge=0.0, le=1.0)
    waveform: Waveform = "sine"
    mode: Literal["dichotic", "diotic", "monaural", "isochronic"] = "dichotic"
    depth: float = Field(1.0, ge=MIN_DEPTH, le=MAX_DEPTH)
    duty: float = Field(0.5, ge=MIN_DUTY, le=MAX_DUTY)
    ramp_ms: float = Field(5.0, ge=MIN_RAMP_MS, le=MAX_RAMP_MS)

    @model_validator(mode="after")
    def _carrier_in_range(self) -> "BeatSpec":
        if self.mode == "isochronic":
            lo, hi = ISOCHRONIC_MIN_CARRIER_HZ, ISOCHRONIC_MAX_CARRIER_HZ
        else:
            lo, hi = MIN_CARRIER_HZ, MAX_CARRIER_HZ
        if not lo <= self.carrier_hz <= hi:
            raise ValueError(
                f"carrier {self.carrier_hz} Hz outside usable range {lo}-{hi} Hz for mode {self.mode!r}"
            )
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
        )


class StartRequest(BaseModel):
    beat: BeatSpec | None = None
    background_id: str | None = None
    background_volume: float = Field(1.0, ge=0.0, le=1.0)


class SpecUpdate(BaseModel):
    """A partial live update. Only the fields present in the body are applied;
    ``beat: null`` removes the beat, ``background_id: null`` removes the background."""

    beat: BeatSpec | None = None
    background_id: str | None = None
    background_volume: float | None = Field(None, ge=0.0, le=1.0)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB_DIR / "index.html").read_text()


@app.get("/api/stream")
def get_state() -> dict:
    return engine.snapshot()


@app.post("/api/stream/start")
def start(req: StartRequest) -> dict:
    try:
        engine.start(
            beat=req.beat.to_engine() if req.beat else None,
            background_id=req.background_id,
            background_volume=req.background_volume,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return engine.snapshot()


@app.post("/api/stream/stop")
def stop() -> dict:
    engine.stop()
    return engine.snapshot()


@app.patch("/api/stream/spec")
def update_spec(update: SpecUpdate) -> dict:
    """Applies whichever fields are present as one atomic update (see
    :meth:`StreamEngine.update`) — a request that changes ``beat`` and
    ``background_id`` together is validated by what they end up as, not by
    which field happens to be applied first."""
    fields = update.model_fields_set
    kwargs: dict[str, Any] = {}
    if "beat" in fields:
        kwargs["beat"] = update.beat.to_engine() if update.beat else None
    if "background_id" in fields:
        kwargs["background_id"] = update.background_id
    if "background_volume" in fields and update.background_volume is not None:
        kwargs["background_volume"] = update.background_volume
    try:
        engine.update(**kwargs)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return engine.snapshot()


@app.get("/api/backgrounds")
def backgrounds() -> list[dict]:
    return [
        {"track_id": e["track_id"], "summary": e["summary"], "rendered": e["rendered"]}
        for e in categories.search()
    ]


@app.get("/stream.wav")
def stream_wav() -> StreamingResponse:
    """Open-ended WAV of the live mix, paced to real time so edits stay near-live."""

    def generate():
        yield wav_stream_header(engine.sample_rate)
        n = int(engine.sample_rate * CHUNK_SECONDS)
        deadline = time.monotonic()
        while engine.running:
            yield to_int16_bytes(engine.read(n))
            deadline += CHUNK_SECONDS
            behind = deadline - time.monotonic()
            if behind > 0:
                time.sleep(behind)

    return StreamingResponse(generate(), media_type="audio/wav")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT)


if __name__ == "__main__":
    main()
