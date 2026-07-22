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
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import assets
from .stream import Beat, StreamEngine, to_int16_bytes, wav_stream_header
from .tone import CARRIER_HZ, MAX_BEAT_HZ, MAX_CARRIER_HZ, MIN_CARRIER_HZ, Waveform

PORT = 8000
CHUNK_SECONDS = 0.2  # render granularity; also the live-edit latency floor

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title="bnb — binaural beat stream")
engine = StreamEngine()


class BeatSpec(BaseModel):
    """A binaural beat. Bounds come straight from the audio-design constraints.

    ``beat_hz`` may be exactly 0: both ears then get the same carrier and no beat
    exists, which is the acoustically-matched **sham** condition (carrier audible,
    Δ absent) the EEG pilot compares against. The engine renders it natively —
    identical phase on both channels. Note this is deliberately looser than
    ``tone.render_binaural``, which still rejects 0 because a 0 Hz *binaural* WAV
    is not a binaural tone at all; the sham only exists as a live stream state.
    """

    carrier_hz: float = Field(CARRIER_HZ, ge=MIN_CARRIER_HZ, le=MAX_CARRIER_HZ)
    beat_hz: float = Field(10.0, ge=0.0, le=MAX_BEAT_HZ)
    volume: float = Field(0.5, ge=0.0, le=1.0)
    waveform: Waveform = "sine"
    mode: Literal["dichotic", "diotic"] = "dichotic"

    def to_engine(self) -> Beat:
        return Beat(self.carrier_hz, self.beat_hz, self.volume, self.waveform, self.mode)


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


def _set_background_or_400(background_id: str | None) -> None:
    try:
        engine.set_background(background_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return engine.snapshot()


@app.post("/api/stream/stop")
def stop() -> dict:
    engine.stop()
    return engine.snapshot()


@app.patch("/api/stream/spec")
def update_spec(update: SpecUpdate) -> dict:
    fields = update.model_fields_set
    if "beat" in fields:
        engine.set_beat(update.beat.to_engine() if update.beat else None)
    if "background_id" in fields:
        _set_background_or_400(update.background_id)
    if "background_volume" in fields and update.background_volume is not None:
        engine.set_background_volume(update.background_volume)
    return engine.snapshot()


@app.get("/api/backgrounds")
def backgrounds() -> list[dict]:
    out = []
    for track_id in assets.list_specs():
        entry = assets.catalog_entry(assets.load_spec(track_id))
        out.append({"track_id": entry["track_id"], "summary": entry["summary"], "rendered": entry["rendered"]})
    return out


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
