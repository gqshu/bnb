"""The live binaural-beat stream engine.

A single, mutable, thread-safe audio source that a request thread renders in
small chunks while the API threads mutate its settings underneath it. The point
of doing this server-side (rather than generating tones in the browser) is that
the *backend* owns the live stream, per the service design.

Two things make live edits sound clean:

- **Phase continuity.** We carry each channel's phase across chunks and step it
  per sample, so changing the carrier or beat frequency doesn't restart the
  waveform and click.
- **Amplitude ramping.** Volume changes are linearly ramped across a chunk
  instead of jumping, which avoids zipper noise on the sliders.

An optional background track (a rendered asset, looked up by ``track_id`` in the
asset repo) is looped and mixed underneath the beat. The stream runs at 44.1 kHz
to match the background masters (``pcm_44100``), so no resampling is needed in the
common case; a mismatched background is linearly resampled on load.
"""

from __future__ import annotations

import struct
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any

import numpy as np
import soundfile as sf

from . import assets
from .tone import CARRIER_HZ

STREAM_SAMPLE_RATE = 44_100
TWO_PI = 2.0 * np.pi

BACKGROUND_FADE_SECONDS = 0.6  # crossfade length when the background track changes
LIMITER_THRESHOLD = 0.8  # mix stays linear below this; peaks above saturate softly


@dataclass
class Beat:
    """The live binaural-beat spec: left ear = carrier, right ear = carrier + beat."""

    carrier_hz: float = CARRIER_HZ
    beat_hz: float = 10.0
    volume: float = 0.3
    waveform: str = "sine"


def _oscillator(phase_rad: np.ndarray, waveform: str) -> np.ndarray:
    """Evaluate a unit-amplitude oscillator from an absolute phase (radians)."""
    frac = np.mod(phase_rad / TWO_PI, 1.0)  # cycle position in [0, 1)
    match waveform:
        case "sine":
            return np.sin(phase_rad)
        case "triangle":
            return 4.0 * np.abs(frac - 0.5) - 1.0
        case "square":
            return np.where(frac < 0.5, 1.0, -1.0)
        case "sawtooth":
            return 2.0 * frac - 1.0
        case _:
            raise ValueError(f"unknown waveform {waveform!r}")


def _soft_limit(x: np.ndarray, threshold: float = LIMITER_THRESHOLD) -> np.ndarray:
    """Soft-knee limiter. Below ``threshold`` it's the identity (so a pure-sine beat
    stays pure, per the audio-design constraint); above it, peaks saturate with a
    tanh knee instead of hard-clipping into harsh inharmonic distortion."""
    amp = np.abs(x)
    knee = threshold + (1.0 - threshold) * np.tanh((amp - threshold) / (1.0 - threshold))
    factor = np.where(amp > threshold, knee / np.maximum(amp, 1e-9), 1.0)
    return x * factor


def _resample(data: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Linear-interpolate stereo float audio to a new sample rate."""
    if sr_from == sr_to:
        return data
    n_from = data.shape[0]
    n_to = round(n_from * sr_to / sr_from)
    x_old = np.linspace(0.0, 1.0, n_from, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_to, endpoint=False)
    out = np.empty((n_to, data.shape[1]), dtype=np.float32)
    for ch in range(data.shape[1]):
        out[:, ch] = np.interp(x_new, x_old, data[:, ch])
    return out


class StreamEngine:
    """One live stream. Guarded by a lock so API edits and rendering interleave safely."""

    def __init__(self, sample_rate: int = STREAM_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._lock = Lock()
        self.running = False
        self.beat: Beat | None = None
        self.background_id: str | None = None
        self.background_volume = 1.0
        # Per-channel phase and ramped amplitude carried across chunks.
        self._left_phase = 0.0
        self._right_phase = 0.0
        self._amp = 0.0
        # Background: the current track plus (during a switch) the outgoing one it
        # crossfades from. ``_fade`` is how faded-in the current track is, 0..1.
        self._bg: np.ndarray | None = None
        self._bg_pos = 0
        self._bg_out: np.ndarray | None = None
        self._bg_out_pos = 0
        self._fade = 1.0
        self._fade_frames = max(1, int(BACKGROUND_FADE_SECONDS * sample_rate))

    # --- control (called from API threads) ---------------------------------

    def start(self, *, beat: Beat | None, background_id: str | None, background_volume: float) -> None:
        with self._lock:
            self.beat = beat
            self.background_volume = background_volume
            self._left_phase = self._right_phase = 0.0
            self._amp = 0.0  # fade in over the first chunk
            self._switch_background(background_id)
            self.running = True

    def stop(self) -> None:
        with self._lock:
            self.running = False

    def set_beat(self, beat: Beat | None) -> None:
        with self._lock:
            self.beat = beat

    def set_background(self, background_id: str | None) -> None:
        with self._lock:
            self._switch_background(background_id)

    def set_background_volume(self, volume: float) -> None:
        with self._lock:
            self.background_volume = volume

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self.running,
                "beat": asdict(self.beat) if self.beat else None,
                "background_id": self.background_id,
                "background_volume": self.background_volume,
                "sample_rate": self.sample_rate,
            }

    def _read_asset(self, background_id: str) -> np.ndarray:
        """Load a rendered background asset as looped stereo at the stream rate."""
        path = assets.find_track(background_id)
        if path is None:
            raise FileNotFoundError(f"no rendered audio for background {background_id!r}")
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        return _resample(data, sr, self.sample_rate)

    def _switch_background(self, background_id: str | None) -> None:
        """Start a crossfade to a new background (or to silence). Caller holds the lock.

        The load happens first, so a bad id raises before any state changes."""
        new = None if background_id is None else self._read_asset(background_id)
        if new is None and self._bg is None and self._bg_out is None:
            self.background_id = None
            return
        # The currently-audible track becomes the outgoing one we fade out from.
        self._bg_out, self._bg_out_pos = self._bg, self._bg_pos
        self._bg, self._bg_pos = new, 0
        self._fade = 0.0
        self.background_id = background_id

    # --- rendering (called from the streaming request thread) --------------

    def read(self, n_frames: int) -> np.ndarray:
        """Render the next ``n_frames`` as float32 stereo in [-1, 1]."""
        with self._lock:
            out = np.zeros((n_frames, 2), dtype=np.float32)
            if self.beat is not None:
                out += self._render_beat(n_frames)
            if self._bg is not None or self._bg_out is not None:
                out += self._render_background(n_frames) * self.background_volume
            return _soft_limit(out)

    def _render_beat(self, n: int) -> np.ndarray:
        beat = self.beat
        assert beat is not None
        step = np.arange(1, n + 1, dtype=np.float64) / self.sample_rate
        left_phase = self._left_phase + TWO_PI * beat.carrier_hz * step
        right_phase = self._right_phase + TWO_PI * (beat.carrier_hz + beat.beat_hz) * step
        self._left_phase = float(left_phase[-1] % TWO_PI)
        self._right_phase = float(right_phase[-1] % TWO_PI)

        stereo = np.stack(
            [_oscillator(left_phase, beat.waveform), _oscillator(right_phase, beat.waveform)],
            axis=1,
        ).astype(np.float32)
        ramp = np.linspace(self._amp, beat.volume, n, dtype=np.float32)
        self._amp = beat.volume
        return stereo * ramp[:, None]

    @staticmethod
    def _take(buf: np.ndarray, pos: int, n: int) -> tuple[np.ndarray, int]:
        """Read ``n`` frames from a looped buffer, returning them and the new position."""
        idx = (pos + np.arange(n)) % buf.shape[0]
        return buf[idx], int((pos + n) % buf.shape[0])

    def _render_background(self, n: int) -> np.ndarray:
        out = np.zeros((n, 2), dtype=np.float32)
        fade_end = min(1.0, self._fade + n / self._fade_frames)
        gain_new = np.linspace(self._fade, fade_end, n, dtype=np.float32)[:, None]
        if self._bg is not None:
            seg, self._bg_pos = self._take(self._bg, self._bg_pos, n)
            out += seg * gain_new
        if self._bg_out is not None:
            seg, self._bg_out_pos = self._take(self._bg_out, self._bg_out_pos, n)
            out += seg * (1.0 - gain_new)
        self._fade = fade_end
        if self._fade >= 1.0:  # crossfade finished; drop the outgoing track
            self._bg_out = None
        return out


def wav_stream_header(sample_rate: int, channels: int = 2, bits: int = 16) -> bytes:
    """A WAV header for an open-ended stream (sizes maxed out, not real lengths)."""
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return (
        b"RIFF"
        + struct.pack("<I", 0xFFFFFFFF)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits)
        + b"data"
        + struct.pack("<I", 0xFFFFFFFF)
    )


def to_int16_bytes(frames: np.ndarray) -> bytes:
    """Convert float32 stereo [-1, 1] to interleaved little-endian 16-bit PCM."""
    clipped = np.clip(frames, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()
