"""Rendering of pure-sine binaural tones.

The left ear gets the carrier, the right ear gets the carrier plus the beat
frequency, so the pitch the user hears stays steady while the beat underneath it
changes. See README.md for where these constraints come from.
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
import soundfile as sf

SAMPLE_RATE = 48_000

Waveform = Literal["sine", "triangle", "square", "sawtooth"]

WAVEFORMS: tuple[Waveform, ...] = ("sine", "triangle", "square", "sawtooth")
"""Sine is the default and the only one the literature backs: harmonic-rich carriers
add spectral energy that muddies the beat percept. The others are here to be measured
against it, not to be shipped."""

CARRIER_HZ = 432.0
"""Default carrier, inside the 400-440 Hz perceptual optimum."""

MIN_CARRIER_HZ = 200.0
MAX_CARRIER_HZ = 900.0

MAX_BEAT_HZ = 40.0
"""Upper bound on Δ, set for experimentation rather than product use.

The beat percept weakens badly above ~30 Hz — the two tones start to separate into
distinct pitches rather than fusing into a beat — so nothing we ship goes near it
(all down-regulation modes sit at ≤ 14 Hz). The headroom to 40 Hz exists so the demo
portal can explore the gamma range; treat anything above ~30 as a research setting,
not a usable beat."""


def _oscillator(freq_hz: float, t: np.ndarray, waveform: Waveform) -> np.ndarray:
    """One cycle-normalised oscillator, peak amplitude 1."""
    # Phase in cycles, wrapped to [0, 1); every waveform is a function of it.
    phase = np.mod(freq_hz * t, 1.0)

    match waveform:
        case "sine":
            return np.sin(2 * np.pi * phase)
        case "triangle":
            return 4 * np.abs(phase - 0.5) - 1
        case "square":
            return np.where(phase < 0.5, 1.0, -1.0)
        case "sawtooth":
            return 2 * phase - 1
        case _:
            raise ValueError(f"unknown waveform {waveform!r}, expected one of {WAVEFORMS}")


def render_binaural(
    beat_hz: float,
    duration_s: float,
    carrier_hz: float = CARRIER_HZ,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.3,
    waveform: Waveform = "sine",
) -> np.ndarray:
    """Render a constant-frequency binaural tone as float32 stereo [n_frames, 2].

    Channel 0 (left) is a tone at ``carrier_hz``; channel 1 (right) is a tone at
    ``carrier_hz + beat_hz``. Both channels are rendered at the same amplitude so
    the image stays centred.

    ``waveform`` defaults to ``"sine"``, which is what the binaural literature uses
    and what we ship; the other shapes exist for comparison.
    """
    if waveform not in WAVEFORMS:
        raise ValueError(f"unknown waveform {waveform!r}, expected one of {WAVEFORMS}")
    if not MIN_CARRIER_HZ <= carrier_hz <= MAX_CARRIER_HZ:
        raise ValueError(
            f"carrier {carrier_hz} Hz outside usable range "
            f"{MIN_CARRIER_HZ}-{MAX_CARRIER_HZ} Hz"
        )
    if not 0 < beat_hz < MAX_BEAT_HZ:
        raise ValueError(f"beat {beat_hz} Hz outside (0, {MAX_BEAT_HZ}) Hz")
    if duration_s <= 0:
        raise ValueError(f"duration {duration_s} s must be positive")
    if not 0 < amplitude <= 1:
        raise ValueError(f"amplitude {amplitude} must be in (0, 1]")

    n_frames = round(duration_s * sample_rate)
    t = np.arange(n_frames, dtype=np.float64) / sample_rate

    left = _oscillator(carrier_hz, t, waveform)
    right = _oscillator(carrier_hz + beat_hz, t, waveform)

    return (amplitude * np.stack([left, right], axis=1)).astype(np.float32)


def write_wav(
    path: str | os.PathLike[str],
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """Write stereo float samples to a 24-bit WAV file."""
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError(f"expected stereo [n_frames, 2], got shape {samples.shape}")
    sf.write(path, samples, sample_rate, subtype="PCM_24")
