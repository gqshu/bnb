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

ISOCHRONIC_CARRIER_HZ = 250.0
"""Default carrier for isochronic tones — low, per doc guidance, for strong phase-locking."""

ISOCHRONIC_MIN_CARRIER_HZ = 100.0
ISOCHRONIC_MAX_CARRIER_HZ = 500.0
"""Isochronic's usable carrier range sits lower than binaural/monaural's, since the
gate (not the carrier pitch) carries the entrainment; a low carrier keeps phase-locking
strong."""

MIN_DEPTH = 0.0
MAX_DEPTH = 1.0
MIN_DUTY = 0.1
MAX_DUTY = 0.9
MIN_RAMP_MS = 2.0
MAX_RAMP_MS = 10.0

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


def _validate_tone_params(
    carrier_hz: float,
    beat_hz: float,
    duration_s: float,
    amplitude: float,
    waveform: Waveform,
    *,
    carrier_range: tuple[float, float] = (MIN_CARRIER_HZ, MAX_CARRIER_HZ),
) -> None:
    """Shared bound checks for the binaural/monaural offline renderers."""
    if waveform not in WAVEFORMS:
        raise ValueError(f"unknown waveform {waveform!r}, expected one of {WAVEFORMS}")
    lo, hi = carrier_range
    if not lo <= carrier_hz <= hi:
        raise ValueError(f"carrier {carrier_hz} Hz outside usable range {lo}-{hi} Hz")
    if not 0 < beat_hz < MAX_BEAT_HZ:
        raise ValueError(f"beat {beat_hz} Hz outside (0, {MAX_BEAT_HZ}) Hz")
    if duration_s <= 0:
        raise ValueError(f"duration {duration_s} s must be positive")
    if not 0 < amplitude <= 1:
        raise ValueError(f"amplitude {amplitude} must be in (0, 1]")


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
    _validate_tone_params(carrier_hz, beat_hz, duration_s, amplitude, waveform)

    n_frames = round(duration_s * sample_rate)
    t = np.arange(n_frames, dtype=np.float64) / sample_rate

    left = _oscillator(carrier_hz, t, waveform)
    right = _oscillator(carrier_hz + beat_hz, t, waveform)

    return (amplitude * np.stack([left, right], axis=1)).astype(np.float32)


def render_monaural(
    beat_hz: float,
    duration_s: float,
    carrier_hz: float = CARRIER_HZ,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.3,
    waveform: Waveform = "sine",
) -> np.ndarray:
    """Render a constant-frequency monaural tone as float32 stereo [n_frames, 2].

    Two tones symmetric around ``carrier_hz`` (``carrier_hz ∓ beat_hz/2``) are
    summed *before* playback and sent identically to both channels, so the beat is
    a real acoustic amplitude envelope rather than a binaural (neural) construct.
    """
    _validate_tone_params(carrier_hz, beat_hz, duration_s, amplitude, waveform)

    n_frames = round(duration_s * sample_rate)
    t = np.arange(n_frames, dtype=np.float64) / sample_rate

    tone_low = _oscillator(carrier_hz - beat_hz / 2, t, waveform)
    tone_high = _oscillator(carrier_hz + beat_hz / 2, t, waveform)
    mono = 0.5 * (tone_low + tone_high)  # halve to avoid clipping when the tones align

    return (amplitude * np.stack([mono, mono], axis=1)).astype(np.float32)


def _gate_envelope(phi: np.ndarray, duty: float, ramp_frac: float) -> np.ndarray:
    """A trapezoid gate in [0, 1], on for ``duty`` of each cycle with raised-cosine
    on/off ramps, evaluated at fractional cycle positions ``phi`` in [0, 1).

    ``ramp_frac`` (the ramp's length as a fraction of one cycle) must already be
    clamped by the caller to ``min(ramp_frac, duty / 2, (1 - duty) / 2)`` so the
    rise and fall edges never overlap or spill past the on/off windows. A hard
    (``ramp_frac <= 0``) gate is supported for comparison but should never ship
    (see doc §4.4 — it clicks and splatters harmonics of the beat frequency).
    """
    if ramp_frac <= 0:
        return np.where(phi < duty, 1.0, 0.0).astype(np.float32)

    rise = 0.5 - 0.5 * np.cos(np.pi * np.clip(phi / ramp_frac, 0.0, 1.0))
    fall = 0.5 - 0.5 * np.cos(np.pi * np.clip((duty - phi) / ramp_frac, 0.0, 1.0))
    gate = np.select(
        [phi < ramp_frac, phi < duty - ramp_frac, phi < duty],
        [rise, np.ones_like(phi), fall],
        default=0.0,
    )
    return gate.astype(np.float32)


def render_isochronic(
    beat_hz: float,
    duration_s: float,
    carrier_hz: float = ISOCHRONIC_CARRIER_HZ,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.3,
    waveform: Waveform = "sine",
    depth: float = 1.0,
    duty: float = 0.5,
    ramp_ms: float = 5.0,
) -> np.ndarray:
    """Render a constant-rate isochronic tone as float32 stereo [n_frames, 2].

    A single carrier is amplitude-gated on/off ``beat_hz`` times per second, with
    mandatory raised-cosine edges (``ramp_ms``) so the gate stays spectrally clean
    (carrier ± ``beat_hz`` sidebands only, no harmonic splatter — doc §4.4). This is
    the no-background measurement-probe path (doc §5): there is deliberately no
    background parameter here, so a probe render can never be constructed with one.
    """
    _validate_tone_params(
        carrier_hz,
        beat_hz,
        duration_s,
        amplitude,
        waveform,
        carrier_range=(ISOCHRONIC_MIN_CARRIER_HZ, ISOCHRONIC_MAX_CARRIER_HZ),
    )
    if not MIN_DEPTH <= depth <= MAX_DEPTH:
        raise ValueError(f"depth {depth} outside [{MIN_DEPTH}, {MAX_DEPTH}]")
    if not MIN_DUTY <= duty <= MAX_DUTY:
        raise ValueError(f"duty {duty} outside [{MIN_DUTY}, {MAX_DUTY}]")
    if not MIN_RAMP_MS <= ramp_ms <= MAX_RAMP_MS:
        raise ValueError(f"ramp_ms {ramp_ms} outside [{MIN_RAMP_MS}, {MAX_RAMP_MS}]")

    n_frames = round(duration_s * sample_rate)
    t = np.arange(n_frames, dtype=np.float64) / sample_rate

    tone = _oscillator(carrier_hz, t, waveform)
    phi = np.mod(t * beat_hz, 1.0)
    ramp_frac = min((ramp_ms / 1000.0) * beat_hz, duty / 2, (1 - duty) / 2)
    gate = _gate_envelope(phi, duty, ramp_frac)
    env = (1 - depth) + depth * gate
    mono = tone * env

    return (amplitude * np.stack([mono, mono], axis=1)).astype(np.float32)


def write_wav(
    path: str | os.PathLike[str],
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """Write stereo float samples to a 24-bit WAV file."""
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError(f"expected stereo [n_frames, 2], got shape {samples.shape}")
    sf.write(path, samples, sample_rate, subtype="PCM_24")
