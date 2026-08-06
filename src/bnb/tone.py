"""Offline renderers for every stimulus modality.

Four modalities, differing in where the beat physically lives:

- **binaural** — left ear carrier, right ear carrier + beat. The beat exists only
  across the ears; the pitch stays steady while the beat underneath it changes.
- **monaural** — the symmetric tone pair summed before playback, so the envelope is
  acoustically real in each ear.
- **isochronic** — one carrier gated on/off at the beat rate. The measurement-probe
  path: no background, by construction (doc §5).
- **AM music** — a music bed used as the carrier and modulated at the beat rate.
  Product audio only, and the one modality that *requires* a background (doc §6).

See README.md and docs/Monaural_and_Isochronic_Beats_Implementation.md for where
these constraints come from.
"""

from __future__ import annotations

import os
from pathlib import Path
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

AM_DEPTH = 0.5
"""Default modulation depth for AM music.

Depth is the drive/pleasantness dial: at 1.0 the music drops to silence every cycle
(maximum cortical drive, but audibly pulsing and tiring over a 30-minute session),
at 0.0 it's untouched music. 0.5 is the "shallow AM" middle the product doc argues
for (bnb_product2.md §2) — a real physical envelope the cortex can track, still
musical enough to sit under a wind-down."""

AM_MODULATORS: tuple[str, ...] = ("sine", "gate")
"""Modulator shapes. ``sine`` is smooth and least damaging to the music's timbre;
``gate`` is the isochronic trapezoid (``duty``/``ramp_ms``), which drives harder but
stamps a more audible pulse onto broadband content."""

MAX_AM_BEAT_HZ = 60.0
"""Upper bound on the AM rate — deliberately higher than ``MAX_BEAT_HZ``.

The ~30 Hz ceiling on beats comes from binaural fusion: past it the ear hears two
pitches instead of one beating tone. AM music has no such limit, because the envelope
is physically present in the signal rather than reconstructed from two carriers. That
makes 40 Hz gamma a first-class target here (doc §7's most-likely-detectable probe),
with headroom above it for experimentation."""

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


def load_background(source: str | os.PathLike[str], target_sample_rate: int | None = None):
    """Load a background bed as ``(stereo float32 [n, 2], sample_rate)``.

    ``source`` is either a path to an audio file or a ``track_id`` in the asset repo
    (``assets/tracks/``), so the CLI can name a rendered library track directly.
    Mono sources are duplicated to stereo. Native rate is kept unless
    ``target_sample_rate`` is given — the masters are 44.1 kHz and resampling music
    to the 48 kHz tone rate costs quality for nothing.
    """
    from bnb import assets  # local: keeps the pure-synthesis path free of the asset repo

    path = Path(source)
    if not path.exists():
        found = assets.find_track(str(source))
        if found is None:
            raise FileNotFoundError(f"no audio file or asset track named {source!r}")
        path = found

    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        raise ValueError(f"expected mono or stereo background, got {data.shape[1]} channels")

    if target_sample_rate is not None and target_sample_rate != sample_rate:
        n_to = round(data.shape[0] * target_sample_rate / sample_rate)
        x_old = np.linspace(0.0, 1.0, data.shape[0], endpoint=False)
        x_new = np.linspace(0.0, 1.0, n_to, endpoint=False)
        data = np.stack([np.interp(x_new, x_old, data[:, ch]) for ch in range(2)], axis=1)
        sample_rate = target_sample_rate

    return data.astype(np.float32), sample_rate


def _fit_length(background: np.ndarray, n_frames: int) -> np.ndarray:
    """Loop or trim the bed to exactly ``n_frames``.

    The library masters are rendered loopable (``assets/specs/*.json``), so tiling is
    the intended way to fill a session longer than one track.
    """
    if background.shape[0] >= n_frames:
        return background[:n_frames]
    repeats = -(-n_frames // background.shape[0])  # ceil
    return np.tile(background, (repeats, 1))[:n_frames]


def render_am_music(
    background: np.ndarray,
    beat_hz: float,
    duration_s: float | None = None,
    sample_rate: int = SAMPLE_RATE,
    depth: float = AM_DEPTH,
    modulator: str = "sine",
    duty: float = 0.5,
    ramp_ms: float = 5.0,
) -> np.ndarray:
    """Amplitude-modulate a music bed at ``beat_hz``; float32 stereo [n_frames, 2].

    Doc §6: the music *is* the carrier, so there is no ``carrier_hz`` here. Gating
    broadband content stamps the beat onto every component at once, which the cortex
    tracks — but it rides on the music's own irregular envelope, so the drive is
    muddier than an isochronic tone's clean carrier-plus-sidebands. That makes this a
    **product-audio** modality only: never use it as a measurement probe (doc §6),
    which is what :func:`render_isochronic` is for.

    Unlike the tone renderers this one needs a bed to modulate, so a background is a
    required argument rather than an option — an AM-music render with no music is a
    contradiction, and the type system should say so.

    The envelope spans ``[1 - depth, 1]``, so the output can never exceed the input's
    peak — no clipping, but mean level drops by roughly ``depth / 2``. Level-match
    before A/B-ing against unmodulated music.
    """
    if modulator not in AM_MODULATORS:
        raise ValueError(f"unknown modulator {modulator!r}, expected one of {AM_MODULATORS}")
    if not 0 < beat_hz <= MAX_AM_BEAT_HZ:
        raise ValueError(f"beat {beat_hz} Hz outside (0, {MAX_AM_BEAT_HZ}] Hz")
    if not MIN_DEPTH <= depth <= MAX_DEPTH:
        raise ValueError(f"depth {depth} outside [{MIN_DEPTH}, {MAX_DEPTH}]")
    if not MIN_DUTY <= duty <= MAX_DUTY:
        raise ValueError(f"duty {duty} outside [{MIN_DUTY}, {MAX_DUTY}]")
    if not MIN_RAMP_MS <= ramp_ms <= MAX_RAMP_MS:
        raise ValueError(f"ramp_ms {ramp_ms} outside [{MIN_RAMP_MS}, {MAX_RAMP_MS}]")
    if background.ndim != 2 or background.shape[1] != 2:
        raise ValueError(f"expected stereo background [n_frames, 2], got shape {background.shape}")
    if duration_s is not None and duration_s <= 0:
        raise ValueError(f"duration {duration_s} s must be positive")

    n_frames = background.shape[0] if duration_s is None else round(duration_s * sample_rate)
    bed = _fit_length(background, n_frames).astype(np.float64)
    t = np.arange(n_frames, dtype=np.float64) / sample_rate
    phi = np.mod(t * beat_hz, 1.0)

    if modulator == "sine":
        # Starts at the trough like the gate does, so both shapes share a phase origin.
        unit = 0.5 * (1 - np.cos(2 * np.pi * phi))
    else:
        ramp_frac = min((ramp_ms / 1000.0) * beat_hz, duty / 2, (1 - duty) / 2)
        unit = _gate_envelope(phi, duty, ramp_frac)

    env = (1 - depth) + depth * unit
    return (bed * env[:, None]).astype(np.float32)


def write_wav(
    path: str | os.PathLike[str],
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """Write stereo float samples to a 24-bit WAV file."""
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError(f"expected stereo [n_frames, 2], got shape {samples.shape}")
    sf.write(path, samples, sample_rate, subtype="PCM_24")
