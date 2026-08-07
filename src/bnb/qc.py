"""Integrity checks for rendered background tracks.

Generative engines fail in a handful of blunt, mechanical ways — a render comes back
silent, or clipped to mush, or as flat hiss, or as a dead constant buffer. Those are
cheap to catch numerically and expensive to catch by ear across a growing library, so
this module scores a file and says whether it is worth listening to.

What this is *not*: a judgement of whether a track is good. Everything here is a
floor, not a taste test. The library is deliberately static and quiet (drones, very
sparse textures — docs/background_music.md §2), so the checks are tuned to pass a
calm ambient bed and only flag renders that are broken as audio.

Two severities, because they call for different actions:

``fail``  unusable — silent, corrupt, clipped, or a dead constant buffer. Re-render.
``warn``  playable but suspect — quiet, hissy, or DC-offset. Listen before shipping it
          into the library.

Thresholds are calibrated against the 25-track ElevenLabs library rather than picked
from theory. On that set the healthy tracks measure: RMS -18.5 dBFS median (quietest
healthy track -36.7), level variation 0.84-10.4 dB, spectral flatness ≤ 0.021. Every
threshold below sits well outside that envelope, so a normal ambient bed passes
without argument. It found three genuinely dead renders (~-53 dBFS RMS, peaks under
0.05) that had been sitting in the library unnoticed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

# Analysis window. 50 ms is long enough for a stable RMS/spectrum estimate and short
# enough that a few seconds of dropout still moves the frame statistics.
FRAME_S = 0.05

SILENCE_DBFS = -60.0
"""A frame quieter than this counts as silence. Below roughly -60 dBFS nothing in a
relaxation bed is audible over a room's noise floor."""

# --- fail thresholds: the track is unusable -------------------------------------
MIN_DURATION_S = 5.0
MIN_PEAK = 1e-4          # digital silence, or so close it makes no difference
MIN_RMS_DBFS = -50.0     # inaudible at any sane playback gain
MAX_SILENT_FRACTION = 0.5
MAX_CLIPPED_FRACTION = 0.01
MIN_LEVEL_VARIATION_DB = 0.05
"""Frame-level RMS spread below this means the signal never changes at all — a stuck
buffer or a pure test tone, not music. Real drones still breathe by a few tenths of
a dB, so this floor sits far below anything musical."""

# --- warn thresholds: playable, but look at it ----------------------------------
QUIET_RMS_DBFS = -35.0
WARN_CLIPPED_FRACTION = 1e-4
MAX_DC_OFFSET = 0.01
NOISY_FLATNESS = 0.35
"""Spectral flatness (Wiener entropy): 0 is a pure tone, 1 is white noise. Ambient
beds with any tonal content sit well below this; a render that lands above it is hiss
rather than music."""

# Deliberately not checked: the loop seam (a level step from a track's end back to its
# start). It looks like an obvious defect for a looping library, but StreamEngine
# crossfades every restart into the track's own head (stream.py, _restart_current), so
# the step is never audible. Measured on the current library it would have flagged 13
# of 25 good tracks — a check that fires on half a healthy library trains people to
# ignore the tool.


@dataclass
class TrackReport:
    """One track's measurements plus the verdict they add up to."""

    path: Path
    ok: bool = True
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        if self.failures:
            return "fail"
        return "warn" if self.warnings else "ok"

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "verdict": self.verdict,
            "failures": self.failures,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


def _dbfs(x: float) -> float:
    return 20 * math.log10(x) if x > 0 else -math.inf


def _frame_rms(mono: np.ndarray, frame_len: int) -> np.ndarray:
    """RMS per non-overlapping frame; the tail remainder is dropped."""
    n = (len(mono) // frame_len) * frame_len
    if n == 0:
        return np.array([np.sqrt(np.mean(mono**2))]) if len(mono) else np.array([0.0])
    frames = mono[:n].reshape(-1, frame_len)
    return np.sqrt(np.mean(frames**2, axis=1))


def _spectral_flatness(mono: np.ndarray, frame_len: int) -> float:
    """Mean Wiener entropy over frames: 0 = pure tone, 1 = white noise.

    Silent frames are skipped — their spectrum is numerically meaningless and would
    drag the average toward whatever the epsilon floor implies.
    """
    n = (len(mono) // frame_len) * frame_len
    if n == 0:
        return 0.0
    frames = mono[:n].reshape(-1, frame_len)
    window = np.hanning(frame_len)
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    power = power[:, 1:]  # drop DC, which would otherwise dominate the geometric mean

    loud = power.sum(axis=1) > 1e-12
    if not loud.any():
        return 0.0
    power = power[loud] + 1e-20
    geometric = np.exp(np.mean(np.log(power), axis=1))
    arithmetic = np.mean(power, axis=1)
    return float(np.mean(geometric / arithmetic))


def check_track(path: Path | str, *, min_duration_s: float = MIN_DURATION_S) -> TrackReport:
    """Measure one audio file and decide whether it is listenable."""
    path = Path(path)
    report = TrackReport(path=path)

    try:
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # unreadable, truncated, or not audio at all
        report.failures.append(f"unreadable: {exc}")
        report.ok = False
        return report

    n_frames, channels = audio.shape
    duration_s = n_frames / sample_rate if sample_rate else 0.0
    report.metrics.update(
        duration_s=round(duration_s, 2), sample_rate=sample_rate, channels=channels
    )

    if n_frames == 0:
        report.failures.append("empty file (0 frames)")
        report.ok = False
        return report
    if not np.isfinite(audio).all():
        report.failures.append("contains NaN or Inf samples")
        report.ok = False
        return report

    mono = audio.mean(axis=1).astype(np.float64)
    frame_len = max(1, int(FRAME_S * sample_rate))

    peak = float(np.abs(audio).max())
    rms = float(np.sqrt(np.mean(mono**2)))
    frame_rms = _frame_rms(mono, frame_len)
    frame_db = np.array([_dbfs(v) for v in frame_rms])
    audible = np.isfinite(frame_db) & (frame_db > SILENCE_DBFS)

    silent_fraction = float(1.0 - audible.mean())
    clipped_fraction = float(np.mean(np.abs(audio) >= 0.999))
    dc_offset = float(np.abs(mono.mean()))
    # Spread of the *audible* frames only: leading silence would otherwise read as
    # variation, making a half-dead render look lively.
    level_variation_db = float(np.std(frame_db[audible])) if audible.any() else 0.0
    flatness = _spectral_flatness(mono, frame_len)

    report.metrics.update(
        peak=round(peak, 4),
        rms_dbfs=round(_dbfs(rms), 1) if rms > 0 else None,
        silent_fraction=round(silent_fraction, 3),
        clipped_fraction=round(clipped_fraction, 5),
        dc_offset=round(dc_offset, 4),
        level_variation_db=round(level_variation_db, 2),
        spectral_flatness=round(flatness, 3),
    )

    if duration_s < min_duration_s:
        report.failures.append(f"too short: {duration_s:.1f}s < {min_duration_s:g}s")
    if peak < MIN_PEAK:
        report.failures.append("silent: no signal above the noise floor")
    elif rms > 0 and _dbfs(rms) < MIN_RMS_DBFS:
        report.failures.append(f"inaudible: {_dbfs(rms):.1f} dBFS RMS")
    if silent_fraction > MAX_SILENT_FRACTION:
        report.failures.append(f"mostly silence: {silent_fraction:.0%} of frames")
    if clipped_fraction > MAX_CLIPPED_FRACTION:
        report.failures.append(f"clipped: {clipped_fraction:.1%} of samples at full scale")
    if peak >= MIN_PEAK and level_variation_db < MIN_LEVEL_VARIATION_DB:
        report.failures.append(f"dead constant level ({level_variation_db:.2f} dB spread)")

    if rms > 0 and MIN_RMS_DBFS <= _dbfs(rms) < QUIET_RMS_DBFS:
        report.warnings.append(f"quiet: {_dbfs(rms):.1f} dBFS RMS")
    if WARN_CLIPPED_FRACTION < clipped_fraction <= MAX_CLIPPED_FRACTION:
        report.warnings.append(f"some clipping: {clipped_fraction:.2%} of samples")
    if dc_offset > MAX_DC_OFFSET:
        report.warnings.append(f"DC offset {dc_offset:.3f}")
    if flatness > NOISY_FLATNESS:
        report.warnings.append(f"noise-like: spectral flatness {flatness:.2f}")
    if 0.0 < silent_fraction <= MAX_SILENT_FRACTION and silent_fraction > 0.1:
        report.warnings.append(f"{silent_fraction:.0%} silence")

    report.ok = not report.failures
    return report


AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}


def check_path(target: Path | str, *, min_duration_s: float = MIN_DURATION_S) -> list[TrackReport]:
    """Check one file, or every audio file anywhere inside a directory (recursively,
    since assets/tracks/ nests audio one subdirectory per category cell)."""
    target = Path(target)
    if target.is_dir():
        files = sorted(p for p in target.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
    elif target.exists():
        files = [target]
    else:
        raise FileNotFoundError(f"no such file or directory: {target}")
    return [check_track(f, min_duration_s=min_duration_s) for f in files]
