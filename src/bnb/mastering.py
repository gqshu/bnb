"""Make a rendered master loop cleanly, and repair the defects that survive rendering.

The cloud build ships one 60 s MP3 per track and the client loops it forever
(``audio.ts``: ``src.loop = true``). That is a different contract from the streaming
path, where ``StreamEngine`` crossfades every restart into the track's own head — which
is exactly why ``qc.py`` deliberately does *not* check the loop seam. Nothing crossfades
in the cloud path, so the seam is audible, and on this library it is the loudest defect
there is: most renders fade out to near-silence at 60 s but start at full level, so every
lap ends with a fade to nothing followed by a step straight back into a loud head.

So this module prepares a master for looping rather than merely transcoding it:

    declick  ->  trim  ->  crossfade  ->  peak limit

``trim`` drops the leading silence and the trailing fade-out, ``crossfade`` folds the
new tail back over the new head so the wrap point is sample-continuous, and the file
that comes out is one whose last sample runs into its first. The client's own
``findLoopWindow`` (which trims the MP3 encoder's padding silence) then lands its loop
points on real content, because after this there is no other silence for it to find.

``declick`` is the conservative half. Generative renders occasionally contain a true
sample-scale discontinuity — a splice step, a decoder glitch — and those are worth
repairing. What they must not be confused with is *content* that is legitimately
impulsive: a fireplace crackle, a stream splash, a chime attack. Both look like a spike
in a first-difference plot, and no threshold on a single event separates them cleanly —
set it high enough to spare the crackle and it sleeps through half the real clicks.

What does separate them is **how often they happen**. A defect is rare, a few in a minute
at worst; a fireplace is *made of* impulses and detects at hundreds a minute. So the
detector runs at a sensitivity that actually catches clicks (:data:`CLICK_Z`) and the
per-track rate decides whether to act on what it found (:data:`CLICK_RATE_PER_MIN`): a
track over the floor is reported and returned untouched. A crackle is content, and
erasing it would be a worse bug than the one being fixed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

SILENCE_FLOOR = 1e-3
"""-60 dBFS. The same threshold the mini program's ``findLoopWindow`` uses to find the
audible span, so "silence" means the same thing on both sides of the upload."""

# --- declick -------------------------------------------------------------------
CLICK_Z = 20.0
"""How far above its own neighbourhood an impulse must sit to count as a defect.

The residual is a third difference (:func:`_residual`) measured against a 50 ms moving
mean of itself, so this is a *local* ratio: it does not care how loud the track is, only
how unlike its surroundings one sample is. Measured against the shipped library, 20
recovers 98% of injected one-sample steps at -20 dBFS and 75% at -26 dBFS; the threshold
that would also clear fireplace crackle on its own (60+) recovers barely half of them.
Separating defects from crackle is :data:`CLICK_RATE_PER_MIN`'s job, not this one's."""

CLICK_JUMP = 0.05
"""Absolute floor on the sample-to-sample step, so the ratio test can't fire inside
near-silence, where the local mean collapses and every dither bit looks like an event."""

MAX_CLICK_MS = 1.5
"""Longest run that gets repaired. A digital click is a handful of samples; anything
wider is an attack transient, and interpolating across it would punch a hole in the
music. Wider runs are still *reported* — they are the ones worth a listen."""

CLICK_RATE_PER_MIN = 12.0
"""Above this detection rate, a track's impulses are its *content* and none are repaired.

This is what separates a defect from a crackle, and it does it far better than any
threshold on a single event can. A splice step or a decoder glitch is rare — a handful
in a minute at worst. A fireplace bed detects at ~1000 a minute, a stream at ~200, rain
at ~15, because they are *made of* impulses. On the shipped library this floor
quarantines the nine impulsive-by-design tracks — both fireplaces, both streams, both
energizer/warmth beds, a chillhop, a rain and a noise texture — and lets through the ~80
isolated events scattered across the other 27. A flagged track is reported, never
modified."""

# --- loop preparation ----------------------------------------------------------
CROSSFADE_S = 2.0
"""How much of the tail is folded back over the head. Long enough that a level
difference between the two ends reads as a swell rather than a step, short enough to
cost little of a 60 s bed."""

TRIM_BELOW_DB = -12.0
"""Edges quieter than this, relative to the track's own median level, are trimmed. A
plain silence gate (-60 dBFS) is not enough: the renders don't end in silence, they end
in a fade, and a fade left in place is a hole in the loop — the crossfade would spend its
two seconds climbing out of the dip instead of hiding the join. -12 dB is far enough down
to be inside the fade rather than inside the music (an ambient bed breathes by a few dB),
and on the shipped library it costs a median 1.7 s off a 60 s track."""

MAX_TRIM_FRACTION = 0.25
"""Never trim away more than this much of a track. A bed that is genuinely quiet at both
ends is a track we'd be mangling, not fixing, so the trim backs off to a plain silence
gate and then to nothing at all."""

PEAK_DBFS = -1.0
"""Ceiling applied before encoding. MP3 decoding overshoots the samples it was given, so
a master that already touches full scale comes back clipped on the device. Only tracks
above the ceiling are touched, and only downward — relative loudness across the library
is a product decision, not this module's."""

FRAME_S = 0.02


@dataclass
class MasterReport:
    """What mastering did to one track, in the units you'd want to eyeball."""

    duration_s: float
    clicks_found: int = 0
    clicks_repaired: int = 0
    clicks_left: int = 0
    clicks_impulsive: bool = False
    trimmed_head_s: float = 0.0
    trimmed_tail_s: float = 0.0
    crossfade_s: float = 0.0
    seam_jump_before: float = 0.0
    seam_jump_after: float = 0.0
    seam_level_db_before: float = 0.0
    seam_level_db_after: float = 0.0
    gain_db: float = 0.0
    peak: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}

    @property
    def notes(self) -> list[str]:
        """One-line summaries of anything a human should know about this track."""
        out: list[str] = []
        if self.clicks_repaired:
            out.append(f"repaired {self.clicks_repaired} click(s)")
        if self.clicks_impulsive:
            out.append(f"impulsive content ({self.clicks_found} events) — declick skipped")
        elif self.clicks_left:
            out.append(f"{self.clicks_left} impulse(s) left alone (too wide to interpolate)")
        if self.gain_db < -0.01:
            out.append(f"attenuated {self.gain_db:.1f} dB for headroom")
        return out


# ── declick ─────────────────────────────────────────────────────────────────────


def _residual(x: np.ndarray) -> np.ndarray:
    """Third difference: a high-pass sharp enough that a one-sample step dominates it.

    Real audio at 44.1 kHz is band-limited, so even a hard attack spreads its energy over
    many samples and its third difference stays in proportion to the signal around it. A
    genuine discontinuity does not — that asymmetry is the whole detector.
    """
    r = np.zeros_like(x)
    r[3:] = x[3:] - 3 * x[2:-1] + 3 * x[1:-2] - x[:-3]
    return r


def _moving_mean(values: np.ndarray, width: int) -> np.ndarray:
    """Centred moving mean, by prefix sums rather than convolution.

    The window is 50 ms — 2205 samples at 44.1 kHz — and ``np.convolve`` is O(n·width),
    which is minutes per track over a whole library. This is O(n). It also normalises by
    the window's *actual* extent at the edges, where a zero-padded convolution would
    divide a partial sum by the full width and report the first and last 25 ms as far
    quieter than they are — which is precisely where a detector keyed on a ratio would
    then invent clicks.
    """
    pad = width // 2
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    index = np.arange(len(values))
    lo = np.maximum(index - pad, 0)
    hi = np.minimum(index + pad + 1, len(values))
    return (prefix[hi] - prefix[lo]) / (hi - lo)


def _dilate(values: np.ndarray, radius: int) -> np.ndarray:
    """Rolling maximum over +/- ``radius`` samples (grey dilation, for a tiny radius)."""
    out = values.copy()
    for shift in range(1, radius + 1):
        out[shift:] = np.maximum(out[shift:], values[:-shift])
        out[:-shift] = np.maximum(out[:-shift], values[shift:])
    return out


def find_clicks(
    channel: np.ndarray,
    sample_rate: int,
    *,
    z: float = CLICK_Z,
    min_jump: float = CLICK_JUMP,
    window_s: float = 0.05,
) -> list[tuple[int, int]]:
    """Impulsive discontinuities in one channel, as inclusive ``(start, end)`` runs.

    A sample qualifies when its residual is ``z`` times the local mean residual *and* a
    step of at least ``min_jump`` happened within a few samples of it. Neighbouring hits
    are merged, since one click smears across the differencing kernel.
    """
    x = np.asarray(channel, dtype=np.float64)
    if len(x) < 8:
        return []
    magnitude = np.abs(_residual(x))
    width = max(3, int(window_s * sample_rate) | 1)
    local = _moving_mean(magnitude, width)
    jump = np.zeros_like(x)
    jump[1:] = np.abs(np.diff(x))

    # The residual peaks a sample or two *after* the step that caused it (the kernel is
    # four samples wide), while the step itself is one sample. Requiring both to peak on
    # the same index misses real clicks — so the amplitude gate asks whether a step of at
    # least `min_jump` happened anywhere within the kernel's reach.
    hits = np.where((magnitude > z * (local + 1e-9)) & (_dilate(jump, 3) > min_jump))[0]
    if not len(hits):
        return []
    # A click occupies a few samples of the residual; merging within the kernel's own
    # width keeps one defect from being counted (and repaired) as three.
    breaks = np.where(np.diff(hits) > 8)[0]
    return [(int(g[0]), int(g[-1])) for g in np.split(hits, breaks + 1)]


def _interpolate(x: np.ndarray, start: int, end: int) -> None:
    """Replace ``x[start:end+1]`` with a Catmull-Rom spline through the flanking samples.

    In place. The spline matches value *and* slope at both anchors, so the repair joins
    the signal smoothly instead of trading a click for two smaller ones.
    """
    a, b = x[start - 2], x[start - 1]
    c, d = x[end + 1], x[end + 2]
    n = end - start + 1
    t = (np.arange(1, n + 1) / (n + 1))[:, None] ** np.array([0, 1, 2, 3])
    coeff = 0.5 * np.array([2 * b, -a + c, 2 * a - 5 * b + 4 * c - d, -a + 3 * b - 3 * c + d])
    x[start : end + 1] = t @ coeff


@dataclass
class ClickPass:
    """What the declick stage saw and what it decided to do about it."""

    found: int = 0
    repaired: int = 0
    rate_per_min: float = 0.0
    impulsive: bool = False
    """Set when the detection rate says the impulses are content. Nothing was repaired."""

    @property
    def left(self) -> int:
        return self.found - self.repaired


def repair_clicks(
    audio: np.ndarray,
    sample_rate: int,
    *,
    z: float = CLICK_Z,
    min_jump: float = CLICK_JUMP,
    max_click_ms: float = MAX_CLICK_MS,
    max_rate_per_min: float = CLICK_RATE_PER_MIN,
) -> tuple[np.ndarray, ClickPass]:
    """Interpolate over short impulsive defects. Returns ``(audio, pass)``.

    Detection runs first over the whole track, because the decision to repair is not a
    per-event one: a track detecting faster than ``max_rate_per_min`` is impulsive by
    design and comes back untouched (see :data:`CLICK_RATE_PER_MIN`). Below that floor,
    runs longer than ``max_click_ms`` are still counted but left alone — past a
    millisecond or so the thing being "repaired" is much more likely to be a transient
    the track is supposed to have.
    """
    out = np.array(audio, dtype=np.float64, copy=True)
    hits = [find_clicks(out[:, ch], sample_rate, z=z, min_jump=min_jump) for ch in range(out.shape[1])]
    found = sum(len(h) for h in hits)
    minutes = len(out) / sample_rate / 60 or 1.0
    result = ClickPass(found=found, rate_per_min=found / minutes)
    if result.rate_per_min > max_rate_per_min:
        result.impulsive = True
        return out, result

    max_len = max(1, int(max_click_ms * sample_rate / 1000))
    for ch, channel_hits in enumerate(hits):
        x = out[:, ch]
        for start, end in channel_hits:
            # Widen by one sample either side: the discontinuity sits *between* the last
            # good sample and the first bad one, so the flanks are suspect too.
            lo, hi = start - 1, end + 1
            if hi - lo + 1 > max_len or lo < 2 or hi + 2 >= len(x):
                continue
            _interpolate(x, lo, hi)
            result.repaired += 1
    return out, result


# ── loop preparation ────────────────────────────────────────────────────────────


def _frame_rms(mono: np.ndarray, frame_len: int) -> np.ndarray:
    n = (len(mono) // frame_len) * frame_len
    if n == 0:
        return np.array([float(np.sqrt(np.mean(mono**2)))]) if len(mono) else np.array([0.0])
    return np.sqrt(np.mean(mono[:n].reshape(-1, frame_len) ** 2, axis=1))


def _audible_span(rms: np.ndarray, floor: float) -> tuple[int, int] | None:
    above = np.where(rms >= floor)[0]
    return (int(above[0]), int(above[-1])) if len(above) else None


def trim_edges(
    audio: np.ndarray,
    sample_rate: int,
    *,
    below_db: float = TRIM_BELOW_DB,
    max_fraction: float = MAX_TRIM_FRACTION,
) -> tuple[np.ndarray, float, float]:
    """Drop the leading silence and the trailing fade. Returns ``(audio, head_s, tail_s)``.

    The gate is relative to the track's own median level, because the defect being fixed
    is a *fade*, not silence — a tail at -45 dBFS is inaudible next to a head at -12 and
    loops as a hole all the same. Two backstops keep that from eating real music: if the
    relative gate would trim more than ``max_fraction`` of the track it falls back to a
    plain -60 dBFS silence gate, and if even that is too greedy nothing is trimmed at all.
    """
    frame_len = max(1, int(FRAME_S * sample_rate))
    mono = audio.mean(axis=1)
    rms = _frame_rms(mono, frame_len)
    audible = rms[rms >= SILENCE_FLOOR]
    if not len(audible):
        return audio, 0.0, 0.0

    reference = float(np.median(audible))
    for floor in (max(SILENCE_FLOOR, reference * 10 ** (below_db / 20)), SILENCE_FLOOR):
        span = _audible_span(rms, floor)
        if span is None:
            continue
        head, tail = span[0] * frame_len, min(len(mono), (span[1] + 1) * frame_len)
        if (head + len(mono) - tail) <= max_fraction * len(mono):
            return audio[head:tail], head / sample_rate, (len(mono) - tail) / sample_rate
    return audio, 0.0, 0.0


def loop_crossfade(audio: np.ndarray, sample_rate: int, seconds: float = CROSSFADE_S) -> np.ndarray:
    """Fold the tail back over the head so the file's end runs into its own start.

    Equal-power (sine/cosine) rather than linear: the two ends are uncorrelated material,
    so a linear pair would dip ~3 dB in the middle of every lap. The output is shorter by
    the crossfade length, and its last sample is the original sample immediately before
    its first — which is what makes the wrap continuous rather than merely quiet.
    """
    n = len(audio)
    width = min(int(seconds * sample_rate), n // 4)
    if width < 2:
        return audio
    t = np.linspace(0.0, 1.0, width, endpoint=False)[:, None]
    out = np.array(audio[: n - width], dtype=audio.dtype, copy=True)
    out[:width] = audio[:width] * np.sin(0.5 * np.pi * t) + audio[n - width :] * np.cos(0.5 * np.pi * t)
    return out


def limit_peak(audio: np.ndarray, target_dbfs: float = PEAK_DBFS) -> tuple[np.ndarray, float]:
    """Scale down to ``target_dbfs`` if the peak is above it. Returns ``(audio, gain_db)``."""
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    ceiling = 10 ** (target_dbfs / 20)
    if peak <= ceiling or peak <= 0:
        return audio, 0.0
    gain = ceiling / peak
    return audio * gain, 20 * math.log10(gain)


def seam(audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """How bad the wrap point is: ``(sample_jump, level_step_db)``.

    ``sample_jump`` is the discontinuity a looping player hears as a click; ``level_step``
    is the tail-to-head loudness change it hears as the music lurching. Both are reported
    before and after mastering, so the fix is visible rather than asserted.
    """
    if len(audio) < 2:
        return 0.0, 0.0
    jump = float(np.abs(audio[0] - audio[-1]).max())
    edge = max(1, int(0.05 * sample_rate))
    head = float(np.sqrt(np.mean(audio[:edge] ** 2)))
    tail = float(np.sqrt(np.mean(audio[-edge:] ** 2)))
    if head <= 0 or tail <= 0:
        return jump, -math.inf if tail <= 0 < head else 0.0
    return jump, 20 * math.log10(tail / head)


def master_for_loop(
    audio: np.ndarray,
    sample_rate: int,
    *,
    declick: bool = True,
    loop: bool = True,
    crossfade_s: float = CROSSFADE_S,
    click_z: float = CLICK_Z,
    max_click_ms: float = MAX_CLICK_MS,
    peak_dbfs: float = PEAK_DBFS,
) -> tuple[np.ndarray, MasterReport]:
    """Run the full chain over one stereo master. Returns ``(audio, report)``.

    Order is not arbitrary. Declicking runs on the untouched signal, where a defect still
    stands out against its own neighbourhood; trimming then removes the edges that would
    otherwise dominate the crossfade; and the peak limit runs last, over whatever the
    crossfade summed to.
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    before_jump, before_level = seam(x, sample_rate)
    report = MasterReport(
        duration_s=len(x) / sample_rate,
        seam_jump_before=before_jump,
        seam_level_db_before=before_level,
    )

    if declick:
        x, clicks = repair_clicks(x, sample_rate, z=click_z, max_click_ms=max_click_ms)
        report.clicks_found = clicks.found
        report.clicks_repaired = clicks.repaired
        report.clicks_left = clicks.left
        report.clicks_impulsive = clicks.impulsive

    if loop:
        x, head, tail = trim_edges(x, sample_rate)
        report.trimmed_head_s, report.trimmed_tail_s = head, tail
        width = min(int(crossfade_s * sample_rate), len(x) // 4)
        x = loop_crossfade(x, sample_rate, crossfade_s)
        report.crossfade_s = max(0, width) / sample_rate if width >= 2 else 0.0

    x, report.gain_db = limit_peak(x, peak_dbfs)
    report.seam_jump_after, report.seam_level_db_after = seam(x, sample_rate)
    report.duration_s = len(x) / sample_rate
    report.peak = float(np.abs(x).max()) if x.size else 0.0
    return x.astype(np.float32), report
