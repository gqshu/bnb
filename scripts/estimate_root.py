#!/usr/bin/env python3
"""
estimate_root.py — estimate each track's musical root so the client can snap the
synthesized binaural/monaural carrier to a frequency that is *consonant* with the
bed instead of colliding with it.

Why this exists
---------------
The mini program synthesizes the carrier live (``audio.ts``: ``osc.frequency =
carrier``); it is never baked into the mp3. So a track's "root" is metadata, not
audio — estimating it changes one number in ``manifest.json`` and re-renders
nothing. Pair the live carrier with an octave of the track's key tonic and the
tone fuses into the chord (you hear the *beat*, not a separate oscillator);
pick an unrelated carrier and it beats against the bed's partials and pops out.

What it measures (numpy + soundfile only — no librosa/scipy, so it can sit in the
manifest pipeline next to prepare_cloud_assets.py without new heavy deps):

  key / mode / key_confidence
        Krumhansl-Schmuckler key finding over a tuning-corrected chromagram.
        confidence = correlation margin of the winning key; low = weakly tonal.
  tuning_cents
        global detuning from A440 (ambient is often a few cents off, and some is
        deliberately tuned to 432 etc.). Folded into every pitch estimate.
  bass_peak_hz / bass_pc / bass_agrees
        strongest stable partial in 55-300 Hz (a drone's pedal tone). When its
        pitch class agrees with the key tonic, it corroborates carrier_hz —
        reported as separate evidence, not folded into the number itself.
  carrier_hz
        the tonic pitch class rendered in the octave nearest CARRIER_TARGET_HZ.
        This is the number that goes into the manifest — the profile's mode
        (binaural/monaural/isochronic) is a separate, product-level choice made
        elsewhere and is never touched by this script or by how confident an
        estimate is.
  stability / low_confidence
        how much the key wanders across the track (per-window agreement, measured
        against the *global* key's profile so ordinary chord movement within one
        key doesn't read as instability — only a genuine key change or atonal
        drift does). low_confidence is diagnostic only, for a human judging how
        much to trust carrier_hz on a given track; nothing reads it automatically.

Usage
-----
    uv run scripts/estimate_root.py track.mp3
    uv run scripts/estimate_root.py run/cloud_assets/bg/*.mp3 --json roots.json
    uv run scripts/estimate_root.py run/cloud_assets/bg --sample 12   # a spread
    uv run scripts/estimate_root.py run/cloud_assets/bg --csv roots.csv

Everything is a *measured* coordinate, matching the project rule that the client
consumes measured values, not requested ones.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    from bnb.tone import MAX_CARRIER_HZ, MIN_CARRIER_HZ
except ImportError:
    # Lets this script run standalone (no bnb package on path) without silently
    # drifting from the product's real bound — same numbers tone.py enforces today.
    MIN_CARRIER_HZ, MAX_CARRIER_HZ = 200.0, 900.0

# ── constants ─────────────────────────────────────────────────────────────

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles (major / minor), tonic-relative, starting on the
# tonic. Correlating the chroma against all 24 rotations picks key + mode.
KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# Chroma is built only from this range: below skips sub-bass rumble whose pitch is
# unreliable, above skips brightness/air that carries little tonal information.
CHROMA_LO_HZ = 55.0
CHROMA_HI_HZ = 2000.0

# The bass pedal / drone fundamental lives here. This is where a "root" you can
# hear as a held tone actually sits.
BASS_LO_HZ = 55.0
BASS_HI_HZ = 300.0

# Where a pleasant carrier wants to live: low enough to sit under the music, but
# never below MIN_CARRIER_HZ — tone.py's render_binaural rejects anything outside
# [MIN_CARRIER_HZ, MAX_CARRIER_HZ], so a target below that bound would suggest
# carriers the renderer refuses. Sits near the low end of that accepted range.
CARRIER_TARGET_HZ = max(240.0, MIN_CARRIER_HZ + 40.0)

# STFT: 32 kHz sources -> ~0.26 s frames, 4x overlap. Fine for tonal analysis and
# sub-second per track.
N_FFT = 8192
HOP = 2048

# Per-track stability is measured over this many equal windows.
N_WINDOWS = 8

# A track is flagged for AM-routing (no fixed carrier) when the key is too weak or
# too unstable to trust a single root.
MIN_KEY_CONFIDENCE = 0.55
MIN_STABILITY = 0.6


# ── io ────────────────────────────────────────────────────────────────────

def load_mono(path):
    """Read any soundfile-supported file, downmix to mono float64."""
    y, sr = sf.read(str(path), dtype="float64", always_2d=True)
    return y.mean(axis=1), sr


# ── chroma ────────────────────────────────────────────────────────────────

def _stft_mag(y, n_fft=N_FFT, hop=HOP):
    """Magnitude STFT via numpy rfft, Hann-windowed. Returns (n_bins, n_frames)."""
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    win = np.hanning(n_fft)
    n_frames = 1 + (len(y) - n_fft) // hop
    # strided frames without copying the whole thing per frame
    idx = np.arange(n_fft)[:, None] + hop * np.arange(n_frames)[None, :]
    frames = y[idx] * win[:, None]
    return np.abs(np.fft.rfft(frames, axis=0))


def _bin_freqs(sr, n_fft=N_FFT):
    return np.fft.rfftfreq(n_fft, 1.0 / sr)


def estimate_tuning_cents(mag, freqs):
    """
    Global detuning from equal-tempered A440, in cents (-50..50).

    Take the loud bins in the tonal range, express each as a fractional MIDI pitch,
    and look at how far each sits from the nearest semitone. The circular mean of
    those deviations is the tuning offset. Weighted by magnitude so strong partials
    dominate.
    """
    spec = mag.mean(axis=1)
    band = (freqs >= CHROMA_LO_HZ) & (freqs <= CHROMA_HI_HZ) & (spec > 0)
    f = freqs[band]
    w = spec[band]
    if f.size == 0:
        return 0.0
    # keep only the stronger half — weak bins are mostly noise between partials
    thr = np.median(w)
    sel = w >= thr
    f, w = f[sel], w[sel]
    if f.size == 0:
        return 0.0
    midi = 69.0 + 12.0 * np.log2(f / 440.0)
    dev = midi - np.round(midi)  # in [-0.5, 0.5] semitones
    # circular mean over a semitone period to avoid wrap bias
    ang = dev * 2.0 * np.pi
    mean_ang = np.arctan2(np.average(np.sin(ang), weights=w),
                          np.average(np.cos(ang), weights=w))
    return float(mean_ang / (2.0 * np.pi) * 100.0)


def chroma_vector(mag, freqs, tuning_cents=0.0):
    """
    12-D pitch-class profile from a magnitude STFT, tuning-corrected.

    Each in-range bin's magnitude is folded onto its pitch class (nearest of 12),
    summed across frames. Normalized to unit max so it correlates cleanly against
    the KS profiles.
    """
    band = (freqs >= CHROMA_LO_HZ) & (freqs <= CHROMA_HI_HZ)
    f = freqs[band]
    if f.size == 0:
        return np.zeros(12)
    midi = 69.0 + 12.0 * np.log2(f / 440.0) - tuning_cents / 100.0
    pc = np.mod(np.round(midi).astype(int), 12)
    energy = mag[band, :].sum(axis=1)
    chroma = np.zeros(12)
    np.add.at(chroma, pc, energy)
    m = chroma.max()
    return chroma / m if m > 0 else chroma


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


def find_key(chroma):
    """
    Krumhansl-Schmuckler: correlate the chroma against all 24 rotated major/minor
    profiles. Returns (tonic_pc, mode, confidence) where confidence is the margin
    of the winner over the runner-up, scaled to ~0..1 — a stand-in for "how clearly
    tonal is this".
    """
    scores = []
    for tonic in range(12):
        scores.append((_corr(chroma, np.roll(KS_MAJOR, tonic)), tonic, "major"))
        scores.append((_corr(chroma, np.roll(KS_MINOR, tonic)), tonic, "minor"))
    scores.sort(reverse=True)
    best_r, tonic, mode = scores[0]
    second_r = scores[1][0]
    # margin between best and next, lifted into a friendlier 0..1-ish range
    confidence = max(0.0, best_r - second_r) * 3.0 + max(0.0, best_r) * 0.3
    return tonic, mode, min(1.0, confidence)


# ── bass fundamental ──────────────────────────────────────────────────────

def find_bass_root(mag, freqs, tuning_cents=0.0):
    """
    Strongest stable partial in the bass band -> (hz, pitch_class) or (None, None).

    This is the held pedal tone of a drone. When it agrees with the key tonic it is
    the ideal carrier target: the live tone hides *inside* it (simultaneous masking)
    and you only perceive the beat.
    """
    spec = mag.mean(axis=1)
    band = (freqs >= BASS_LO_HZ) & (freqs <= BASS_HI_HZ)
    if not band.any():
        return None, None
    f = freqs[band]
    s = spec[band]
    if s.max() <= 0:
        return None, None
    peak = int(np.argmax(s))
    hz = float(f[peak])
    midi = 69.0 + 12.0 * np.log2(hz / 440.0) - tuning_cents / 100.0
    pc = int(round(midi)) % 12
    return hz, pc


# ── carrier suggestion ────────────────────────────────────────────────────

def tonic_to_carrier(tonic_pc, tuning_cents=0.0, target_hz=CARRIER_TARGET_HZ):
    """
    Render a pitch class as a frequency in whichever octave lands nearest the
    pleasant-carrier target, then nudge by whole octaves (preserving the pitch
    class / consonance) until it falls inside [MIN_CARRIER_HZ, MAX_CARRIER_HZ] —
    the range tone.py's render_binaural actually accepts. Tuning-corrected so a
    432-ish track gets a 432-ish carrier, keeping consonance exact.
    """
    # frequency of this pitch class in the octave starting at C0, then shift octaves
    # to sit closest to target in log space.
    base = 440.0 * 2.0 ** ((tonic_pc - 9) / 12.0) * 2.0 ** (tuning_cents / 1200.0)
    # base is the pitch class somewhere near ~C4; move by whole octaves toward target
    k = round(np.log2(target_hz / base))
    hz = base * 2.0 ** k
    while hz < MIN_CARRIER_HZ:
        hz *= 2.0
    while hz > MAX_CARRIER_HZ:
        hz /= 2.0
    return float(hz)


# ── stability ─────────────────────────────────────────────────────────────

# How far below the window's own best-fitting key the *global* key's score may
# sit and still count as "still in this key". Needed because a plain I-IV-V-I
# progression's IV/V windows often score another tonic higher than the tonic
# itself (IV and V are themselves KS-profile peaks) — requiring the window to
# *rediscover* the exact global tonic penalizes ordinary diatonic motion, not
# actual key change. Correlating against the global profile directly and asking
# "is it still competitive" survives normal chord movement; it drops only when
# the window has genuinely modulated or gone atonal.
STABILITY_MARGIN = 0.15


def measure_stability(y, sr, global_tonic, global_mode, tuning_cents):
    """
    Fraction of equal-length windows that still fit the track's overall key.

    1.0 = rock-steady tonal center (a drone, or a stable progression in one key)
          -> the carrier estimate is trustworthy for the whole track.
    Low  = the key wanders / the track is atonal -> treat the carrier estimate
          with less confidence (this is diagnostic only; nothing auto-switches
          on it).
    """
    n = len(y)
    w = n // N_WINDOWS
    if w < N_FFT:
        return 1.0  # too short to window meaningfully; treat as stable
    global_profile = np.roll(KS_MAJOR if global_mode == "major" else KS_MINOR, global_tonic)
    freqs = _bin_freqs(sr)
    agree = 0
    total = 0
    for i in range(N_WINDOWS):
        seg = y[i * w:(i + 1) * w]
        mag = _stft_mag(seg)
        ch = chroma_vector(mag, freqs, tuning_cents)
        if ch.max() == 0:
            continue
        total += 1
        global_score = _corr(ch, global_profile)
        best_score = max(
            _corr(ch, np.roll(KS_MAJOR, t)) for t in range(12)
        )
        best_score = max(
            best_score, max(_corr(ch, np.roll(KS_MINOR, t)) for t in range(12))
        )
        if global_score >= best_score - STABILITY_MARGIN:
            agree += 1
    return agree / total if total else 1.0


# ── per-track driver ──────────────────────────────────────────────────────

def analyze(path):
    y, sr = load_mono(path)
    mag = _stft_mag(y)
    freqs = _bin_freqs(sr)

    tuning = estimate_tuning_cents(mag, freqs)
    chroma = chroma_vector(mag, freqs, tuning)
    tonic, mode, conf = find_key(chroma)
    bass_hz, bass_pc = find_bass_root(mag, freqs, tuning)
    stability = measure_stability(y, sr, tonic, mode, tuning)

    bass_agrees = bass_pc is not None and bass_pc == tonic
    # carrier_hz is always rendered into the carrier band (same pitch class, octave
    # placed near CARRIER_TARGET_HZ) so it is directly usable as-is — a raw 66 Hz
    # bass pedal is the right *pitch class* but too low to perceive a beat on.
    # bass_agrees carries "the measured pedal confirms the key" as separate evidence.
    carrier_hz = tonic_to_carrier(tonic, tuning)

    # Diagnostic only, for a human deciding how much to trust carrier_hz on this
    # track — nothing reads this to change engine behavior automatically.
    low_confidence = conf < MIN_KEY_CONFIDENCE or stability < MIN_STABILITY

    return {
        "file": Path(path).name,
        "key": f"{PITCH_NAMES[tonic]} {mode}",
        "tonic_pc": tonic,
        "mode": mode,
        "key_confidence": round(conf, 3),
        "tuning_cents": round(tuning, 1),
        "bass_peak_hz": round(bass_hz, 2) if bass_hz else None,
        "bass_pc": PITCH_NAMES[bass_pc] if bass_pc is not None else None,
        "bass_agrees": bass_agrees,
        "carrier_hz": round(carrier_hz, 2),
        "stability": round(stability, 3),
        "low_confidence": low_confidence,
    }


# ── cli ───────────────────────────────────────────────────────────────────

def collect_paths(inputs, sample):
    paths = []
    for p in inputs:
        p = Path(p)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.mp3")) + sorted(p.glob("*.wav")))
        else:
            paths.append(p)
    if sample and len(paths) > sample:
        # even spread across the (sorted) library so we see a variety of substrates
        idx = np.linspace(0, len(paths) - 1, sample).round().astype(int)
        paths = [paths[i] for i in sorted(set(idx))]
    return paths


def print_table(rows):
    hdr = ["file", "key", "conf", "tune", "bass_hz", "bass", "agree",
           "carrier_hz", "stab", "low_conf"]
    widths = [40, 9, 5, 5, 8, 5, 5, 10, 5, 8]
    line = "  ".join(h.ljust(w) for h, w in zip(hdr, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        cells = [
            r["file"][:40],
            r["key"],
            f"{r['key_confidence']:.2f}",
            f"{r['tuning_cents']:+.0f}",
            f"{r['bass_peak_hz']}" if r["bass_peak_hz"] else "-",
            r["bass_pc"] or "-",
            "y" if r["bass_agrees"] else "n",
            f"{r['carrier_hz']:.1f}",
            f"{r['stability']:.2f}",
            "low" if r["low_confidence"] else "",
        ]
        print("  ".join(str(c).ljust(w) for c, w in zip(cells, widths)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="audio files or a directory")
    ap.add_argument("--sample", type=int, default=0,
                    help="analyze an even spread of N tracks from the inputs")
    ap.add_argument("--json", type=Path, help="write full results as JSON")
    ap.add_argument("--csv", type=Path, help="write results as CSV")
    args = ap.parse_args()

    paths = collect_paths(args.inputs, args.sample)
    if not paths:
        sys.exit("no audio files found")

    rows = []
    for p in paths:
        try:
            rows.append(analyze(p))
        except Exception as e:  # noqa: BLE001 - one bad file shouldn't sink the run
            print(f"[skip] {Path(p).name}: {e}", file=sys.stderr)

    print_table(rows)
    n_low = sum(r["low_confidence"] for r in rows)
    print(f"\n{len(rows)} tracks | {n_low} carrier estimates flagged low-confidence "
          f"({100 * n_low / max(1, len(rows)):.0f}%)")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.json}")
    if args.csv and rows:
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
