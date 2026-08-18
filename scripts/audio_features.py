#!/usr/bin/env python3
"""
audio_features.py — reverse-engineer an audio file's features into the
background-music taxonomy coordinates used by the entrainment product.

Reports, for an mp3/wav/flac/etc:

  Taxonomy axes (map onto substrate x style x T x E x S x M x X x N):
    - tempo_bpm            (T) estimated tempo; low confidence flagged
    - energy_lufs / rms    (E) integrated loudness + RMS
    - spectral_centroid    (S) brightness, Hz + descriptive bin
    - mode_major_minor     (M) major/minor guess + strength
    - texture_density      (X) onset rate, events/sec + descriptive bin
    - nature_bed           (N) heuristic: broadband-noise vs tonal fraction
    - substrate_guess          drone / melodic / field-rec / noise / perc-tail
    - development_arc          static / slow-swell / motif-evolving  (the axis
                               the test feedback ("progression") points at)

  Entrainment-relevant (product-specific):
    - am_depth_by_band     amplitude-modulation depth of the broadband envelope
                           in delta/theta/alpha/beta/gamma rate bands. This is
                           what you'd measure on a competitor AM track (Brain.fm)
                           and what band_guard is meant to keep OUT of the active
                           entrainment band.
    - melodic_salience     pitch-contour presence: distinguishes "plain noise"
                           from "has tones and slow melody".

Usage:
    python3 audio_features.py track.mp3
    python3 audio_features.py track1.mp3 track2.wav --json out.json
    python3 audio_features.py *.mp3 --csv library.csv

Everything is a *measured* feature vector, matching the project rule that the
bandit consumes measured coordinates, not just requested ones.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

try:
    import librosa
except ImportError:
    sys.exit("librosa is required: pip install librosa --break-system-packages")


# ----- descriptive binning helpers (turn numbers into taxonomy words) -----

def _bin(value, edges, labels):
    for e, l in zip(edges, labels):
        if value <= e:
            return l
    return labels[-1]


def spectral_centroid_bin(hz):
    # dark / warm / neutral / bright
    return _bin(hz, [800, 1800, 3200], ["dark", "warm", "neutral", "bright"])


def texture_bin(onset_rate):
    # events per second -> density word
    return _bin(onset_rate, [0.15, 0.5, 1.5],
                ["very_sparse", "sparse", "moderate", "dense"])


def tempo_bin(bpm):
    return _bin(bpm, [50, 70, 100], ["very_slow", "slow", "moderate", "up"])


# ----- envelope AM-depth in EEG-relevant rate bands -----

BANDS = {
    "delta": (2, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta":  (12, 30),
    "gamma": (30, 45),
}


def am_depth_by_band(y, sr):
    """
    Modulation depth of the broadband amplitude envelope in each EEG rate band.

    The envelope is extracted, its spectrum computed, and the fractional power
    in each band is expressed relative to the envelope DC (mean) level. This is
    the quantity that determines whether background content is imposing an
    amplitude modulation that would land on the active entrainment band and
    corrupt credit assignment (band_guard's target).
    """
    # broadband amplitude envelope (Hilbert magnitude), decimated to a low env fs
    from scipy.signal import hilbert, welch
    # downmix, then take analytic-signal magnitude on a decimated version
    env_fs = 200  # plenty for rate bands up to 45 Hz
    hop = max(1, int(sr / env_fs))
    # RMS envelope is more robust than raw hilbert on full-rate audio
    frame = hop * 4
    env = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    env = env - np.mean(env)  # remove DC for modulation analysis
    if np.allclose(env, 0):
        return {b: 0.0 for b in BANDS}, {b: 0.0 for b in BANDS}
    dc = np.mean(librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0])
    fs_env = sr / hop

    f, pxx = welch(env, fs=fs_env, nperseg=min(len(env), 1024))
    total = np.trapezoid(pxx, f) + 1e-12

    depth = {}
    frac = {}
    for name, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        band_power = np.trapezoid(pxx[m], f[m]) if m.any() else 0.0
        # modulation depth ~ sqrt(2 * band power) / DC  (rms of band / mean level)
        depth[name] = float(np.sqrt(2 * band_power) / (dc + 1e-12))
        frac[name] = float(band_power / total)
    return depth, frac


# ----- melodic salience: tones/melody vs plain noise -----

def melodic_salience(y, sr):
    """
    Fraction of frames with a confident pitch, plus pitch-contour movement.
    Separates 'plain noise/drone bed' (low salience, flat contour) from
    'has tones and slow melody' (higher salience, moving contour) — the
    distinction the test feedback is about.
    """
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y, fmin=65, fmax=1500, sr=sr,
            frame_length=4096, hop_length=1024)
    except Exception:
        return {"pitch_presence": 0.0, "contour_movement_semitones": 0.0,
                "melodic": False}
    voiced = np.nan_to_num(voiced_prob)
    presence = float(np.mean(voiced > 0.5))
    valid = f0[~np.isnan(f0)]
    if len(valid) > 8:
        semis = 12 * np.log2(valid / np.median(valid))
        movement = float(np.percentile(semis, 90) - np.percentile(semis, 10))
    else:
        movement = 0.0
    return {"pitch_presence": presence,
            "contour_movement_semitones": movement,
            "melodic": presence > 0.25 and movement > 1.0}


# ----- development / arc: does the track evolve over time? -----

def development_arc(y, sr):
    """
    Measures temporal evolution — the 'progression' the feedback wants.
    Splits the track into thirds and compares timbral + loudness centroids;
    also measures long-term novelty. Returns static / slow_swell / motif_evolving.
    """
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=1024))
    if S.shape[1] < 6:
        return {"arc": "static", "loudness_range_db": 0.0, "timbral_drift": 0.0}
    rms = librosa.feature.rms(S=S)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-9)
    loud_range = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))

    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    n = len(cent)
    a, b, c = cent[:n // 3], cent[n // 3:2 * n // 3], cent[2 * n // 3:]
    seg_means = np.array([a.mean(), b.mean(), c.mean()])
    timbral_drift = float((seg_means.max() - seg_means.min()) /
                          (seg_means.mean() + 1e-9))

    # classify
    if loud_range < 4 and timbral_drift < 0.10:
        arc = "static"
    elif timbral_drift >= 0.20 or loud_range >= 9:
        arc = "motif_evolving"
    else:
        arc = "slow_swell"
    return {"arc": arc, "loudness_range_db": loud_range,
            "timbral_drift": timbral_drift}


# ----- mode (major/minor) -----

def mode_estimate(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    # Krumhansl-Schmuckler-ish major/minor profiles
    maj = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                    2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minr = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    best_maj = max(np.corrcoef(np.roll(maj, k), chroma_mean)[0, 1]
                   for k in range(12))
    best_min = max(np.corrcoef(np.roll(minr, k), chroma_mean)[0, 1]
                   for k in range(12))
    mode = "major" if best_maj >= best_min else "minor"
    strength = float(abs(best_maj - best_min))
    return {"mode": mode, "mode_confidence": strength}


# ----- substrate guess from the measured vector -----

def substrate_guess(feat):
    onset = feat["texture_density"]["onset_rate_per_s"]
    mel = feat["melodic_salience"]["melodic"]
    noise_frac = feat["nature_bed"]["broadband_fraction"]
    flat = feat["nature_bed"]["spectral_flatness"]
    if noise_frac > 0.55 and flat > 0.35 and not mel:
        return "noise_texture_or_field_recording"
    if onset > 0.8 and feat["development_arc"]["timbral_drift"] > 0.15:
        return "percussive_with_tail"
    if mel and onset < 0.8:
        return "melodic_instrument"
    if not mel and onset < 0.2:
        return "drone"
    return "mixed"


# ----- nature-bed / noisiness -----

def nature_bed(y, sr):
    flat = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    # broadband fraction: energy spread vs concentrated in tonal peaks
    S = np.abs(librosa.stft(y))
    # tonal-to-total via harmonic-percussive: percussive+residual ~ broadband
    y_h, y_p = librosa.effects.hpss(y)
    e_h = np.sum(y_h ** 2)
    e_p = np.sum(y_p ** 2)
    broadband_fraction = float(e_p / (e_h + e_p + 1e-12))
    return {"spectral_flatness": flat,
            "broadband_fraction": broadband_fraction}


# ----- top-level extraction -----

def extract(path, sr=22050, max_seconds=120):
    y, sr = librosa.load(path, sr=sr, mono=True, duration=max_seconds)
    if len(y) < sr:
        raise ValueError(f"{path}: too short / unreadable")
    y = y / (np.max(np.abs(y)) + 1e-9)

    # tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    # energy
    rms = float(np.mean(librosa.feature.rms(y=y)))
    rms_db = float(librosa.amplitude_to_db(np.array([rms]))[0])

    # spectral centroid
    cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    # onset density
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    dur = len(y) / sr
    onset_rate = len(onsets) / dur

    depth, frac = am_depth_by_band(y, sr)

    feat = {
        "file": Path(path).name,
        "duration_s_analyzed": round(dur, 1),
        "tempo": {"bpm": round(tempo, 1), "bin": tempo_bin(tempo)},
        "energy": {"rms": round(rms, 5), "rms_db": round(rms_db, 1)},
        "spectral_centroid": {"hz": round(cent, 1),
                              "bin": spectral_centroid_bin(cent)},
        "texture_density": {"onset_rate_per_s": round(onset_rate, 3),
                            "bin": texture_bin(onset_rate)},
        "mode": mode_estimate(y, sr),
        "nature_bed": nature_bed(y, sr),
        "melodic_salience": melodic_salience(y, sr),
        "development_arc": development_arc(y, sr),
        "am_depth_by_band": {k: round(v, 4) for k, v in depth.items()},
        "am_power_fraction_by_band": {k: round(v, 4) for k, v in frac.items()},
    }
    feat["substrate_guess"] = substrate_guess(feat)
    return feat


def pretty(feat):
    f = feat
    lines = []
    lines.append(f"── {f['file']}  ({f['duration_s_analyzed']}s analyzed) ──")
    lines.append(f"  substrate (guess): {f['substrate_guess']}")
    lines.append(f"  development/arc  : {f['development_arc']['arc']}"
                 f"  (loudness range {f['development_arc']['loudness_range_db']:.1f} dB,"
                 f" timbral drift {f['development_arc']['timbral_drift']:.2f})")
    lines.append(f"  tempo (T)        : {f['tempo']['bpm']} bpm [{f['tempo']['bin']}]")
    lines.append(f"  energy (E)       : {f['energy']['rms_db']} dB RMS")
    lines.append(f"  centroid (S)     : {f['spectral_centroid']['hz']} Hz"
                 f" [{f['spectral_centroid']['bin']}]")
    lines.append(f"  mode (M)         : {f['mode']['mode']}"
                 f" (conf {f['mode']['mode_confidence']:.2f})")
    lines.append(f"  texture (X)      : {f['texture_density']['onset_rate_per_s']}/s"
                 f" [{f['texture_density']['bin']}]")
    lines.append(f"  nature/noise (N) : broadband {f['nature_bed']['broadband_fraction']:.2f},"
                 f" flatness {f['nature_bed']['spectral_flatness']:.2f}")
    ms = f['melodic_salience']
    lines.append(f"  melodic salience : presence {ms['pitch_presence']:.2f},"
                 f" movement {ms['contour_movement_semitones']:.1f} semitones"
                 f"  -> melodic={ms['melodic']}")
    lines.append("  AM depth by band (envelope modulation, watch the active band):")
    for b in BANDS:
        d = f['am_depth_by_band'][b]
        fr = f['am_power_fraction_by_band'][b]
        bar = "#" * min(40, int(d * 200))
        lines.append(f"      {b:6s} depth {d:.4f}  frac {fr:.3f}  {bar}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="audio files (mp3/wav/flac/...)")
    ap.add_argument("--json", help="write full feature vectors to JSON")
    ap.add_argument("--csv", help="write flat table to CSV")
    ap.add_argument("--max-seconds", type=int, default=120,
                    help="analyze up to N seconds (default 120)")
    args = ap.parse_args()

    results = []
    for path in args.files:
        try:
            feat = extract(path, max_seconds=args.max_seconds)
            results.append(feat)
            print(pretty(feat))
            print()
        except Exception as e:
            print(f"!! {path}: {e}", file=sys.stderr)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}")

    if args.csv and results:
        import csv
        def flat(d, prefix=""):
            out = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    out.update(flat(v, f"{prefix}{k}."))
                else:
                    out[f"{prefix}{k}"] = v
            return out
        rows = [flat(r) for r in results]
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
