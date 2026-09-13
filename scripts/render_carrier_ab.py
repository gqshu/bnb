#!/usr/bin/env python3
"""
render_carrier_ab.py — render a "measured carrier" vs "static default carrier" A/B
pair per track, so a human can actually listen and judge whether estimate_root.py's
carrier choice fuses better than a fixed default.

For each input track, writes two mp3s into --out:
    <track>__<measured>hz_measured.mp3
    <track>__<static>hz_static.mp3

Both use the same segment of the bed, the same beat rate/amplitude, and the same
headroom-scaled mix — carrier_hz is the only thing that differs, so whatever you
hear differ is attributable to it and nothing else.

Reuses the product's own binaural renderer (bnb.tone.render_binaural) and the same
mp3 encode path prepare_cloud_assets.py uses (lameenc + bnb.stream.to_int16_bytes),
so what you hear here is representative of what the client would actually produce.

Usage:
    uv run scripts/render_carrier_ab.py run/cloud_assets/bg/energizer_chillhop_seed45697.mp3
    uv run scripts/render_carrier_ab.py TRACK1.mp3 TRACK2.mp3 --out run/carrier_ab
    uv run scripts/render_carrier_ab.py TRACK.mp3 --beat-hz 6 --static-carrier 440
"""

import argparse
from pathlib import Path

import lameenc
import numpy as np

from bnb import tone
from bnb.stream import to_int16_bytes

from estimate_root import analyze  # sibling script — same carrier_hz this reports

BEAT_HZ = 10.0
"""Alpha, a common relax/demo rate. Not tied to any one profile — this is a listening
test, not a rendering of a specific shipped preset."""

BEAT_AMPLITUDE = 0.15
"""Deliberately louder than the shipped EEG-profile default (0.05, see audio.ts) so
the carrier is easy to judge by ear. A demo you can't hear the beat on can't be A/B'd;
production would run quieter once the carrier itself is validated."""

BG_VOLUME = 0.6  # matches audio.ts's bgVol default
FADE_IN_S = 3.0  # short version of audio.ts's fadeInBeat, so the beat doesn't slam in
CLIP_DURATION_S = 45.0
OUTPUT_CEILING = 0.98  # matches audio.ts's OUTPUT_CEILING
STATIC_CARRIER_HZ = 400.0

MP3_BITRATE_KBPS = 192
MP3_QUALITY = 2  # lameenc: 0=best/slowest; a handful of one-off files, not a batch job


def _segment(bg: np.ndarray, sr: int, duration_s: float) -> np.ndarray:
    """A representative middle chunk rather than the cold intro, which is often
    thinner/quieter than the settled loop body."""
    n = round(duration_s * sr)
    if bg.shape[0] <= n:
        reps = -(-n // bg.shape[0])  # ceil
        return np.tile(bg, (reps, 1))[:n]
    start = (bg.shape[0] - n) // 3  # a third of the way in, past most intros
    return bg[start : start + n]


def _mix(bg: np.ndarray, sr: int, carrier_hz: float, beat_hz: float) -> np.ndarray:
    beat = tone.render_binaural(
        beat_hz, bg.shape[0] / sr, carrier_hz=carrier_hz, sample_rate=sr, amplitude=BEAT_AMPLITUDE
    )
    beat = beat[: bg.shape[0]]

    # Fade the beat in under the background, same reasoning as audio.ts's
    # fadeInBeat: background is present from sample 0, the beat rises under it —
    # not the other way round, and not both slamming in together.
    fade_n = min(round(FADE_IN_S * sr), beat.shape[0])
    if fade_n > 0:
        ramp = np.linspace(0.0, 1.0, fade_n)[:, None]
        beat[:fade_n] *= ramp

    mix = bg * BG_VOLUME + beat
    # Headroom: same shape as audio.ts's updateHeadroom — scale by this exact mix's
    # true peak rather than a fixed gain, so neither A/B variant clips or ends up
    # quieter than the other for a reason unrelated to carrier choice.
    peak = float(np.max(np.abs(mix))) or 1.0
    gain = min(1.0, OUTPUT_CEILING / peak)
    return (mix * gain).astype(np.float32)


def _encode_mp3(samples: np.ndarray, sr: int) -> bytes:
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(MP3_BITRATE_KBPS)
    encoder.set_in_sample_rate(sr)
    encoder.set_channels(2)
    encoder.set_quality(MP3_QUALITY)
    return bytes(encoder.encode(to_int16_bytes(samples))) + bytes(encoder.flush())


def render_pair(path: Path, out_dir: Path, beat_hz: float, static_carrier_hz: float) -> None:
    info = analyze(path)
    measured_hz = info["carrier_hz"]
    print(
        f"{path.name}: key={info['key']} conf={info['key_confidence']:.2f} "
        f"stability={info['stability']:.2f} -> measured {measured_hz:.1f} Hz "
        f"vs static {static_carrier_hz:.0f} Hz"
    )

    bg, sr = tone.load_background(path)
    seg = _segment(bg, sr, CLIP_DURATION_S)

    for label, hz in [
        (f"{measured_hz:g}hz_measured", measured_hz),
        (f"{static_carrier_hz:g}hz_static", static_carrier_hz),
    ]:
        mix = _mix(seg, sr, hz, beat_hz)
        mp3 = _encode_mp3(mix, sr)
        out_path = out_dir / f"{path.stem}__{label}.mp3"
        out_path.write_bytes(mp3)
        print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("tracks", nargs="+", type=Path, help="mp3/wav files to render pairs for")
    ap.add_argument("--out", type=Path, default=Path("run/carrier_ab"))
    ap.add_argument("--beat-hz", type=float, default=BEAT_HZ)
    ap.add_argument("--static-carrier", type=float, default=STATIC_CARRIER_HZ)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for track in args.tracks:
        render_pair(track, args.out, args.beat_hz, args.static_carrier)


if __name__ == "__main__":
    main()
