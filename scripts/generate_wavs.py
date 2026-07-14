"""Render sample binaural WAVs into run/ so you can listen to them.

    uv run scripts/generate_wavs.py                      # all modes, 60 s each
    uv run scripts/generate_wavs.py sleep_prep --duration 300
    uv run scripts/generate_wavs.py wind_down --waveform square

Nothing here is the product: these are constant-beat renders for ear-checking the
tone generator. The real session steps the beat down from the user's measured
baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bnb.tone import CARRIER_HZ, WAVEFORMS, render_binaural, write_wav

RUN_DIR = Path(__file__).resolve().parent.parent / "run"

# The down-regulation modes from the product doc, with the beat each one targets.
MODES: dict[str, float] = {
    "recharge": 10.0,  # alpha
    "wind_down": 7.83,  # the doc's Wind Down target
    "sleep_prep": 4.0,  # delta; the doc warns against chasing sub-4 Hz
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "modes",
        nargs="*",
        choices=[*MODES, []],
        default=list(MODES),
        help="modes to render (default: all)",
    )
    parser.add_argument("--duration", type=float, default=60.0, help="seconds (default: 60)")
    parser.add_argument("--carrier", type=float, default=CARRIER_HZ, help="carrier Hz")
    parser.add_argument("--waveform", choices=WAVEFORMS, default="sine")
    parser.add_argument("--out-dir", type=Path, default=RUN_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for mode in args.modes:
        beat_hz = MODES[mode]
        samples = render_binaural(
            beat_hz=beat_hz,
            duration_s=args.duration,
            carrier_hz=args.carrier,
            amplitude=0.3,
            waveform=args.waveform,
        )
        path = args.out_dir / f"{mode}_{beat_hz:g}hz_{args.waveform}.wav"
        write_wav(path, samples)
        print(
            f"{path}  "
            f"carrier {args.carrier:g} Hz / {args.carrier + beat_hz:g} Hz, "
            f"beat {beat_hz:g} Hz, {args.duration:g} s"
        )


if __name__ == "__main__":
    main()
