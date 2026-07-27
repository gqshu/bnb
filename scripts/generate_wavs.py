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

from bnb.tone import (
    CARRIER_HZ,
    ISOCHRONIC_CARRIER_HZ,
    WAVEFORMS,
    render_binaural,
    render_isochronic,
    render_monaural,
    write_wav,
)

RUN_DIR = Path(__file__).resolve().parent.parent / "run"

# The down-regulation modes from the product doc, with the beat each one targets.
MODES: dict[str, float] = {
    "recharge": 10.0,  # alpha
    "wind_down": 7.83,  # the doc's Wind Down target
    "sleep_prep": 4.0,  # delta; the doc warns against chasing sub-4 Hz
}

# Offline renderer per stimulus type (doc: Monaural_and_Isochronic_Beats_Implementation.md).
RENDERERS = {
    "binaural": render_binaural,
    "monaural": render_monaural,
    "isochronic": render_isochronic,
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
    parser.add_argument(
        "--carrier", type=float, default=None, help="carrier Hz (default: per-stimulus)"
    )
    parser.add_argument("--waveform", choices=WAVEFORMS, default="sine")
    parser.add_argument(
        "--stimulus",
        choices=list(RENDERERS),
        default="binaural",
        help="stimulus type (default: binaural)",
    )
    parser.add_argument("--depth", type=float, default=1.0, help="isochronic gate depth 0..1")
    parser.add_argument("--duty", type=float, default=0.5, help="isochronic duty cycle 0.1..0.9")
    parser.add_argument("--ramp-ms", type=float, default=5.0, help="isochronic ramp edge, 2..10 ms")
    parser.add_argument("--out-dir", type=Path, default=RUN_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    render = RENDERERS[args.stimulus]
    carrier = args.carrier
    if carrier is None:
        carrier = ISOCHRONIC_CARRIER_HZ if args.stimulus == "isochronic" else CARRIER_HZ

    for mode in args.modes:
        beat_hz = MODES[mode]
        kwargs = dict(beat_hz=beat_hz, duration_s=args.duration, carrier_hz=carrier, amplitude=0.3, waveform=args.waveform)
        if args.stimulus == "isochronic":
            kwargs.update(depth=args.depth, duty=args.duty, ramp_ms=args.ramp_ms)
        samples = render(**kwargs)
        path = args.out_dir / f"{mode}_{args.stimulus}_{beat_hz:g}hz_{args.waveform}.wav"
        write_wav(path, samples)
        print(
            f"{path}  "
            f"carrier {carrier:g} Hz, beat {beat_hz:g} Hz, {args.duration:g} s ({args.stimulus})"
        )


if __name__ == "__main__":
    main()
