"""Render sample stimulus WAVs into run/ so you can listen to them.

    uv run scripts/generate_wavs.py                      # all modes, 60 s each
    uv run scripts/generate_wavs.py sleep_prep --duration 300
    uv run scripts/generate_wavs.py wind_down --waveform square
    uv run scripts/generate_wavs.py --stimulus isochronic --duty 0.4

  AM music — modulates a background bed instead of a synthesized carrier, so it
  needs one (a track_id from assets/tracks/, or any audio file path):

    uv run scripts/generate_wavs.py --stimulus am_music --background lofi_drone_seed47621
    uv run scripts/generate_wavs.py --stimulus am_music --background bed.wav --depth 0.8
    uv run scripts/generate_wavs.py --stimulus am_music --background bed.wav --modulator gate
    uv run scripts/generate_wavs.py --stimulus am_music --background bed.wav --beat 40  # gamma

Nothing here is the product: these are constant-beat renders for ear-checking the
generators. The real session steps the beat down from the user's measured baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bnb.tone import (
    AM_DEPTH,
    AM_MODULATORS,
    CARRIER_HZ,
    ISOCHRONIC_CARRIER_HZ,
    WAVEFORMS,
    load_background,
    render_am_music,
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
# am_music isn't here: it takes a bed rather than a carrier, so it doesn't share the
# tone renderers' signature and is dispatched separately.
RENDERERS = {
    "binaural": render_binaural,
    "monaural": render_monaural,
    "isochronic": render_isochronic,
}
STIMULI = (*RENDERERS, "am_music")


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
    parser.add_argument(
        "--beat",
        type=float,
        default=None,
        help="beat Hz, overriding the mode's target (e.g. 40 for gamma AM music)",
    )
    parser.add_argument("--waveform", choices=WAVEFORMS, default="sine")
    parser.add_argument(
        "--stimulus",
        choices=STIMULI,
        default="binaural",
        help="stimulus type (default: binaural)",
    )
    parser.add_argument(
        "--background",
        help="am_music only: a track_id in assets/tracks/, or a path to an audio file",
    )
    parser.add_argument(
        "--modulator",
        choices=AM_MODULATORS,
        default="sine",
        help="am_music envelope shape (default: sine, the least intrusive)",
    )
    parser.add_argument(
        "--depth",
        type=float,
        default=None,
        help=f"modulation depth 0..1 (default: 1.0 isochronic, {AM_DEPTH} am_music)",
    )
    parser.add_argument("--duty", type=float, default=0.5, help="gate duty cycle 0.1..0.9")
    parser.add_argument("--ramp-ms", type=float, default=5.0, help="gate ramp edge, 2..10 ms")
    parser.add_argument("--out-dir", type=Path, default=RUN_DIR)
    return parser.parse_args()


def targets(args: argparse.Namespace) -> dict[str, float]:
    """label -> beat Hz. ``--beat`` collapses the mode set to one explicit render."""
    if args.beat is not None:
        return {"custom": args.beat}
    return {mode: MODES[mode] for mode in args.modes}


def render_am(args: argparse.Namespace, label: str, beat_hz: float) -> None:
    """AM music: modulate a real bed, at the bed's own sample rate."""
    if not args.background:
        raise SystemExit("--stimulus am_music needs --background (a track_id or an audio file path)")
    try:
        background, sample_rate = load_background(args.background)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))

    depth = AM_DEPTH if args.depth is None else args.depth
    samples = render_am_music(
        background,
        beat_hz=beat_hz,
        duration_s=args.duration,
        sample_rate=sample_rate,
        depth=depth,
        modulator=args.modulator,
        duty=args.duty,
        ramp_ms=args.ramp_ms,
    )
    bed = Path(args.background).stem
    path = args.out_dir / f"{label}_am_music_{beat_hz:g}hz_{args.modulator}_depth{depth:g}_{bed}.wav"
    write_wav(path, samples, sample_rate)
    print(
        f"{path}  "
        f"bed {bed} @ {sample_rate} Hz, beat {beat_hz:g} Hz, depth {depth:g}, "
        f"{args.duration:g} s ({args.modulator} modulator)"
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.stimulus == "am_music":
        for label, beat_hz in targets(args).items():
            render_am(args, label, beat_hz)
        return

    render = RENDERERS[args.stimulus]
    carrier = args.carrier
    if carrier is None:
        carrier = ISOCHRONIC_CARRIER_HZ if args.stimulus == "isochronic" else CARRIER_HZ

    for label, beat_hz in targets(args).items():
        kwargs = dict(beat_hz=beat_hz, duration_s=args.duration, carrier_hz=carrier, amplitude=0.3, waveform=args.waveform)
        if args.stimulus == "isochronic":
            kwargs.update(depth=1.0 if args.depth is None else args.depth, duty=args.duty, ramp_ms=args.ramp_ms)
        samples = render(**kwargs)
        path = args.out_dir / f"{label}_{args.stimulus}_{beat_hz:g}hz_{args.waveform}.wav"
        write_wav(path, samples)
        print(
            f"{path}  "
            f"carrier {carrier:g} Hz, beat {beat_hz:g} Hz, {args.duration:g} s ({args.stimulus})"
        )


if __name__ == "__main__":
    main()
