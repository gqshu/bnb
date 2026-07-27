"""Control the live binaural-beat stream from the command line.

A thin CLI over bnb.client.StreamClient. The service must be running
(`uv run scripts/serve.py`).

    uv run scripts/control.py backgrounds                 # 1. list background track meta
    uv run scripts/control.py current                     # 2. get the current background
    uv run scripts/control.py play --background neutral_noise_seed50801 --beat 10 --volume 0.3
    uv run scripts/control.py play --beat 6 --volume 0.4  # 3. start/change (beat only, no background)
    uv run scripts/control.py volume 0.5                  # 4. change the beat volume
    uv run scripts/control.py freq 7.83                   # 5. change the beat frequency
    uv run scripts/control.py state                       # full state
    uv run scripts/control.py stop

Use --url to target a non-default host/port.
"""

from __future__ import annotations

import argparse
import json
import sys

from bnb.client import DEFAULT_BASE_URL, StreamClient, StreamClientError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help=f"service base URL (default {DEFAULT_BASE_URL})")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("backgrounds", help="list background track meta")
    sub.add_parser("current", help="get the current background")
    sub.add_parser("state", help="print full stream state")
    sub.add_parser("stop", help="stop the stream")

    play = sub.add_parser("play", help="start or change background with a beat + volume combo")
    play.add_argument("--background", default=None, help="background track_id (omit for no background)")
    play.add_argument("--beat", type=float, default=None, help="beat frequency in Hz (omit for no beat)")
    play.add_argument("--volume", type=float, default=None, help="beat volume 0..1")
    play.add_argument("--carrier", type=float, default=None, help="carrier Hz (default 432)")
    play.add_argument("--waveform", default=None, choices=["sine", "triangle", "square", "sawtooth"])
    play.add_argument("--bg-volume", type=float, default=None, help="background volume 0..1")
    play.add_argument(
        "--mode", default=None, choices=["dichotic", "diotic", "monaural", "isochronic"]
    )
    play.add_argument("--depth", type=float, default=None, help="isochronic gate depth 0..1")
    play.add_argument("--duty", type=float, default=None, help="isochronic duty cycle 0.1..0.9")
    play.add_argument("--ramp-ms", type=float, default=None, help="isochronic ramp edge, 2..10 ms")

    vol = sub.add_parser("volume", help="change the beat volume")
    vol.add_argument("value", type=float, help="beat volume 0..1")

    freq = sub.add_parser("freq", help="change the beat frequency")
    freq.add_argument("value", type=float, help="beat frequency in Hz")

    return parser.parse_args()


def _emit(data: object) -> None:
    print(json.dumps(data, indent=2))


def run(args: argparse.Namespace, client: StreamClient) -> None:
    if args.command == "backgrounds":
        rows = client.list_backgrounds()
        for b in rows:
            mark = "▶" if b["rendered"] else "·"
            print(f"{mark} {b['track_id']}\n    {b['summary']}")
        print(f"\n{sum(b['rendered'] for b in rows)}/{len(rows)} rendered (▶ = playable)")
    elif args.command == "current":
        _emit(client.get_background())
    elif args.command == "state":
        _emit(client.get_state())
    elif args.command == "stop":
        _emit(client.stop())
    elif args.command == "play":
        kwargs = {"beat_hz": args.beat, "volume": args.volume, "background_volume": args.bg_volume}
        if args.carrier is not None:
            kwargs["carrier_hz"] = args.carrier
        if args.waveform is not None:
            kwargs["waveform"] = args.waveform
        if args.mode is not None:
            kwargs["mode"] = args.mode
        if args.depth is not None:
            kwargs["depth"] = args.depth
        if args.duty is not None:
            kwargs["duty"] = args.duty
        if args.ramp_ms is not None:
            kwargs["ramp_ms"] = args.ramp_ms
        _emit(client.set_background(args.background, **kwargs))
    elif args.command == "volume":
        _emit(client.set_beat_volume(args.value))
    elif args.command == "freq":
        _emit(client.set_beat_frequency(args.value))


def main() -> None:
    args = parse_args()
    try:
        with StreamClient(args.url) as client:
            run(args, client)
    except StreamClientError as exc:
        sys.exit(f"error: {exc}")
    except OSError as exc:
        sys.exit(f"cannot reach {args.url} (is the service running?): {exc}")


if __name__ == "__main__":
    main()
