"""Check rendered background tracks for basic listenability.

    uv run scripts/check_background.py                       # every track in assets/tracks/
    uv run scripts/check_background.py run/sa3               # any directory
    uv run scripts/check_background.py bed.wav               # one file
    uv run scripts/check_background.py --verbose             # per-track metrics
    uv run scripts/check_background.py --json                # machine-readable report

Catches the ways a generative render comes back unusable — silent, clipped, hiss,
or a dead constant buffer — so a bad asset is caught before it reaches the library
rather than by ear months later. Exits non-zero if anything fails, so it can gate a
render pipeline.

Thresholds and their rationale live in bnb.qc; this is only the reporting shell.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bnb import assets, qc

MARKS = {"ok": "✓", "warn": "!", "fail": "✗"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "targets",
        nargs="*",
        type=Path,
        help="files or directories to check (default: assets/tracks/)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=qc.MIN_DURATION_S,
        help=f"fail tracks shorter than this many seconds (default: {qc.MIN_DURATION_S:g})",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="print per-track metrics")
    parser.add_argument("--json", action="store_true", help="emit the full report as JSON")
    parser.add_argument(
        "--fail-on-warn", action="store_true", help="exit non-zero on warnings too"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = args.targets or [assets.TRACKS_DIR]

    reports = []
    for target in targets:
        try:
            reports.extend(qc.check_path(target, min_duration_s=args.min_duration))
        except FileNotFoundError as exc:
            raise SystemExit(str(exc))

    if not reports:
        raise SystemExit(f"no audio files found in: {', '.join(str(t) for t in targets)}")

    if args.json:
        print(json.dumps([r.as_dict() for r in reports], indent=2))
    else:
        for report in reports:
            print(f"{MARKS[report.verdict]} {report.path.name}")
            for note in report.failures:
                print(f"    fail: {note}")
            for note in report.warnings:
                print(f"    warn: {note}")
            if args.verbose:
                metrics = ", ".join(f"{k}={v}" for k, v in report.metrics.items())
                print(f"    {metrics}")

        failed = sum(r.verdict == "fail" for r in reports)
        warned = sum(r.verdict == "warn" for r in reports)
        print(f"\n{len(reports)} checked: {len(reports) - failed - warned} ok, {warned} warn, {failed} fail")

    bad = [r for r in reports if r.failures or (args.fail_on_warn and r.warnings)]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
