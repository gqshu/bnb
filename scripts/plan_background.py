"""Plan background-soundscape specs — offline, free, no API and no key.

Spec management is decoupled from rendering (scripts/render_background.py) so you
can preview and inspect the catalog before spending on a paid model, and so a
local model can render the same specs later. This writes the render-independent
request records into the asset repository:

    assets/specs/<track_id>.json    the spec (§3), render block still null
    assets/catalog.json             regenerated index of compact descriptors

    uv run scripts/plan_background.py                    # write the curated sample set
    uv run scripts/plan_background.py --list             # print the taxonomy axes + sample set
    uv run scripts/plan_background.py --coverage         # print the substrate × style count matrix
    uv run scripts/plan_background.py buddhist_meditative:drone neutral:noise_texture
    uv run scripts/plan_background.py --only-new         # skip pairs whose spec already exists

  Grow the library by coverage guide (evenly fill the substrate × style grid):

    uv run scripts/plan_background.py --fill 10          # add the 10 next specs, evenly distributed
    uv run scripts/plan_background.py --per-cell 2       # top every cell up to 2 tracks
    uv run scripts/plan_background.py --fill 6 --styles buddhist_meditative,neutral

    uv run scripts/plan_background.py --rebuild-catalog  # just rebuild catalog.json from specs

Once specs look right, render audio with scripts/render_background.py.
"""

from __future__ import annotations

import argparse

from bnb import assets
from bnb.background import (
    SAMPLE_PAIRS,
    STYLES,
    SUBSTRATES,
    Signature,
    build_signature,
    coverage_report,
    fill_to_per_cell,
    plan_coverage,
    sample_signatures,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "pairs",
        nargs="*",
        metavar="STYLE:SUBSTRATE",
        help="pairs to plan (default: the curated sample set)",
    )
    parser.add_argument("--duration", type=int, default=60, help="seconds per track (default: 60)")

    coverage = parser.add_argument_group("coverage guides (grow the library)")
    coverage.add_argument("--fill", type=int, metavar="N", help="add the next N specs, evenly across the grid")
    coverage.add_argument("--per-cell", type=int, metavar="K", help="add specs until every cell has K tracks")
    coverage.add_argument("--substrates", help="comma-separated substrate subset to restrict coverage to")
    coverage.add_argument("--styles", help="comma-separated style subset to restrict coverage to")

    parser.add_argument("--only-new", action="store_true", help="skip pairs whose spec already exists in the repo")
    parser.add_argument("--coverage", action="store_true", help="print current grid coverage and exit")
    parser.add_argument("--rebuild-catalog", action="store_true", help="rebuild catalog.json from specs and exit")
    parser.add_argument("--list", action="store_true", help="print the substrate/style axes and sample set, then exit")
    return parser.parse_args()


def _split(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",")] if value else None


def existing_cells() -> list[tuple[str, str]]:
    """The (substrate, style) of every spec currently in the repo."""
    return [(s["substrate"], s["style"]) for s in (assets.load_spec(t) for t in assets.list_specs())]


def print_axes() -> None:
    print("Substrates (Axis A):")
    for name in SUBSTRATES:
        print(f"  {name}")
    print("\nStyles (Axis B):")
    for name in STYLES:
        print(f"  {name}")
    print("\nDefault sample set (style:substrate):")
    for style, substrate in SAMPLE_PAIRS:
        print(f"  {style}:{substrate}")


def print_coverage(substrates: list[str] | None, styles: list[str] | None) -> None:
    """Print the substrate × style count matrix with marginals."""
    report = coverage_report(existing_cells(), substrates, styles)
    subs = substrates or list(SUBSTRATES)
    stys = styles or list(STYLES)
    shorts = [SUBSTRATES[s].short for s in subs]
    col_w = max(6, *(len(s) for s in shorts)) + 1
    style_w = max(len(s) for s in stys) + 1

    print(" " * style_w + "".join(f"{sh:>{col_w}}" for sh in shorts) + f"{'Σ':>{col_w}}")
    for sty in stys:
        cells = "".join(f"{report['per_cell'][f'{sty}:{sub}']:>{col_w}}" for sub in subs)
        print(f"{sty:<{style_w}}{cells}{report['per_style'][sty]:>{col_w}}")
    footer = "".join(f"{report['per_substrate'][sub]:>{col_w}}" for sub in subs)
    print(f"{'Σ':<{style_w}}{footer}{report['total']:>{col_w}}")


def planned_signatures(args: argparse.Namespace) -> list[Signature]:
    """Resolve the CLI into the signatures to write as specs."""
    if args.fill is not None or args.per_cell is not None:
        subs, stys = _split(args.substrates), _split(args.styles)
        common = dict(
            existing_cells=existing_cells(),
            used_track_ids=set(assets.list_specs()),
            substrates=subs,
            styles=stys,
        )
        if args.per_cell is not None:
            return fill_to_per_cell(args.per_cell, args.duration, **common)
        return plan_coverage(args.fill, args.duration, **common)

    if args.pairs:
        signatures = []
        for pair in args.pairs:
            if ":" not in pair:
                raise SystemExit(f"expected STYLE:SUBSTRATE, got {pair!r}")
            style, substrate = pair.split(":", 1)
            signatures.append(build_signature(substrate, style, args.duration))
        return signatures

    return sample_signatures(args.duration)


def main() -> None:
    args = parse_args()

    if args.list:
        print_axes()
        return
    if args.coverage:
        print_coverage(_split(args.substrates), _split(args.styles))
        return
    if args.rebuild_catalog:
        catalog = assets.rebuild_catalog()
        print(f"rebuilt {assets.CATALOG_PATH} ({catalog['count']} tracks)")
        return

    written = skipped = 0
    for sig in planned_signatures(args):
        if args.only_new and assets.has_spec(sig.track_id):
            skipped += 1
            print(f"skip (exists)  {sig.track_id}")
            continue
        assets.write_spec(sig.spec())
        written += 1
        print(f"planned        {sig.track_id}  ({sig.substrate.name} x {sig.style.name})")

    catalog = assets.rebuild_catalog()
    print(f"\n{written} specs written, {skipped} skipped; catalog: {catalog['count']} tracks")

    if args.fill is not None or args.per_cell is not None:
        print()
        print_coverage(_split(args.substrates), _split(args.styles))


if __name__ == "__main__":
    main()
