"""Plan background-soundscape specs — offline, free, no API and no key.

Spec management is decoupled from rendering (scripts/render_background.py) so you
can preview and inspect the catalog before spending on a paid model, and so a
local model can render the same specs later. This writes the render-independent
request records into the asset repository, one subdirectory per category cell:

    assets/specs/<cell>/<track_id>.json    the spec (§3), render block still null
    assets/catalog.json                    regenerated index of compact descriptors

    uv run scripts/plan_background.py                    # write the curated sample set
    uv run scripts/plan_background.py --list             # print the taxonomy axes + sample set
    uv run scripts/plan_background.py --coverage         # print the substrate × style count matrix
    uv run scripts/plan_background.py buddhist_meditative:drone neutral:noise_texture

  Planning never overwrites: a cell's seed is deterministic, so a target whose spec
  already exists on disk is reported and skipped. Replanning one is a deliberate,
  manual act — delete the spec file (or the whole cell directory) and run again.
  That is also how you change a spec's --duration or pick up a prompt edit.

  Special cells (a keyword-driven category outside the substrate × style grid; for
  now only natural_sounds) take the same STYLE:SUBSTRATE-shaped syntax with a group
  name on the left instead of a style — or the bare group name for all its keywords:

    uv run scripts/plan_background.py natural_sounds            # every keyword in the group
    uv run scripts/plan_background.py natural_sounds:rain natural_sounds:ocean

  Grow the library by coverage guide — place the next specs so the cells fill evenly:

    uv run scripts/plan_background.py --fill 10          # add the 10 next specs, evenly distributed
    uv run scripts/plan_background.py --per-cell 2       # top every cell up to 2 tracks
    uv run scripts/plan_background.py --fill 6 --styles buddhist_meditative,neutral

  The guides work the substrate × style grid by default; --groups points them at
  special cells instead (a different taxonomy, so one or the other per run). This is
  also the only way to plan more than one track per keyword — a plain
  GROUP:KEYWORD target is always the cell's first seed:

    uv run scripts/plan_background.py --per-cell 3 --groups natural_sounds
    uv run scripts/plan_background.py --fill 4 --groups natural_sounds

    uv run scripts/plan_background.py --rebuild-catalog  # just rebuild catalog.json from specs

Once specs look right, render audio with scripts/render_background.py.
"""

from __future__ import annotations

import argparse
import difflib
from collections.abc import Iterable

from bnb.background import (
    DEVELOPMENT_FRAGMENT,
    GOALS,
    SAMPLE_PAIRS,
    SPECIAL_GROUPS,
    STYLES,
    SUBSTRATES,
    KeywordSignature,
    Signature,
    build_keyword_signature,
    build_signature,
    coverage_report,
    fill_special_to_per_cell,
    fill_to_per_cell,
    mode_filter_summary,
    plan_coverage,
    plan_special_coverage,
    sample_signatures,
    special_coverage_report,
)
from bnb.catalog import CategoryManager

AnySignature = Signature | KeywordSignature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "targets",
        nargs="*",
        metavar="TARGET",
        help=(
            "grid cells (STYLE:SUBSTRATE), special cells (GROUP:KEYWORD), or a bare "
            "special group (GROUP) for all its keywords; default: the curated sample set"
        ),
    )
    parser.add_argument("--duration", type=int, default=60, help="seconds per track (default: 60)")
    parser.add_argument(
        "--goal",
        choices=["relax", "focus"],
        default="relax",
        help="arousal goal for grid targets/coverage (default: relax); a run plans one goal at a time",
    )
    parser.add_argument(
        "--development",
        choices=list(DEVELOPMENT_FRAGMENT),
        default=None,
        help="progression axis for explicit STYLE:SUBSTRATE targets (default: the goal's own "
        "default_development); must be one the goal admits (see --list)",
    )

    coverage = parser.add_argument_group(
        "coverage guides (grow the library; the substrate × style grid, or special cells with --groups)"
    )
    coverage.add_argument("--fill", type=int, metavar="N", help="add the next N specs, evenly across the cells")
    coverage.add_argument("--per-cell", type=int, metavar="K", help="add specs until every cell has K tracks")
    coverage.add_argument("--substrates", help="comma-separated substrate subset to restrict coverage to")
    coverage.add_argument("--styles", help="comma-separated style subset to restrict coverage to")
    coverage.add_argument(
        "--groups",
        help="comma-separated special groups: fill their keyword cells instead of the grid",
    )

    parser.add_argument("--coverage", action="store_true", help="print current coverage and exit")
    parser.add_argument("--rebuild-catalog", action="store_true", help="rebuild catalog.json from specs and exit")
    parser.add_argument("--list", action="store_true", help="print the taxonomy axes + special groups, then exit")
    return parser.parse_args()


def _split(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",")] if value else None


def unknown_value(name: str, known: Iterable[str], *, label: str, note: str = "") -> SystemExit:
    """The CLI's error for a name that isn't in a taxonomy — a typo is the usual cause,
    so lead with the closest match. Returned, not raised, to read as ``raise unknown_value(...)``.
    """
    options = list(known)
    close = difflib.get_close_matches(name, options, n=1)
    hint = f" — did you mean {close[0]!r}?" if close else ""
    lines = [f"unknown {label}: {name!r}{hint}", f"  expected one of: {', '.join(options)}"]
    if note:
        lines.append(f"  {note}")
    return SystemExit("\n".join(lines))


def existing_grid_cells(manager: CategoryManager, goal: str) -> list[tuple[str, str]]:
    """The (substrate, style) of every grid spec for ``goal`` currently in the repo.

    Filtered by goal (not just ``kind="grid"``) because relax and focus renders of the
    same (substrate, style) are different tracks — counting them together would make a
    focus coverage run think a cell is filled by relax tracks alone.
    """
    return [(e["substrate"], e["style"]) for e in manager.search(kind="grid", goal=goal)]


def existing_special_cells(manager: CategoryManager) -> list[tuple[str, str]]:
    """The (group, keyword) of every special spec currently in the repo."""
    return [(e["group"], e["keyword"]) for e in manager.search(kind="special")]


def print_axes() -> None:
    print("Substrates (Axis A):")
    for name, sub in SUBSTRATES.items():
        print(f"  {name}  [goals: {', '.join(sorted(sub.goals))}]")
    print("\nStyles (Axis B):")
    for name, sty in STYLES.items():
        print(f"  {name}  [goals: {', '.join(sorted(sty.goals))}]")
    print("\nDevelopment (progression axis), per-goal mode filter:")
    for name in GOALS:
        f = mode_filter_summary(name)
        allowed = ", ".join(sorted(f["allow_development"]))
        print(f"  {name}  [development: {allowed}]  (default: {f['default_development']})")
    print("\nDefault sample set (style:substrate):")
    for style, substrate in SAMPLE_PAIRS:
        print(f"  {style}:{substrate}")
    print("\nSpecial groups (outside the grid; GROUP plans every keyword in it):")
    for name, group in SPECIAL_GROUPS.items():
        print(f"  {name}")
        for keyword in group.keywords:
            print(f"    {name}:{keyword}")


def print_grid_coverage(
    manager: CategoryManager, substrates: list[str] | None, styles: list[str] | None, goal: str
) -> None:
    """Print the substrate × style count matrix with marginals, for one goal."""
    report = coverage_report(existing_grid_cells(manager, goal), substrates, styles, goal)
    subs = [s for s in (substrates or SUBSTRATES) if goal in SUBSTRATES[s].goals]
    stys = [s for s in (styles or STYLES) if goal in STYLES[s].goals]
    shorts = [SUBSTRATES[s].short for s in subs]
    col_w = max(6, *(len(s) for s in shorts)) + 1
    style_w = max(len(s) for s in stys) + 1

    print(" " * style_w + "".join(f"{sh:>{col_w}}" for sh in shorts) + f"{'Σ':>{col_w}}")
    for sty in stys:
        cells = "".join(f"{report['per_cell'][f'{sty}:{sub}']:>{col_w}}" for sub in subs)
        print(f"{sty:<{style_w}}{cells}{report['per_style'][sty]:>{col_w}}")
    footer = "".join(f"{report['per_substrate'][sub]:>{col_w}}" for sub in subs)
    print(f"{'Σ':<{style_w}}{footer}{report['total']:>{col_w}}")


def print_special_coverage(manager: CategoryManager, groups: list[str] | None) -> None:
    """Print per-keyword counts for each special group.

    One block per group rather than a matrix: keywords don't line up across groups
    (each belongs to exactly one), so a shared column layout would be mostly holes.
    """
    report = special_coverage_report(existing_special_cells(manager), groups)
    for name, keywords in report["keywords"].items():
        col_w = max(6, *(len(k) for k in keywords)) + 1
        name_w = max(len(g) for g in report["keywords"]) + 1
        counts = "".join(f"{report['per_cell'][f'{name}:{k}']:>{col_w}}" for k in keywords)
        print(" " * name_w + "".join(f"{k:>{col_w}}" for k in keywords) + f"{'Σ':>{col_w}}")
        print(f"{name:<{name_w}}{counts}{report['per_group'][name]:>{col_w}}")


def print_coverage(args: argparse.Namespace, manager: CategoryManager) -> None:
    """Print coverage for both taxonomies — or just the one an axis filter selects."""
    substrates, styles, groups = _split(args.substrates), _split(args.styles), _split(args.groups)
    grid_only = substrates is not None or styles is not None
    special_only = groups is not None

    if not special_only:
        print_grid_coverage(manager, substrates, styles, args.goal)
    if not grid_only:
        if not special_only:
            print("\nSpecial groups (outside the grid):")
        print_special_coverage(manager, groups)


def build_target(
    target: str, duration_s: int, goal: str, development: str | None = None
) -> list[AnySignature]:
    """One CLI target, resolved into the signatures it names.

    ``STYLE:SUBSTRATE`` is one grid cell and ``GROUP:KEYWORD`` one special cell (the
    left side disambiguates them). A bare ``GROUP`` is every keyword in that special
    group — the whole of natural_sounds in one go, since a group is small, fixed, and
    usually wanted entire. There is no bare-style equivalent: a style spans the grid,
    which is what the coverage guides are for. ``goal`` only applies to grid targets —
    special cells have no goal axis (§ background.py's ``KeywordEntry.goals`` note).
    ``development`` likewise only applies to grid targets; omitted, ``build_signature``
    resolves it to the goal's own default.
    """
    if ":" not in target:
        if target in SPECIAL_GROUPS:
            group = SPECIAL_GROUPS[target]
            return [build_keyword_signature(target, kw, duration_s) for kw in group.keywords]
        raise unknown_value(
            target,
            SPECIAL_GROUPS,
            label="special group",
            note="a bare target must name a group; grid cells are STYLE:SUBSTRATE",
        )
    left, right = target.split(":", 1)
    if left in STYLES:
        if right not in SUBSTRATES:
            raise unknown_value(right, SUBSTRATES, label=f"substrate in {target!r}")
        return [build_signature(right, left, goal, duration_s, development=development)]
    if left in SPECIAL_GROUPS:
        keywords = SPECIAL_GROUPS[left].keywords
        if right not in keywords:
            raise unknown_value(right, keywords, label=f"keyword in {target!r}")
        return [build_keyword_signature(left, right, duration_s)]
    raise unknown_value(left, [*STYLES, *SPECIAL_GROUPS], label=f"style or special group in {target!r}")


def planned_signatures(args: argparse.Namespace, manager: CategoryManager) -> list[AnySignature]:
    """Resolve the CLI into the signatures to write as specs."""
    if args.fill is not None or args.per_cell is not None:
        # --groups switches the guide's domain from the grid to special cells; the
        # two taxonomies have different axes, so a run fills one or the other.
        if args.groups:
            common = dict(
                existing_cells=existing_special_cells(manager),
                used_track_ids={e["track_id"] for e in manager.search(kind="special")},
                groups=_split(args.groups),
            )
            if args.per_cell is not None:
                return fill_special_to_per_cell(args.per_cell, args.duration, **common)
            return plan_special_coverage(args.fill, args.duration, **common)

        common = dict(
            existing_cells=existing_grid_cells(manager, args.goal),
            used_track_ids={e["track_id"] for e in manager.search(kind="grid", goal=args.goal)},
            substrates=_split(args.substrates),
            styles=_split(args.styles),
            goal=args.goal,
        )
        if args.per_cell is not None:
            return fill_to_per_cell(args.per_cell, args.duration, **common)
        return plan_coverage(args.fill, args.duration, **common)

    if args.targets:
        return [
            sig
            for target in args.targets
            for sig in build_target(target, args.duration, args.goal, args.development)
        ]

    if args.goal != "relax":
        raise SystemExit(
            "the curated sample set (no targets given) is relax-only; pass explicit "
            "STYLE:SUBSTRATE targets or --fill/--per-cell to plan focus specs"
        )
    return sample_signatures(args.duration)


def describe(spec: dict) -> str:
    if spec.get("kind", "grid") == "special":
        return f"{spec['group']}:{spec['keyword']}"
    return f"{spec['substrate']} x {spec['style']}"


def warn_orphan_tracks(manager: CategoryManager) -> None:
    """Flag rendered audio whose spec was deleted: replanning that cell reproduces the
    same deterministic track_id, which would silently adopt the stale audio."""
    orphans = manager.orphan_tracks()
    if not orphans:
        return
    print(f"warning: {len(orphans)} rendered file(s) have no spec; delete them too, or they")
    print("         will be adopted by the next spec that reuses the same track_id:")
    for path in orphans:
        print(f"           {path.relative_to(manager.root)}")
    print()


def check_axis_filters(args: argparse.Namespace) -> None:
    """Validate the coverage-guide filters before any work happens.

    Everything downstream (the report, the guides) assumes these names resolve, and
    the library layer signals a bad one with a ``ValueError`` — a traceback, not an
    error message. Catching them here means a typo'd flag fails immediately, cleanly,
    and before the catalog rebuild.
    """
    if args.groups and (args.substrates or args.styles):
        raise SystemExit("--groups selects special cells; it can't be combined with --substrates/--styles")
    if args.groups and not (args.coverage or args.fill is not None or args.per_cell is not None):
        raise SystemExit("--groups restricts a coverage guide; pass it with --fill, --per-cell or --coverage")
    if args.development is not None and (args.fill is not None or args.per_cell is not None):
        raise SystemExit(
            "--development only applies to explicit STYLE:SUBSTRATE targets, not the coverage "
            "guides (--fill/--per-cell), which plan the substrate x style grid only"
        )
    if args.development is not None and args.development not in GOALS[args.goal].allowed_development:
        raise SystemExit(
            f"goal {args.goal!r} does not allow --development {args.development!r} "
            f"(allowed: {', '.join(sorted(GOALS[args.goal].allowed_development))})"
        )

    for flag, known, label in (
        ("substrates", SUBSTRATES, "substrate"),
        ("styles", STYLES, "style"),
        ("groups", SPECIAL_GROUPS, "special group"),
    ):
        for name in _split(getattr(args, flag)) or ():
            if name not in known:
                raise unknown_value(name, known, label=f"{label} (--{flag})")


def main() -> None:
    args = parse_args()
    check_axis_filters(args)
    manager = CategoryManager()

    if args.list:
        print_axes()
        return

    # Every run starts from disk truth: the rebuild re-derives catalog.json from the
    # spec tree, so specs deleted by hand (to replan them) are already gone from the
    # coverage counts and the skip set below.
    catalog = manager.rebuild()
    warn_orphan_tracks(manager)

    if args.coverage:
        print_coverage(args, manager)
        return
    if args.rebuild_catalog:
        print(f"rebuilt catalog.json ({catalog['count']} tracks)")
        return

    existing = manager.spec_ids()
    written = skipped = 0
    try:
        signatures = planned_signatures(args, manager)
    except ValueError as exc:  # a taxonomy name check_axis_filters didn't cover
        raise SystemExit(str(exc)) from exc
    for sig in signatures:
        spec = sig.spec()
        if spec["track_id"] in existing:
            skipped += 1
            print(f"exists         {spec['track_id']}  (delete its spec to replan)")
            continue
        manager.add_spec(spec, rebuild=False)
        existing.add(spec["track_id"])
        written += 1
        print(f"planned        {spec['track_id']}  ({describe(spec)})")

    catalog = manager.rebuild()
    print(f"\n{written} specs written, {skipped} already existed; catalog: {catalog['count']} tracks")

    if args.fill is not None or args.per_cell is not None:
        print()
        print_coverage(args, manager)


if __name__ == "__main__":
    main()
