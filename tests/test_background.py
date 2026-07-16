import pytest

from bnb.background import (
    STYLES,
    SUBSTRATES,
    build_signature,
    coverage_report,
    fill_to_per_cell,
    plan_coverage,
)

BASELINE = [
    ("percussive_with_tail", "buddhist_meditative"),
    ("drone", "buddhist_meditative"),
    ("melodic_instrument", "buddhist_meditative"),
    ("field_recording", "buddhist_meditative"),
    ("noise_texture", "neutral"),
    ("melodic_instrument", "neoclassical"),
    ("melodic_instrument", "lofi"),
    ("field_recording", "nature_ambient"),
]


def test_variant_seed_is_stable_and_distinct():
    # variant 0 keeps the original identity; higher variants give new seeds.
    base = build_signature("drone", "buddhist_meditative", 60)
    assert build_signature("drone", "buddhist_meditative", 60, 0).seed == base.seed
    seeds = {build_signature("drone", "buddhist_meditative", 60, v).seed for v in range(5)}
    assert len(seeds) == 5


def test_plan_coverage_fills_empty_cells_first():
    sigs = plan_coverage(len(SUBSTRATES) * len(STYLES) - len(BASELINE), 60, existing_cells=BASELINE)
    cells = BASELINE + [(s.substrate.name, s.style.name) for s in sigs]
    report = coverage_report(cells)
    # Filling exactly the deficit levels every cell to one track.
    assert set(report["per_cell"].values()) == {1}


def test_plan_coverage_balances_marginals():
    sigs = plan_coverage(10, 60, existing_cells=BASELINE)
    report = coverage_report(BASELINE + [(s.substrate.name, s.style.name) for s in sigs])
    spread = max(report["per_style"].values()) - min(report["per_style"].values())
    assert spread <= 1  # styles stay within one of each other


def test_plan_coverage_seeds_and_ids_unique():
    sigs = plan_coverage(30, 60, existing_cells=BASELINE)
    assert len({s.track_id for s in sigs}) == len(sigs)
    assert len({s.seed for s in sigs}) == len(sigs)


def test_plan_coverage_avoids_used_track_ids():
    used = {build_signature("drone", "neutral", 60, 0).track_id}
    sig = plan_coverage(1, 60, existing_cells=[], used_track_ids=used, substrates=["drone"], styles=["neutral"])[0]
    assert sig.track_id not in used  # bumped to the next variant


def test_fill_to_per_cell_reaches_target():
    target = 3
    sigs = fill_to_per_cell(target, 60, existing_cells=BASELINE)
    report = coverage_report(BASELINE + [(s.substrate.name, s.style.name) for s in sigs])
    assert min(report["per_cell"].values()) == target


def test_coverage_can_restrict_to_subsets():
    sigs = plan_coverage(4, 60, existing_cells=[], substrates=["drone", "noise_texture"], styles=["lofi"])
    assert {s.substrate.name for s in sigs} == {"drone", "noise_texture"}
    assert {s.style.name for s in sigs} == {"lofi"}


def test_unknown_axis_names_rejected():
    with pytest.raises(ValueError):
        plan_coverage(1, 60, substrates=["gong"])
