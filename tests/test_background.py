import pytest

from bnb.background import (
    SPECIAL_GROUPS,
    STYLES,
    SUBSTRATES,
    build_keyword_signature,
    build_signature,
    composition_plan_for_model,
    coverage_report,
    fill_to_per_cell,
    plan_coverage,
    prompt_for_provider,
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


def test_composition_plan_v1_is_sections():
    spec = build_signature("noise_texture", "neutral", 60).spec()
    plan = composition_plan_for_model(spec, "music_v1")
    assert "sections" in plan and "chunks" not in plan


def test_composition_plan_v2_is_instrumental_chunks():
    from elevenlabs.types import CompositionPlan  # validate against the real SDK model

    spec = build_signature("noise_texture", "neutral", 60).spec()
    plan = composition_plan_for_model(spec, "music_v2")
    CompositionPlan.model_validate(plan)
    chunk = plan["chunks"][0]
    assert 3000 <= chunk["duration_ms"] <= 120000
    assert "instrumental" in chunk["positive_styles"]
    assert "vocals" in chunk["negative_styles"]  # wordless guardrail without force_instrumental


# --- grid spec schema (kind/group/keyword unification) -----------------------


def test_grid_spec_carries_null_special_fields():
    spec = build_signature("drone", "buddhist_meditative", 60).spec()
    assert spec["kind"] == "grid"
    assert spec["group"] is None and spec["keyword"] is None
    assert spec["substrate"] == "drone" and spec["style"] == "buddhist_meditative"


# --- special cells (keyword-driven categories outside the grid) --------------


def test_keyword_signature_resolves_prompt_and_track_id():
    sig = build_keyword_signature("natural_sounds", "rain", 60)
    assert sig.track_id == f"natural_sounds_rain_seed{sig.seed}"
    assert "rain" in sig.prompt.lower()
    spec = sig.spec()
    assert spec["kind"] == "special"
    assert spec["substrate"] is None and spec["style"] is None
    assert spec["group"] == "natural_sounds" and spec["keyword"] == "rain"
    assert spec["duration_s"] == 60


def test_keyword_signature_seed_is_stable_and_distinct_per_variant():
    base = build_keyword_signature("natural_sounds", "rain", 60)
    assert build_keyword_signature("natural_sounds", "rain", 60, 0).seed == base.seed
    seeds = {build_keyword_signature("natural_sounds", "rain", 60, v).seed for v in range(4)}
    assert len(seeds) == 4


def test_keyword_signature_rejects_unknown_group_or_keyword():
    with pytest.raises(ValueError, match="unknown special group"):
        build_keyword_signature("bogus_group", "rain", 60)
    with pytest.raises(ValueError, match="unknown keyword"):
        build_keyword_signature("natural_sounds", "bogus_keyword", 60)


def test_keyword_signature_composition_plan_forces_instrumental():
    spec = build_keyword_signature("natural_sounds", "ocean", 60).spec()
    plan = composition_plan_for_model(spec, "music_v2")
    chunk = plan["chunks"][0]
    assert "instrumental" in chunk["positive_styles"]
    assert "vocals" in chunk["negative_styles"]


def test_every_special_group_keyword_builds_without_error():
    for group_name, group in SPECIAL_GROUPS.items():
        for keyword in group.keywords:
            sig = build_keyword_signature(group_name, keyword, 30)
            assert sig.prompt and sig.negative_prompt


# --- provider prompt adaptation ----------------------------------------------


def test_prompt_for_provider_elevenlabs_is_passthrough():
    spec = build_signature("melodic_instrument", "neoclassical", 60).spec()
    assert prompt_for_provider(spec, "elevenlabs") == spec["prompt"]


def test_prompt_for_provider_stable_audio_appends_audiosparx_tags():
    spec = build_signature("melodic_instrument", "neoclassical", 60).spec()
    adapted = prompt_for_provider(spec, "stable_audio")
    assert adapted.startswith(spec["prompt"])
    assert "TrackType: Music" in adapted
    assert "VocalType: Instrumental" in adapted
    assert "Instruments: Piano" in adapted  # felt_piano instrumentation -> AudioSparx "Piano"


def test_prompt_for_provider_stable_audio_omits_instruments_when_unmapped():
    spec = build_signature("noise_texture", "neutral", 60).spec()  # brown_noise, distant_surf
    adapted = prompt_for_provider(spec, "stable_audio")
    assert "Instruments:" not in adapted


def test_prompt_for_provider_stable_audio_uses_keyword_entry_instrument():
    spec = build_keyword_signature("natural_sounds", "chimes", 60).spec()
    adapted = prompt_for_provider(spec, "stable_audio")
    assert "Instruments: Chimes" in adapted
