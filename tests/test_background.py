import pytest

from bnb.background import (
    GOALS,
    SPECIAL_GROUPS,
    STYLES,
    SUBSTRATES,
    build_keyword_signature,
    build_signature,
    composition_plan_for_model,
    coverage_report,
    fill_special_to_per_cell,
    fill_to_per_cell,
    plan_coverage,
    plan_special_coverage,
    prompt_for_provider,
    special_cells,
    special_coverage_report,
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
    base = build_signature("drone", "buddhist_meditative", "relax", 60)
    assert build_signature("drone", "buddhist_meditative", "relax", 60, 0).seed == base.seed
    seeds = {build_signature("drone", "buddhist_meditative", "relax", 60, v).seed for v in range(5)}
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
    used = {build_signature("drone", "neutral", "relax", 60, 0).track_id}
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


# --- coverage over special groups ------------------------------------------------

NATURAL = list(SPECIAL_GROUPS["natural_sounds"].keywords)


def _cells(sigs):
    return [(s.group.name, s.keyword) for s in sigs]


def test_special_cells_enumerates_every_keyword():
    assert special_cells(["natural_sounds"]) == [("natural_sounds", k) for k in NATURAL]
    assert special_cells() == special_cells(["natural_sounds"])  # only group, for now


def test_plan_special_coverage_spreads_across_keywords():
    sigs = plan_special_coverage(len(NATURAL), 60)
    assert sorted(k for _, k in _cells(sigs)) == sorted(NATURAL)


def test_plan_special_coverage_tops_up_the_thinnest_keywords_first():
    existing = [("natural_sounds", "rain")] * 2
    sigs = plan_special_coverage(len(NATURAL) - 1, 60, existing_cells=existing)
    assert "rain" not in [k for _, k in _cells(sigs)]


def test_plan_special_coverage_advances_the_seed_per_keyword():
    # The only route to more than one track per keyword: variant 0 is what a plain
    # GROUP:KEYWORD target always produces.
    sigs = plan_special_coverage(len(NATURAL) * 3, 60)
    assert len({s.track_id for s in sigs}) == len(sigs)
    assert len({s.seed for s in sigs}) == len(sigs)
    assert build_keyword_signature("natural_sounds", "rain", 60).track_id in {s.track_id for s in sigs}


def test_plan_special_coverage_avoids_used_track_ids():
    used = {build_keyword_signature("natural_sounds", "rain", 60).track_id}
    sig = plan_special_coverage(1, 60, used_track_ids=used, groups=["natural_sounds"])[0]
    assert sig.track_id not in used


def test_fill_special_to_per_cell_reaches_target():
    sigs = fill_special_to_per_cell(3, 60, existing_cells=[("natural_sounds", "rain")])
    report = special_coverage_report([("natural_sounds", "rain")] + _cells(sigs))
    assert set(report["per_cell"].values()) == {3}
    assert report["per_group"]["natural_sounds"] == 3 * len(NATURAL)


def test_special_coverage_report_ignores_cells_outside_the_selection():
    report = special_coverage_report([("natural_sounds", "rain"), ("other_group", "x")])
    assert report["total"] == 1


def test_unknown_special_group_rejected():
    with pytest.raises(ValueError, match="unknown special group"):
        plan_special_coverage(1, 60, groups=["birdsong"])


def test_composition_plan_v1_is_sections():
    spec = build_signature("noise_texture", "neutral", "relax", 60).spec()
    plan = composition_plan_for_model(spec, "music_v1")
    assert "sections" in plan and "chunks" not in plan


def test_composition_plan_v2_is_instrumental_chunks():
    from elevenlabs.types import CompositionPlan  # validate against the real SDK model

    spec = build_signature("noise_texture", "neutral", "relax", 60).spec()
    plan = composition_plan_for_model(spec, "music_v2")
    CompositionPlan.model_validate(plan)
    chunk = plan["chunks"][0]
    assert 3000 <= chunk["duration_ms"] <= 120000
    assert "instrumental" in chunk["positive_styles"]
    assert "vocals" in chunk["negative_styles"]  # wordless guardrail without force_instrumental


# --- grid spec schema (kind/group/keyword unification) -----------------------


def test_grid_spec_carries_null_special_fields():
    spec = build_signature("drone", "buddhist_meditative", "relax", 60).spec()
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
    spec = build_signature("melodic_instrument", "neoclassical", "relax", 60).spec()
    assert prompt_for_provider(spec, "elevenlabs") == spec["prompt"]


def test_prompt_for_provider_stable_audio_appends_audiosparx_tags():
    spec = build_signature("melodic_instrument", "neoclassical", "relax", 60).spec()
    adapted = prompt_for_provider(spec, "stable_audio")
    assert adapted.startswith(spec["prompt"])
    assert "TrackType: Music" in adapted
    assert "VocalType: Instrumental" in adapted
    assert "Instruments: Piano" in adapted  # felt_piano instrumentation -> AudioSparx "Piano"


def test_prompt_for_provider_stable_audio_omits_instruments_when_unmapped():
    spec = build_signature("noise_texture", "neutral", "relax", 60).spec()  # brown_noise, distant_surf
    adapted = prompt_for_provider(spec, "stable_audio")
    assert "Instruments:" not in adapted


def test_prompt_for_provider_stable_audio_slates_special_cells_as_sfx():
    # A field recording is a sound effect, not music: the music tags are what made the
    # first natural_sounds renders come back as ambient drones.
    spec = build_keyword_signature("natural_sounds", "rain", 60).spec()
    adapted = prompt_for_provider(spec, "stable_audio")
    assert adapted.endswith("TrackType: SFX")
    assert "TrackType: Music" not in adapted
    assert "Genre: Ambient" not in adapted


def test_special_prompts_lead_with_the_subject():
    # The keyword, not 50 words of MER vocabulary, is the first thing the encoder sees.
    for keyword in SPECIAL_GROUPS["natural_sounds"].keywords:
        prompt = build_keyword_signature("natural_sounds", keyword, 60).prompt
        assert "no music" in prompt
        assert prompt.split(",")[0].strip()  # opens on the source, not a genre label
    assert build_keyword_signature("natural_sounds", "rain", 60).prompt.startswith("Steady soft rainfall")


def test_every_prompt_bounds_how_busy_it_can_get():
    # The failure this guards: "forest" arriving as a full dawn chorus. Density is
    # capped in the positive prompt because the negative one only reaches the model
    # under guidance, which the distilled checkpoints don't use.
    # Event-driven sounds are told to space events out...
    for prompt in (
        build_signature("percussive_with_tail", "buddhist_meditative", "relax", 60).prompt,
        build_keyword_signature("natural_sounds", "forest", 60).prompt,
    ):
        assert "at most one or two things sounding" in prompt
    # ...while a continuous bed is told to stay even, because asking a cricket wash for
    # "long stretches of near-stillness" breaks it into discrete chirps instead.
    for prompt in (
        build_signature("noise_texture", "neutral", "relax", 60).prompt,
        build_keyword_signature("natural_sounds", "rain", 60).prompt,
        build_keyword_signature("natural_sounds", "night", 60).prompt,
    ):
        assert "Even and unbroken throughout" in prompt
        assert "near-stillness" not in prompt
    assert "no dawn chorus" in build_keyword_signature("natural_sounds", "forest", 60).prompt


def test_grid_prompts_carry_motion_and_recording_character():
    # What keeps a 60-minute listen from going flat — and the axes stay untouched.
    prompt = build_signature("drone", "lofi", "relax", 60).prompt
    assert "evolves slowly" in prompt
    assert "tape saturation" in prompt  # the lofi style's character clause
    assert "very low energy" in prompt  # the MER axes still there
    assert "No vocals, no percussion hits" in prompt


# --- goal axis (relax / focus) ------------------------------------------------


def test_unknown_goal_rejected():
    with pytest.raises(ValueError, match="unknown goal"):
        build_signature("drone", "lofi", "gamma", 60)


def test_relax_only_substrate_rejects_focus():
    with pytest.raises(ValueError, match="percussive_with_tail.*focus"):
        build_signature("percussive_with_tail", "neutral", "focus", 60)


def test_relax_only_style_rejects_focus():
    with pytest.raises(ValueError, match="buddhist_meditative.*focus"):
        build_signature("drone", "buddhist_meditative", "focus", 60)


def test_track_id_always_includes_goal():
    relax = build_signature("drone", "lofi", "relax", 60)
    focus = build_signature("drone", "lofi", "focus", 60)
    assert relax.track_id == f"lofi_drone_relax_seed{relax.seed}"
    assert focus.track_id == f"lofi_drone_focus_seed{focus.seed}"
    assert relax.track_id != focus.track_id


def test_relax_and_focus_signatures_of_the_same_cell_get_distinct_seeds():
    relax = build_signature("drone", "lofi", "relax", 60)
    focus = build_signature("drone", "lofi", "focus", 60)
    assert relax.seed != focus.seed


def test_focus_prompt_differs_from_relax_in_motion_and_dynamics():
    relax = build_signature("melodic_instrument", "lofi", "relax", 60)
    focus = build_signature("melodic_instrument", "lofi", "focus", 60)
    assert "never builds, resolves, or arrives anywhere" in relax.prompt
    assert "repeats in a steady, unsurprising loop" in focus.prompt
    assert relax.negative_prompt == GOALS["relax"].negative_prompt
    assert focus.negative_prompt == GOALS["focus"].negative_prompt
    assert "lyrics" in focus.negative_prompt
    # Identity (what the sound is) stays put across goals; only arousal changes.
    assert relax.instrumentation == focus.instrumentation


def test_focus_overrides_only_touch_declared_fields():
    # noise_texture declares a focus_overrides dict; body/instrumentation (never in
    # that dict) must be untouched, only the arousal fields it lists should move.
    relax_sub = SUBSTRATES["noise_texture"]
    focus_spec = build_signature("noise_texture", "neutral", "focus", 60).spec()
    assert focus_spec["requested_features"]["energy"] == "low"  # overridden from very_low
    assert relax_sub.requested["energy"] == "very_low"  # the base definition is untouched


def test_spec_records_goal():
    assert build_signature("drone", "lofi", "focus", 60).spec()["goal"] == "focus"
    assert build_signature("drone", "lofi", "relax", 60).spec()["goal"] == "relax"


def test_keyword_signature_has_no_goal_key():
    # Special groups are goal-agnostic (metadata-only, see KeywordEntry.goals) —
    # the render doesn't fork by goal, so the spec carries no "goal" at all.
    spec = build_keyword_signature("natural_sounds", "rain", 60).spec()
    assert "goal" not in spec


def test_coverage_functions_restrict_grid_to_the_requested_goal():
    # percussive_with_tail / buddhist_meditative never appear in a focus plan.
    sigs = plan_coverage(50, 60, goal="focus")
    assert all(s.substrate.name != "percussive_with_tail" for s in sigs)
    assert all(s.style.name != "buddhist_meditative" for s in sigs)
    assert all(s.goal.name == "focus" for s in sigs)


def test_coverage_report_counts_relax_and_focus_separately():
    relax_cells = [("drone", "lofi")]
    focus_cells = [("drone", "lofi"), ("drone", "lofi")]
    relax_report = coverage_report(relax_cells, goal="relax")
    focus_report = coverage_report(focus_cells, goal="focus")
    assert relax_report["per_cell"]["lofi:drone"] == 1
    assert focus_report["per_cell"]["lofi:drone"] == 2


def test_fill_to_per_cell_respects_goal_restricted_grid():
    # Asking for full coverage of a goal-restricted grid must not try to place
    # percussive_with_tail/buddhist_meditative cells for focus.
    sigs = fill_to_per_cell(1, 60, goal="focus")
    cells = {(s.substrate.name, s.style.name) for s in sigs}
    expected = {
        (sub, sty)
        for sub in SUBSTRATES
        if "focus" in SUBSTRATES[sub].goals
        for sty in STYLES
        if "focus" in STYLES[sty].goals
    }
    assert cells == expected
