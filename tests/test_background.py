import pytest

from bnb.background import (
    NEGATIVE_PROMPT,
    _keyword_prompt as rb_keyword_prompt,
    DEVELOPMENT_FRAGMENT,
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
    mode_filter_summary,
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
ENERGIZER = list(SPECIAL_GROUPS["energizer"].keywords)


def _cells(sigs):
    return [(s.group.name, s.keyword) for s in sigs]


def test_special_cells_enumerates_every_keyword():
    assert special_cells(["natural_sounds"]) == [("natural_sounds", k) for k in NATURAL]
    assert special_cells(["energizer"]) == [("energizer", k) for k in ENERGIZER]
    assert special_cells() == special_cells(["natural_sounds"]) + special_cells(["energizer"])


def test_plan_special_coverage_spreads_across_keywords():
    sigs = plan_special_coverage(len(NATURAL), 60, groups=["natural_sounds"])
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
        # universe words this differently ("A sound, not music: ..."), but every keyword
        # in this group must still tell the encoder it is not making music.
        assert "no music" in prompt or "not music" in prompt
        assert prompt.split(",")[0].strip()  # opens on the source, not a genre label
    assert build_keyword_signature("natural_sounds", "rain", 60).prompt.startswith("Steady soft rainfall")


def test_every_prompt_bounds_how_busy_it_can_get():
    # The failure this guards: "forest" arriving as a full dawn chorus. Density is
    # capped in the positive prompt because the negative one only reaches the model
    # under guidance, which the distilled checkpoints don't use. Every prompt keeps a
    # *countable* cap; which cap depends on how the sound is made and how much it develops.
    # Event-driven sounds with no melody to play are told to space events out...
    for prompt in (
        build_signature("percussive_with_tail", "buddhist_meditative", "relax", 60, development="static").prompt,
        build_keyword_signature("natural_sounds", "forest", 60).prompt,
    ):
        assert "at most one or two things sounding" in prompt
    # ...while a continuous bed is told to stay even, because asking a cricket wash for
    # "long stretches of near-stillness" breaks it into discrete chirps instead.
    for prompt in (
        build_signature("noise_texture", "neutral", "relax", 60, development="static").prompt,
        build_keyword_signature("natural_sounds", "rain", 60).prompt,
        build_keyword_signature("natural_sounds", "night", 60).prompt,
    ):
        assert "Even and unbroken throughout" in prompt
        assert "near-stillness" not in prompt
    assert "no dawn chorus" in build_keyword_signature("natural_sounds", "forest", 60).prompt


def test_melodic_development_swaps_the_stillness_bound_for_a_flowing_one():
    # A tune needs room to be played: the stillness clauses ("long stretches of
    # near-stillness", "nothing stepping forward out of it, no separate events and no
    # layering") forbid a phrase from continuing, whatever `development` asks for — which
    # is why the un-banning in Change 1 produced no audible melody on its own.
    for substrate in ("melodic_instrument", "drone"):  # event-driven and not
        prompt = build_signature(substrate, "lofi", "relax", 60).prompt  # default slow_swell
        assert "Full but unhurried" in prompt
        assert "near-stillness" not in prompt
        assert "no separate events and no layering" not in prompt
        assert "three or four gentle layers" in prompt  # the cap survives, just looser


def test_melodic_development_relaxes_the_prose_texture_word_only():
    # "very sparse texture" in the same prompt as "three or four gentle layers" is the
    # same self-contradiction that made lofi unlistenable — but the MER coordinate the
    # bandit consumes must not move, because that is the cell's address in the taxonomy.
    sig = build_signature("drone", "lofi", "relax", 60)  # default slow_swell
    assert "uncluttered but full texture" in sig.prompt
    assert "very sparse texture" not in sig.prompt
    assert sig.requested_features["texture_density"] == "very_sparse"
    # static keeps the original word, and the coordinate is identical either way.
    still = build_signature("drone", "lofi", "relax", 60, development="static")
    assert "very sparse texture" in still.prompt
    assert still.requested_features["texture_density"] == sig.requested_features["texture_density"]


def test_the_two_goals_differ_in_the_sentences_that_actually_carry_the_music():
    # The regression this guards: relax and focus renders coming back indistinguishable.
    # Tempo/timbre adjectives were the only fork, while the development fragment and the
    # density clause — the two most musically salient sentences — were byte-identical.
    relax = build_signature("melodic_instrument", "lofi", "relax", 60)
    focus = build_signature("melodic_instrument", "lofi", "focus", 60)
    assert DEVELOPMENT_FRAGMENT[relax.development] not in focus.prompt
    assert DEVELOPMENT_FRAGMENT[focus.development] not in relax.prompt
    assert GOALS["relax"].flowing not in focus.prompt
    assert GOALS["focus"].flowing not in relax.prompt
    # Focus is the fuller of the two, and carries an actual tune.
    assert "four or five layers" in focus.prompt and "three or four gentle layers" in relax.prompt
    assert "melody" in focus.prompt


def test_focus_beds_are_gap_free_because_they_carry_an_am_carrier():
    # Not a taste call: focus ships as am_music, where the bed *is* the carrier the
    # entrainment envelope multiplies. A gap in the bed is a gap in the stimulus.
    for substrate in SUBSTRATES:
        if "focus" not in SUBSTRATES[substrate].goals:
            continue
        prompt = build_signature(substrate, "neutral", "focus", 60).prompt
        assert "never drops to silence" in prompt
        assert "present at every single moment" in prompt
        for phrase in ("near-stillness", "no separate events and no layering"):
            assert phrase not in prompt, f"focus/{substrate}: {phrase!r}"


def test_relax_is_no_longer_authored_at_the_extreme_of_every_axis():
    # "very soft" + "warm and dark, low spectral brightness" + very-low energy + sparse
    # stacks into muffled rather than restful.
    relax = GOALS["relax"]
    assert "very soft" not in relax.dynamics
    assert "low spectral brightness" not in relax.brightness
    assert "clarity" in relax.brightness
    # ...but it is still unambiguously the down-regulating side of the pair.
    assert "never bright or harsh" in relax.brightness
    assert "warm" in relax.brightness


def test_keyword_nature_beds_never_get_the_flowing_bound():
    # They have no development axis at all; a rain bed must never grow layers.
    for keyword in ("rain", "forest", "night"):
        assert "Full but unhurried" not in build_keyword_signature("natural_sounds", keyword, 60).prompt


def test_grid_prompts_carry_motion_and_recording_character():
    # What keeps a 60-minute listen from going flat — and the axes stay untouched.
    prompt = build_signature("drone", "lofi", "relax", 60).prompt
    assert "swells and recedes" in prompt  # relax default development=slow_swell
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


def test_focus_prompt_differs_from_relax_in_dynamics_and_negative_prompt():
    relax = build_signature("melodic_instrument", "lofi", "relax", 60)
    focus = build_signature("melodic_instrument", "lofi", "focus", 60)
    assert "soft, even dynamics" in relax.prompt
    assert "smooth, controlled dynamics" in focus.prompt
    assert relax.negative_prompt == GOALS["relax"].negative_prompt
    assert focus.negative_prompt == GOALS["focus"].negative_prompt
    assert "lyrics" in focus.negative_prompt
    assert "melodic hook" not in focus.negative_prompt  # Change 1: no longer globally banned
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


# --- special groups ------------------------------------------------------------


def test_natural_sounds_has_the_new_keywords():
    keywords = SPECIAL_GROUPS["natural_sounds"].keywords
    for name in ("universe", "fireplace"):
        assert name in keywords
        # Neither is event-driven. A fire is a flame roar *plus* crackles, and the
        # event-driven clause would demand "long stretches of near-stillness", detaching
        # the pops from the bed — the cricket-wash failure. Spacing lives in the wording.
        assert keywords[name].event_driven is False
        assert "near-stillness" not in build_keyword_signature("natural_sounds", name, 60).prompt
    assert "occasional gentle crackle" in keywords["fireplace"].description


def test_fireplace_takes_the_groups_stillness_bound():
    prompt = build_keyword_signature("natural_sounds", "fireplace", 60).prompt
    assert "Even and unbroken throughout" in prompt


def test_universe_overrides_the_group_to_actually_move():
    # It sits in a still, no-music shell while needing the opposite. Left on the group's
    # defaults it came back as one undifferentiated hum: "nothing stepping forward out of
    # it, no separate events and no layering" plus "changing almost imperceptibly".
    entry = SPECIAL_GROUPS["natural_sounds"].keywords["universe"]
    assert entry.flowing is not None and entry.body is not None
    prompt = build_keyword_signature("natural_sounds", "universe", 60).prompt
    assert "changing almost imperceptibly" not in prompt
    assert "Even and unbroken throughout" not in prompt  # the group's stillness bound
    assert "nothing stepping forward out of it" not in prompt
    assert "It travels" in prompt and "never still and never uniform" in prompt
    assert "one distinct passage giving way to the next" in prompt
    # ...and it stays a sound rather than becoming music, by banning the machinery
    # instead of the tonal content the sweeps are made of.
    assert "no beat, no drums, no chord progression" in prompt
    assert "sweeping tones rising and falling" in prompt
    # The rest of the group is untouched by the override.
    assert SPECIAL_GROUPS["natural_sounds"].keywords["rain"].flowing is None


def test_natural_sounds_bounds_the_rate_not_just_the_character():
    # "Calm and unhurried" constrains character but says nothing about *rate*, so a bed
    # could be calm in timbre and still patter or chirp away continuously.
    for keyword in SPECIAL_GROUPS["natural_sounds"].keywords:
        entry = SPECIAL_GROUPS["natural_sounds"].keywords[keyword]
        if entry.body is not None:
            continue  # universe carries its own shell; see its own test
        prompt = build_keyword_signature("natural_sounds", keyword, 60).prompt
        assert "infrequent and widely spaced" in prompt
        assert "the overall rate stays low and never picks up" in prompt


def test_energizer_prompts_meet_every_stated_requirement():
    for keyword in ENERGIZER:
        sig = build_keyword_signature("energizer", keyword, 60)
        prompt = sig.prompt
        # 1. a real continuous melody, enjoyable, mainstream — not a drone
        assert "melod" in prompt
        assert "familiar to an ordinary listener" in prompt
        # 2. medium-pace beat, but not loud
        assert "never loud, never intense and never dramatic" in prompt
        # 3. some energy, not the library's floor
        assert "Moderate, steady energy" in prompt
        assert "awake and warm rather than sleepy" in prompt
        # 4. a continuous body an AM carrier can ride on
        assert "present at every single moment" in prompt
        assert "never thins out, never drops to silence" in prompt
        # ...and it says so once, not three times over.
        assert prompt.count("never drops to silence") == 1


def test_energizer_is_not_saddled_with_the_down_regulation_negatives():
    # The shared relax negatives ban "energetic, fast, drums, EDM, bright" — this group's
    # entire purpose. What has to stay out is drama and volume, plus gaps (an AM fault).
    negative = SPECIAL_GROUPS["energizer"].negative_prompt
    for wanted in ("energetic", "fast", "drums", "EDM", "bright"):
        assert wanted not in negative, f"energizer must not ban {wanted!r}"
    for banned in ("dramatic climax", "buildup", "drop", "loud", "silence", "gaps"):
        assert banned in negative
    # ...while natural_sounds keeps the relax pair it always had.
    assert SPECIAL_GROUPS["natural_sounds"].negative_prompt == NEGATIVE_PROMPT


def test_special_group_negatives_reach_the_composition_plan_too():
    # Not just the prose prompt: ElevenLabs takes its direction from the plan.
    plan = build_keyword_signature("energizer", "uplift", 60).composition_plan
    assert plan["negative_global_styles"] == list(SPECIAL_GROUPS["energizer"].negative_global_styles)
    assert "silence" in plan["negative_global_styles"]
    assert "drums" not in plan["negative_global_styles"]
    assert plan["sections"][0]["negative_local_styles"] == plan["negative_global_styles"]


def test_energizer_bed_is_asked_to_develop_not_merely_to_keep_playing():
    # Continuous != developing. The continuity clauses ask the bed to keep *playing*;
    # nothing asked it to go anywhere — the same gap that made un-banning melody on the
    # grid produce no melody. It borrows the grid's axis rather than restating it.
    group = SPECIAL_GROUPS["energizer"]
    assert group.development == GOALS["focus"].default_development == "motif_evolving"
    for keyword in ENERGIZER:
        prompt = build_keyword_signature("energizer", keyword, 60).prompt
        assert DEVELOPMENT_FRAGMENT["motif_evolving"] in prompt
        assert "changing audibly every few bars" in prompt
        assert "never a single held note or a static wash" in prompt
    # natural_sounds has no progression clause at all.
    assert SPECIAL_GROUPS["natural_sounds"].development is None
    assert DEVELOPMENT_FRAGMENT["motif_evolving"] not in build_keyword_signature(
        "natural_sounds", "rain", 60
    ).prompt


def test_special_group_development_must_be_a_known_value():
    from dataclasses import replace

    bad = replace(SPECIAL_GROUPS["energizer"], development="bogus")
    with pytest.raises(ValueError, match="unknown development"):
        rb_keyword_prompt(bad, bad.keywords["uplift"])


def test_energizer_reuses_the_focus_goals_continuity_clause_verbatim():
    # This group *is* the focus pack; one wording means one thing to keep true.
    assert SPECIAL_GROUPS["energizer"].flowing == GOALS["focus"].flowing
    assert SPECIAL_GROUPS["natural_sounds"].flowing is None


def test_energizer_keywords_are_focus_only():
    for keyword, entry in SPECIAL_GROUPS["energizer"].keywords.items():
        assert entry.goals == frozenset({"focus"}), keyword


def test_energizer_cells_get_distinct_ids_and_seeds():
    ids = {k: build_keyword_signature("energizer", k, 60) for k in ENERGIZER}
    assert len({s.track_id for s in ids.values()}) == len(ENERGIZER)
    assert len({s.seed for s in ids.values()}) == len(ENERGIZER)
    for keyword, sig in ids.items():
        assert sig.track_id.startswith(f"energizer_{keyword}_seed")
        assert sig.spec()["kind"] == "special" and sig.spec()["group"] == "energizer"


# --- development axis (progression) -------------------------------------------


def test_development_defaults_per_goal():
    relax = build_signature("drone", "lofi", "relax", 60)
    focus = build_signature("drone", "lofi", "focus", 60)
    assert relax.development == GOALS["relax"].default_development == "slow_swell"
    # Up-regulation defaults to the melodic end — sharing relax's default was what made the
    # two goals near-indistinguishable, since this drives the prompt's most salient sentence.
    assert focus.development == GOALS["focus"].default_development == "motif_evolving"
    assert relax.spec()["requested_features"]["development"] == "slow_swell"
    assert focus.spec()["requested_features"]["development"] == "motif_evolving"


def test_development_gated_by_goal():
    # relax admits static/slow_swell but not motif_evolving.
    with pytest.raises(ValueError, match="relax.*motif_evolving"):
        build_signature("drone", "lofi", "relax", 60, development="motif_evolving")
    # focus admits slow_swell/motif_evolving but not static.
    with pytest.raises(ValueError, match="focus.*static"):
        build_signature("drone", "lofi", "focus", 60, development="static")
    # both directions succeed within their own allow-list.
    assert build_signature("drone", "lofi", "relax", 60, development="static").development == "static"
    assert (
        build_signature("drone", "lofi", "focus", 60, development="motif_evolving").development
        == "motif_evolving"
    )


def test_unknown_development_rejected():
    with pytest.raises(ValueError, match="unknown development"):
        build_signature("drone", "lofi", "relax", 60, development="bogus")


def test_development_fragment_lands_in_the_prompt():
    for value, fragment in DEVELOPMENT_FRAGMENT.items():
        goal = "relax" if value in GOALS["relax"].allowed_development else "focus"
        prompt = build_signature("drone", "lofi", goal, 60, development=value).prompt
        assert fragment in prompt


def test_default_development_track_id_unchanged():
    # The no-argument call must reproduce the exact pre-axis track_id/seed for every
    # already-planned/rendered track — the backward-compatibility guarantee.
    sig = build_signature("drone", "buddhist_meditative", "relax", 60)
    assert sig.track_id == f"buddhist_meditative_drone_relax_seed{sig.seed}"
    assert "_dev" not in sig.track_id


def test_explicit_development_gets_distinct_seed_and_track_id():
    default = build_signature("drone", "lofi", "relax", 60)
    explicit = build_signature("drone", "lofi", "relax", 60, development="static")
    assert explicit.track_id != default.track_id
    assert explicit.seed != default.seed
    assert explicit.track_id.endswith("_devstatic")
    # Requesting the goal's own default explicitly is identical to omitting it.
    same = build_signature("drone", "lofi", "relax", 60, development="slow_swell")
    assert same.track_id == default.track_id
    assert same.seed == default.seed


# --- mode filter summary (Change 3) --------------------------------------------


def test_mode_filter_summary_matches_axis_allow_lists():
    for goal_name in GOALS:
        summary = mode_filter_summary(goal_name)
        assert summary["allow_substrate"] == {n for n, s in SUBSTRATES.items() if goal_name in s.goals}
        assert summary["allow_style"] == {n for n, s in STYLES.items() if goal_name in s.goals}
        assert summary["allow_development"] == set(GOALS[goal_name].allowed_development)
        assert summary["default_development"] == GOALS[goal_name].default_development


def test_mode_filter_summary_rejects_unknown_goal():
    with pytest.raises(ValueError, match="unknown goal"):
        mode_filter_summary("gamma")


def test_no_substrate_hardcodes_a_melody_ban_that_fights_the_development_axis():
    # Change 1's whole point: nothing should hard-suppress melody/harmonic movement
    # anymore, at any goal — that's now development's job (gated per-goal, Change 2).
    # Regression guard for SUBSTRATES["drone"].focus_overrides["harmony"] having said
    # "static, no melodic development", which silently fought development=motif_evolving.
    banned_phrases = ("no melodic development", "no melodic hook", "melodic hook")
    for name, substrate in SUBSTRATES.items():
        for goal_name in substrate.goals:  # "neutral" style supports every goal
            sig = build_signature(name, "neutral", goal_name, 60)
            for phrase in banned_phrases:
                assert phrase not in sig.prompt, f"{name}/{goal_name}: {phrase!r} in prompt"
                assert phrase not in sig.negative_prompt, f"{name}/{goal_name}: {phrase!r} in negative_prompt"


def test_focus_drone_allows_motif_evolving_without_self_contradiction():
    sig = build_signature("drone", "lofi", "focus", 60, development="motif_evolving")
    assert "A clear, pleasant melody plays over a slow chord progression" in sig.prompt
    assert "no melodic development" not in sig.prompt


def test_development_asks_for_pitch_not_just_amplitude():
    # The original bug: every fragment was a *loudness* instruction ("swelling and
    # receding"), so nothing in the prompt ever asked for notes. Both moving values must
    # now name harmonic content, and say it pleasantly — a model given only "harmonic
    # movement" drifts somewhere sour as readily as somewhere warm.
    for value in ("slow_swell", "motif_evolving"):
        fragment = DEVELOPMENT_FRAGMENT[value]
        assert "chord progression" in fragment
        assert "pleasant" in fragment
        assert "consonant" in fragment or "resolving" in fragment


def test_nothing_in_a_grid_prompt_still_bans_progression():
    # Companion to the melody-ban guard above: Change 1 removed "melodic hook, catchy
    # melody" from the focus negative prompt but left "key change, chord progression",
    # which banned in as many words the thing the development axis exists to request.
    for goal_name in GOALS:
        for value in sorted(GOALS[goal_name].allowed_development):
            sig = build_signature("melodic_instrument", "lofi", goal_name, 60, development=value)
            for phrase in ("no key changes", "key change", "chord progression"):
                assert phrase not in sig.negative_prompt, f"{goal_name}/{value}: {phrase!r}"
    # ...and drama stays banned on focus, which is what actually hurts sustained attention.
    for phrase in ("dramatic climax", "buildup", "drop", "sudden transitions"):
        assert phrase in GOALS["focus"].negative_prompt


def test_lofi_does_not_ask_for_the_noise_its_negative_prompt_bans():
    # It read as unpleasant noise because its character *was* a noise floor while the
    # density clauses removed the music that floor sits under — and it asked for hiss
    # that NEGATIVE_PROMPT bans, so the style fought its own negative prompt. Scoped to
    # the text lofi itself contributes: a substrate may still legitimately *negate* noise
    # (noise_texture's focus timbre says "no harsh hiss"), which is the opposite problem.
    lofi = STYLES["lofi"]
    contributed = " ".join(
        (lofi.character, " ".join(lofi.global_styles), *(o.body for o in lofi.overrides.values()))
    ).lower()
    for phrase in ("noise floor", "hiss", "white noise", "dusty"):
        assert phrase not in contributed, f"lofi still asks for {phrase!r}"
    assert "tape" in contributed  # the medium, not the noise, is what makes lo-fi pleasant
    # Every substrate lofi supports gets real chords from it, not just grain.
    for substrate in ("drone", "melodic_instrument"):
        assert substrate in lofi.overrides
        assert "chord" in lofi.overrides[substrate].body.lower()
