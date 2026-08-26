import json

import pytest
from fastapi.testclient import TestClient

from bnb.background import GOALS, soundscape_problems
from bnb.profiles import (
    BEAT_MODES,
    GOAL_HUES,
    MAX_BADGES,
    SOURCES,
    gradient_matches_goal,
    load_profiles,
    validate_profiles,
)
from bnb.server import app

client = TestClient(app)


def card(**overrides):
    """A minimal valid card, for tests that break exactly one thing about it."""
    return {
        "id": "x",
        "title": "X",
        "source": "community",
        "gradient": "linear-gradient(135deg, #4a56b8, #1d2050)",
        "spec": {"goal": "relax", "mode": "binaural"},
    } | overrides


def test_shipped_profiles_are_valid():
    validate_profiles(load_profiles())  # the real assets/profiles.json cross-checks the taxonomy


def test_endpoint_returns_the_profile_catalog():
    res = client.get("/api/profiles")
    assert res.status_code == 200
    ids = [p["id"] for p in res.json()["profiles"]]
    assert ids == [p["id"] for p in load_profiles()]  # served list == file on disk


def test_the_endpoint_serves_personal_cards_too():
    """Only the cloud build curates by source; the server has no bucket, so it returns
    the authored file whole and lets the caller decide."""
    shipped = load_profiles() + [card(id="p", source="personal", manual=True, spec=None)]
    validate_profiles(shipped)


def test_every_profile_is_well_formed():
    for p in load_profiles():
        assert p["id"] and p["title"]
        assert p["source"] in SOURCES
        assert len(p.get("badges") or []) <= MAX_BADGES
        # image and gradient are both optional: a card with neither gets a gradient
        # generated from its goal on the client (§ connect.ts gradientFor)
        assert p.get("image") or p.get("gradient") or (p.get("spec") or {}).get("goal")
        spec = p.get("spec")
        if spec is None:
            assert p.get("manual"), f"{p['id']} has no spec but isn't the manual card"
        else:
            assert spec["goal"] in GOALS
            assert "type" not in spec, f"{p['id']} still uses spec.type; it is spec.soundscape now"
            for keyword in spec.get("soundscape") or []:
                assert soundscape_problems(keyword) == []
            if spec.get("mode") is not None:
                assert spec["mode"] in BEAT_MODES


def test_a_card_may_leave_its_gradient_out():
    """Colour is optional: the client generates one from the goal (§ connect.ts
    gradientFor), so a card that has no identity of its own doesn't have to invent one —
    and can't invent one that contradicts its goal."""
    bare = card()
    bare.pop("gradient")
    validate_profiles([bare])


def test_profiles_are_reread_each_call(tmp_path):
    """Editing the JSON takes effect without a restart: two reads of a file that changed
    in between return the two different contents."""
    p = tmp_path / "profiles.json"
    p.write_text(json.dumps({"profiles": [card(id="a", manual=True, spec=None)]}))
    assert [x["id"] for x in load_profiles(tmp_path)] == ["a"]
    p.write_text(json.dumps({"profiles": [card(id="b", manual=True, spec=None)]}))
    assert [x["id"] for x in load_profiles(tmp_path)] == ["b"]


def test_load_accepts_a_bare_list(tmp_path):
    (tmp_path / "profiles.json").write_text(json.dumps([card(id="a", manual=True, spec=None)]))
    assert [x["id"] for x in load_profiles(tmp_path)] == ["a"]


def test_endpoint_500s_on_a_broken_config(tmp_path, monkeypatch):
    """A malformed profiles.json surfaces as a 500 with the reason, not an empty grid."""
    import bnb.profiles as profiles_mod

    bad = tmp_path / "profiles.json"
    bad.write_text(json.dumps({"profiles": [card(spec={"goal": "gamma"})]}))
    monkeypatch.setattr(profiles_mod, "profiles_path", lambda root=None: bad)
    res = client.get("/api/profiles")
    assert res.status_code == 500
    assert "profiles config error" in res.json()["detail"]


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param([card(spec={"goal": "gamma"})], id="unknown goal"),
        pytest.param(
            [card(spec={"goal": "relax", "soundscape": ["techno"]})], id="unknown soundscape term"
        ),
        pytest.param(
            [card(spec={"goal": "relax", "soundscape": ["drone.noise_texture"]})],
            id="soundscape pair no track can carry",
        ),
        pytest.param(
            [card(spec={"goal": "relax", "soundscape": "drone"})], id="soundscape not a list"
        ),
        pytest.param(
            [card(spec={"goal": "relax", "type": "drone"})], id="legacy spec.type is rejected"
        ),
        pytest.param([card(spec=None)], id="no spec, not manual"),
        pytest.param([card(id="x"), card(id="x")], id="duplicate id"),
        pytest.param([card(badges=[{"text": str(n)} for n in range(4)])], id="too many badges"),
        pytest.param([card(source=None)], id="missing source"),
        pytest.param([card(source="curated")], id="unknown source"),
        pytest.param([card(spec={"goal": "relax", "mode": "am_music"})], id="mode the app can't build"),
        pytest.param(
            [card(spec={"goal": "focus", "mode": "binaural"})],  # blue gradient, focus goal
            id="gradient contradicts goal",
        ),
    ],
)
def test_validate_rejects_malformed_profiles(bad):
    with pytest.raises(ValueError):
        validate_profiles(bad)


def test_the_error_names_every_problem_at_once():
    """One bad hand-edit should report all its issues, not one per fix-and-retry."""
    with pytest.raises(ValueError) as exc:
        validate_profiles([card(source="curated", spec={"goal": "focus", "mode": "am_music"})])
    message = str(exc.value)
    assert "source" in message and "spec.mode" in message and "gradient hue" in message


# ── goal is legible from the card's colour ──────────────────────────────────────


def test_warm_reads_as_focus_and_cool_as_relax():
    """The grid leans on colour to say what a card does before anyone reads it."""
    red = "linear-gradient(135deg, #c2453f, #3a1518)"
    blue = "linear-gradient(135deg, #2f8fb0, #123742)"
    assert gradient_matches_goal(red, "focus")
    assert gradient_matches_goal(blue, "relax")
    assert not gradient_matches_goal(red, "relax")
    assert not gradient_matches_goal(blue, "focus")


@pytest.mark.parametrize(
    "gradient, what",
    [
        ("linear-gradient(135deg, #2e7d4f, #16402e)", "green"),
        ("linear-gradient(135deg, #6d4fa8, #241a3c)", "purple"),
        ("linear-gradient(135deg, #8a6a3a, #2e2317)", "gold"),
    ],
)
def test_a_colour_that_is_neither_warm_nor_cool_fails_relax(gradient, what):
    """The bands are narrow on purpose: a green card isn't a taste difference from a blue
    one, it just stops meaning anything."""
    assert not gradient_matches_goal(gradient, "relax"), what


def test_every_stop_is_checked_not_just_the_first():
    """A card that starts blue and ends green is invisible until someone opens the app."""
    assert not gradient_matches_goal("linear-gradient(135deg, #2f8fb0, #16402e)", "relax")


def test_near_black_and_near_grey_stops_are_exempt():
    """The dark end of a gradient is a shadow, not a statement — holding it to the band
    would force fake precision on a colour nobody can name."""
    assert gradient_matches_goal("linear-gradient(135deg, #2f8fb0, #000000)", "relax")
    assert gradient_matches_goal("linear-gradient(135deg, #c2453f, #12110f)", "focus")


def test_shorthand_hex_is_understood():
    assert gradient_matches_goal("linear-gradient(135deg, #39b, #123742)", "relax")


def test_a_goal_with_no_band_declared_is_not_policed():
    """The check is opt-in per goal, so adding a goal to the taxonomy doesn't
    retroactively invalidate every card that uses it."""
    assert "sleep" not in GOAL_HUES
    assert gradient_matches_goal("linear-gradient(135deg, #2e7d4f, #16402e)", "sleep")
