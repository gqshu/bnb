import json

import pytest
from fastapi.testclient import TestClient

from bnb.background import GOALS, SPECIAL_GROUPS, SUBSTRATES
from bnb.profiles import MAX_BADGES, list_profiles, load_profiles, profiles_path, validate_profiles
from bnb.server import app

client = TestClient(app)


def test_shipped_profiles_are_valid():
    validate_profiles(load_profiles())  # the real assets/profiles.json cross-checks the taxonomy


def test_endpoint_returns_the_profile_catalog():
    res = client.get("/api/profiles")
    assert res.status_code == 200
    ids = [p["id"] for p in res.json()["profiles"]]
    assert ids == [p["id"] for p in load_profiles()]  # served list == file on disk
    assert ids[0] == "manual"  # manual leads the grid


def test_every_profile_is_well_formed():
    known_types = set(SUBSTRATES) | set(SPECIAL_GROUPS)
    for p in load_profiles():
        assert p["id"] and p["title"]
        assert len(p.get("badges") or []) <= MAX_BADGES
        assert p.get("image") or p.get("gradient")  # a card must have a background
        spec = p.get("spec")
        if spec is None:
            assert p.get("manual"), f"{p['id']} has no spec but isn't the manual card"
        else:
            assert spec["goal"] in GOALS
            if spec.get("type") is not None:
                assert spec["type"] in known_types


def test_profiles_are_reread_each_call(tmp_path):
    """Editing the JSON takes effect without a restart: two reads of a file that changed
    in between return the two different contents."""
    p = tmp_path / "profiles.json"
    p.write_text(json.dumps({"profiles": [{"id": "a", "title": "A", "manual": True, "gradient": "x"}]}))
    assert [x["id"] for x in load_profiles(tmp_path)] == ["a"]
    p.write_text(json.dumps({"profiles": [{"id": "b", "title": "B", "manual": True, "gradient": "y"}]}))
    assert [x["id"] for x in load_profiles(tmp_path)] == ["b"]


def test_load_accepts_a_bare_list(tmp_path):
    (tmp_path / "profiles.json").write_text(json.dumps([{"id": "a", "title": "A", "manual": True, "gradient": "x"}]))
    assert [x["id"] for x in load_profiles(tmp_path)] == ["a"]


def test_endpoint_500s_on_a_broken_config(tmp_path, monkeypatch):
    """A malformed profiles.json surfaces as a 500 with the reason, not an empty grid."""
    import bnb.profiles as profiles_mod

    bad = tmp_path / "profiles.json"
    bad.write_text(json.dumps({"profiles": [{"id": "x", "title": "X", "spec": {"goal": "gamma"}}]}))
    monkeypatch.setattr(profiles_mod, "profiles_path", lambda root=None: bad)
    res = client.get("/api/profiles")
    assert res.status_code == 500
    assert "profiles config error" in res.json()["detail"]


@pytest.mark.parametrize(
    "bad",
    [
        [{"id": "x", "title": "X", "spec": {"goal": "gamma"}}],  # unknown goal
        [{"id": "x", "title": "X", "spec": {"goal": "relax", "type": "techno"}}],  # unknown type
        [{"id": "x", "title": "X"}],  # no spec, not manual
        [{"id": "x", "title": "A", "manual": True}, {"id": "x", "title": "B", "manual": True}],  # dup id
        [{"id": "x", "title": "X", "manual": True, "badges": [{"text": str(n)} for n in range(4)]}],  # >3 badges
    ],
)
def test_validate_rejects_malformed_profiles(bad):
    with pytest.raises(ValueError):
        validate_profiles(bad)
