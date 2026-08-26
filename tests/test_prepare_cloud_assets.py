"""The cloud build: what lands in the bucket, and what deliberately doesn't."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import prepare_cloud_assets as prep  # noqa: E402  (path hack must precede this import)

from bnb.profiles import validate_profiles  # noqa: E402

RELAX_GRADIENT = "linear-gradient(135deg, #4a56b8, #1d2050)"


def card(pid, goal, soundscape=None, source="community"):
    spec = {"goal": goal, "mode": "binaural"}
    if soundscape:
        spec["soundscape"] = list(soundscape)
    return {"id": pid, "title": pid, "source": source, "gradient": RELAX_GRADIENT, "spec": spec}


def manifest(pairs):
    return {
        "tracks": [
            {"id": f"t{i}", "file": f"bg/t{i}.mp3", "name": "x", "tags": list(tags), "goals": [g]}
            for i, (tags, g) in enumerate(pairs)
        ]
    }


LIBRARY = manifest(
    [
        (["natural_sounds", "rain"], "relax"),
        (["drone", "lofi"], "focus"),
        (["energizer", "uplift"], "relax"),
    ]
)


def test_the_bucket_carries_community_music_only(tmp_path, monkeypatch):
    """The mini program ships the manual panel and the EEG preset built in; publishing
    them again would just duplicate them in the grid."""
    authored = [card("shared", "relax"), card("mine", "relax", source="personal")]
    monkeypatch.setattr(prep, "list_profiles", lambda: authored)
    profiles, _ = prep.build_profiles(tmp_path)
    assert [p["id"] for p in profiles] == ["shared"]


def test_the_shipped_catalogue_survives_the_round_trip(tmp_path):
    """The real assets/profiles.json, through the real validation, into the real output
    shape — the thing that actually gets uploaded."""
    profiles, missing_art = prep.build_profiles(tmp_path)
    assert profiles and not missing_art
    validate_profiles(profiles)
    assert all(p["source"] == "community" for p in profiles)


def test_the_shipped_catalogue_is_playable_against_the_real_library(tmp_path):
    """Every card in the bucket must name a (soundscape, goal) some track can answer, or
    the grid tile throws the moment someone taps it."""
    from bnb.catalog import CategoryManager

    entries = sorted(CategoryManager().search(rendered=True), key=lambda e: e["track_id"])
    profiles, _ = prep.build_profiles(tmp_path)
    assert prep.unplayable_profiles(profiles, prep.build_manifest(entries, 96)) == []


def test_a_card_no_track_can_answer_is_reported(tmp_path):
    """Selection is the client's job now, and its whole filter is goals + soundscape — so
    a card asking for a combination the manifest doesn't hold throws at play time."""
    dead = prep.unplayable_profiles(
        [
            card("fine", "relax", ["natural_sounds"]),
            card("by_keyword", "relax", ["rain"]),  # the group's keyword, not the group
            card("by_pair", "focus", ["drone.lofi"]),  # both tags of one track
            card("one_of_two", "relax", ["techno", "energizer"]),  # a union: one hit is enough
            card("wrong_sound", "relax", ["drone"]),  # drone exists, but only for focus
            card("wrong_goal", "focus", ["natural_sounds"]),
            card("wrong_pair", "focus", ["drone.neutral"]),  # right track, wrong style
            card("any_relax", "relax"),  # no soundscape: any relax track will do
        ],
        LIBRARY,
    )
    assert [d.split(":")[0] for d in dead] == ["wrong_sound", "wrong_goal", "wrong_pair"]
    assert "goal=relax soundscape=['drone']" in dead[0]


def test_a_card_with_no_spec_is_not_called_unplayable():
    """The manual card expands to the control panel, not to a track."""
    assert prep.unplayable_profiles([{"id": "manual", "manual": True}], LIBRARY) == []


def test_playable_tracks_reads_every_goal_a_track_serves():
    """A special-group bed suits more than one goal — rainfall relaxes and masks — so one
    track can vouch for cards on both sides."""
    both = manifest([(["natural_sounds", "rain"], "relax")])
    both["tracks"][0]["goals"] = ["focus", "relax"]
    assert prep.playable_tracks(both) == [(["natural_sounds", "rain"], {"focus", "relax"})]
    assert prep.unplayable_profiles([card("f", "focus", ["rain"])], both) == []


def test_strays_are_the_mp3s_the_manifest_stopped_listing(tmp_path):
    """The output directory is what gets dragged into the bucket, so a track dropped
    from the taxonomy leaves its audio behind for every later upload to carry along."""
    audio = tmp_path / prep.AUDIO_SUBDIR
    audio.mkdir(parents=True)
    for name in ("keep.mp3", "dropped.mp3"):
        (audio / name).write_bytes(b"")
    assert prep.stray_mp3s(tmp_path, {"keep"}) == [audio / "dropped.mp3"]
    assert prep.stray_mp3s(tmp_path, {"keep", "dropped"}) == []
