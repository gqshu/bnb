import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import plan_background as pb  # noqa: E402  (path hack must precede this import)

from bnb.background import SPECIAL_GROUPS, build_keyword_signature  # noqa: E402
from bnb.catalog import CategoryManager  # noqa: E402


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Run the CLI against a throwaway asset repo instead of the real assets/."""
    monkeypatch.setattr(pb, "CategoryManager", lambda: CategoryManager(tmp_path))

    def _run(*argv):
        monkeypatch.setattr(sys, "argv", ["plan_background.py", *argv])
        pb.main()
        return CategoryManager(tmp_path)

    return _run


# --- target resolution ---------------------------------------------------------


def test_bare_special_group_plans_every_keyword(run, tmp_path):
    manager = run("natural_sounds")
    keywords = {e["keyword"] for e in manager.search(group="natural_sounds")}
    assert keywords == set(SPECIAL_GROUPS["natural_sounds"].keywords)
    assert (tmp_path / "specs" / "natural_sounds" / "rain").is_dir()


def test_group_keyword_plans_just_that_cell(run):
    manager = run("natural_sounds:rain")
    assert [e["keyword"] for e in manager.search(kind="special")] == ["rain"]


def test_grid_pair_still_works_alongside_a_group(run):
    manager = run("lofi:drone", "natural_sounds:ocean")
    assert {e["kind"] for e in manager.search()} == {"grid", "special"}


def test_unknown_bare_target_names_the_special_groups(run):
    with pytest.raises(SystemExit) as excinfo:
        run("lofi")  # a style, not a group: bare targets only name special groups
    assert "expected one of: natural_sounds" in str(excinfo.value)


# --- bad names are CLI errors, never tracebacks ---------------------------------


@pytest.mark.parametrize(
    "argv, typo, suggestion",
    [
        (("natural_sounds:rian",), "rian", "rain"),
        (("buddhist_meditative:dron",), "dron", "drone"),
        (("lofy:drone",), "lofy", "lofi"),
        (("natral_sounds",), "natral_sounds", "natural_sounds"),
        (("--per-cell", "2", "--groups", "natrual_sounds"), "natrual_sounds", "natural_sounds"),
        (("--fill", "2", "--styles", "lofy"), "lofy", "lofi"),
        (("--coverage", "--substrates", "dron"), "dron", "drone"),
    ],
)
def test_a_mistyped_name_exits_with_the_closest_match(run, argv, typo, suggestion):
    with pytest.raises(SystemExit) as excinfo:
        run(*argv)
    message = str(excinfo.value)
    assert repr(typo) in message
    assert f"did you mean {suggestion!r}" in message


def test_an_unknown_name_with_no_near_match_still_lists_the_options(run):
    with pytest.raises(SystemExit, match="expected one of: rain, ocean"):
        run("natural_sounds:volcano")


def test_a_bad_axis_name_fails_before_anything_is_written(run, tmp_path):
    with pytest.raises(SystemExit):
        run("--fill", "2", "--styles", "lofy")
    assert not (tmp_path / "catalog.json").exists()  # not even the rebuild ran


# --- planning never overwrites --------------------------------------------------


def _spec_file(manager, track_id):
    from bnb import assets

    return assets.find_spec(track_id, root=manager.root)


def test_existing_specs_are_skipped_not_rewritten(run, capsys):
    track_id = build_keyword_signature("natural_sounds", "rain", 60).track_id
    manager = run("natural_sounds:rain")
    path = _spec_file(manager, track_id)
    edited = json.loads(path.read_text()) | {"prompt": "hand-edited"}
    path.write_text(json.dumps(edited))

    run("natural_sounds:rain", "--duration", "90")

    assert json.loads(path.read_text())["prompt"] == "hand-edited"
    assert "delete its spec to replan" in capsys.readouterr().out


def test_deleting_a_spec_is_what_allows_a_replan(run):
    track_id = build_keyword_signature("natural_sounds", "rain", 60).track_id
    manager = run("natural_sounds:rain")
    _spec_file(manager, track_id).unlink()

    manager = run("natural_sounds:rain", "--duration", "90")

    assert manager.search()[0]["duration_s"] == 90


def test_a_repeated_target_in_one_run_is_only_written_once(run):
    manager = run("natural_sounds", "natural_sounds:rain")
    assert manager.catalog()["count"] == len(SPECIAL_GROUPS["natural_sounds"].keywords)


# --- coverage guides -------------------------------------------------------------


def test_fill_always_adds_new_seeds_rather_than_colliding(run):
    manager = run("--fill", "3")
    first = {e["track_id"] for e in manager.search()}
    manager = run("--fill", "3")
    assert len(manager.search()) == 6
    assert first < {e["track_id"] for e in manager.search()}


def test_groups_fills_special_cells_instead_of_the_grid(run):
    manager = run("--per-cell", "2", "--groups", "natural_sounds")
    entries = manager.search()
    assert {e["kind"] for e in entries} == {"special"}
    assert len(entries) == 2 * len(SPECIAL_GROUPS["natural_sounds"].keywords)


def test_groups_is_how_a_keyword_gets_a_second_track(run):
    manager = run("natural_sounds")  # one seed per keyword
    manager = run("--per-cell", "2", "--groups", "natural_sounds")
    rain = manager.search(keyword="rain")
    assert len(rain) == 2
    assert len({e["seed"] for e in rain}) == 2


def test_groups_cannot_be_mixed_with_grid_axis_filters(run):
    with pytest.raises(SystemExit, match="can't be combined"):
        run("--fill", "2", "--groups", "natural_sounds", "--styles", "lofi")


def test_groups_without_a_coverage_guide_is_rejected(run):
    with pytest.raises(SystemExit, match="restricts a coverage guide"):
        run("natural_sounds:rain", "--groups", "natural_sounds")


def test_coverage_prints_the_grid_and_the_special_groups(run, capsys):
    run("lofi:drone", "natural_sounds:rain")
    capsys.readouterr()
    run("--coverage")
    out = capsys.readouterr().out
    assert "nature_ambient" in out  # the grid matrix
    assert "natural_sounds" in out and "chimes" in out  # the special block


def test_coverage_narrows_to_the_taxonomy_an_axis_filter_names(run, capsys):
    run("lofi:drone", "natural_sounds:rain")
    capsys.readouterr()

    run("--coverage", "--groups", "natural_sounds")
    special_only = capsys.readouterr().out
    assert "natural_sounds" in special_only and "nature_ambient" not in special_only

    run("--coverage", "--styles", "lofi")
    grid_only = capsys.readouterr().out
    assert "lofi" in grid_only and "natural_sounds" not in grid_only


def test_orphan_audio_is_reported(run, tmp_path, capsys):
    track_id = build_keyword_signature("natural_sounds", "rain", 60).track_id
    manager = run("natural_sounds:rain")
    audio = tmp_path / "tracks" / "natural_sounds" / "rain" / f"{track_id}.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"fake")
    _spec_file(manager, track_id).unlink()
    capsys.readouterr()

    run("--coverage")

    assert "have no spec" in capsys.readouterr().out
