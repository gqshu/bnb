import json
import shutil

import pytest

from bnb import assets
from bnb.background import build_keyword_signature, build_signature
from bnb.catalog import CategoryManager


@pytest.fixture
def manager(tmp_path):
    return CategoryManager(tmp_path)


def _grid_spec(**kw):
    kw.setdefault("substrate", "drone")
    kw.setdefault("style", "buddhist_meditative")
    kw.setdefault("goal", "relax")
    return build_signature(kw["substrate"], kw["style"], kw["goal"], 60).spec()


def _keyword_spec(keyword="rain"):
    return build_keyword_signature("natural_sounds", keyword, 60).spec()


# --- add_spec: cell-directory placement --------------------------------------


def test_add_spec_nests_under_style_then_substrate(manager, tmp_path):
    spec = _grid_spec()
    path = manager.add_spec(spec)
    assert path == tmp_path / "specs" / "buddhist_meditative" / "drone" / f"{spec['track_id']}.json"
    assert path.exists()


def test_add_spec_special_cell_nests_under_group_then_keyword(manager, tmp_path):
    spec = _keyword_spec()
    path = manager.add_spec(spec)
    assert path == tmp_path / "specs" / "natural_sounds" / "rain" / f"{spec['track_id']}.json"


def test_add_spec_rebuilds_catalog_by_default(manager):
    manager.add_spec(_grid_spec())
    assert manager.catalog()["count"] == 1


def test_add_spec_can_defer_rebuild(manager):
    manager.add_spec(_grid_spec(), rebuild=False)
    assert not (manager.root / "catalog.json").exists()
    manager.rebuild()
    assert manager.catalog()["count"] == 1


# --- add_render / attach_render ----------------------------------------------


def test_add_render_writes_pcm_wav_into_the_same_cell(manager, tmp_path):
    spec = _grid_spec()
    manager.add_spec(spec, rebuild=False)
    path = manager.add_render(
        spec,
        b"\x00\x00" * 4410,
        output_format="pcm_44100",
        provider="elevenlabs",
        model_version="music_v1",
        license="x",
        generated_at="now",
    )
    assert path == tmp_path / "tracks" / "buddhist_meditative" / "drone" / f"{spec['track_id']}.wav"
    assert path.exists()
    entry = manager.search(rendered=True)[0]
    assert entry["track_id"] == spec["track_id"]
    assert entry["provider"] == "elevenlabs"


def test_attach_render_moves_a_file_already_on_disk(manager, tmp_path):
    spec = _grid_spec()
    manager.add_spec(spec, rebuild=False)
    source = tmp_path / "scratch.wav"
    source.write_bytes(b"fake-wav-bytes")

    path = manager.attach_render(
        spec,
        source,
        provider="stable_audio",
        model_version="small-music:torch",
        license="y",
        generated_at="now",
    )

    assert not source.exists()  # moved, not copied
    assert path.exists()
    assert path.parent == tmp_path / "tracks" / "buddhist_meditative" / "drone"
    assert manager.search(provider="stable_audio")[0]["track_id"] == spec["track_id"]


def test_attach_render_records_an_explicit_output_format(manager, tmp_path):
    # A bare ".wav" suffix can't reconstruct ElevenLabs' exact requested format
    # (sample rate is encoded in the string) — the caller can override it.
    spec = _grid_spec()
    manager.add_spec(spec, rebuild=False)
    source = tmp_path / "scratch.wav"
    source.write_bytes(b"fake-wav-bytes")

    manager.attach_render(
        spec, source, provider="elevenlabs", model_version="music_v1", license="x",
        generated_at="now", output_format="pcm_44100",
    )

    assert spec["render"]["output_format"] == "pcm_44100"


# --- delete / delete_cell ----------------------------------------------------


def test_delete_removes_spec_and_audio(manager):
    spec = _grid_spec()
    manager.add_spec(spec, rebuild=False)
    manager.add_render(spec, b"\x00\x00", output_format="pcm_44100", provider="elevenlabs",
                        model_version="v1", license="x", generated_at="now")

    assert manager.delete(spec["track_id"]) is True
    assert assets.find_spec(spec["track_id"], root=manager.root) is None
    assert assets.find_track(spec["track_id"], root=manager.root) is None
    assert manager.search() == []


def test_delete_unknown_track_returns_false(manager):
    assert manager.delete("does_not_exist") is False


def test_delete_cell_removes_every_track_in_it(manager):
    a = build_signature("drone", "lofi", "relax", 60).spec()
    other = build_signature("melodic_instrument", "lofi", "relax", 60).spec()
    for spec in (a, other):
        manager.add_spec(spec, rebuild=False)
    manager.rebuild()

    removed = manager.delete_cell(("lofi", "drone"))
    assert removed == 1
    remaining = {e["track_id"] for e in manager.search()}
    assert remaining == {other["track_id"]}
    assert not (manager.root / "specs" / "lofi" / "drone").exists()


# --- tags ---------------------------------------------------------------------


def test_add_tag_writes_the_spec_and_shows_up_in_the_catalog(manager):
    spec = _grid_spec()
    manager.add_spec(spec)

    assert manager.add_tag([spec["track_id"]], "warm-bed") == {spec["track_id"]: ["warm-bed"]}
    assert manager.search()[0]["tags"] == ["warm-bed"]
    # On the spec, not just the catalog — so it survives a rebuild from disk.
    on_disk = json.loads(assets.find_spec(spec["track_id"], root=manager.root).read_text())
    assert on_disk["tags"] == ["warm-bed"]
    manager.rebuild()
    assert manager.search()[0]["tags"] == ["warm-bed"]


def test_add_tag_is_idempotent_and_accumulates(manager):
    spec = _grid_spec()
    manager.add_spec(spec)
    manager.add_tag([spec["track_id"]], "warm-bed")
    manager.add_tag([spec["track_id"]], "warm-bed")
    assert manager.add_tag([spec["track_id"]], "aug-pilot") == {spec["track_id"]: ["aug-pilot", "warm-bed"]}


def test_add_tag_spans_a_batch_and_remove_tag_undoes_it(manager):
    a, b = _grid_spec(), _keyword_spec()
    manager.add_spec(a, rebuild=False)
    manager.add_spec(b)
    ids = [a["track_id"], b["track_id"]]

    manager.add_tag(ids, "pilot")
    assert sorted(e["track_id"] for e in manager.search(tag="pilot")) == sorted(ids)
    assert manager.tags() == ["pilot"]

    manager.remove_tag([a["track_id"]], "pilot")
    assert [e["track_id"] for e in manager.search(tag="pilot")] == [b["track_id"]]
    # Removing a tag a track doesn't carry is a no-op, not an error.
    manager.remove_tag([a["track_id"]], "pilot")
    assert manager.search(tag="pilot", cell=("buddhist_meditative", "drone")) == []


def test_add_tag_on_an_unknown_track_writes_nothing(manager):
    spec = _grid_spec()
    manager.add_spec(spec)
    with pytest.raises(FileNotFoundError):
        manager.add_tag([spec["track_id"], "nope"], "pilot")
    # The known track in the batch must be untouched — validate-all-then-write.
    assert manager.search()[0]["tags"] == []


def test_untagged_tracks_carry_an_empty_tag_list(manager):
    manager.add_spec(_grid_spec())
    assert manager.search()[0]["tags"] == []
    assert manager.tags() == []


# --- search -------------------------------------------------------------------


def test_search_filters_by_style_substrate_and_kind(manager):
    grid = _grid_spec()
    special = _keyword_spec()
    manager.add_spec(grid, rebuild=False)
    manager.add_spec(special, rebuild=False)
    manager.rebuild()

    assert [e["track_id"] for e in manager.search(style="buddhist_meditative")] == [grid["track_id"]]
    assert [e["track_id"] for e in manager.search(kind="special")] == [special["track_id"]]
    assert [e["track_id"] for e in manager.search(group="natural_sounds", keyword="rain")] == [special["track_id"]]
    assert manager.search(style="lofi") == []


def test_search_filters_by_goal(manager):
    relax = _grid_spec(substrate="drone", style="lofi", goal="relax")
    focus = _grid_spec(substrate="drone", style="lofi", goal="focus")
    special = _keyword_spec()
    manager.add_spec(relax, rebuild=False)
    manager.add_spec(focus, rebuild=False)
    manager.add_spec(special, rebuild=False)
    manager.rebuild()

    assert [e["track_id"] for e in manager.search(goal="focus")] == [focus["track_id"]]
    assert [e["track_id"] for e in manager.search(goal="relax")] == [relax["track_id"]]
    # Special-group entries carry no goal, so they never match a goal filter.
    assert special["track_id"] not in [e["track_id"] for e in manager.search(goal="relax")]
    assert manager.pick(goal="focus")["track_id"] == focus["track_id"]


def test_search_by_cell_covers_both_grid_and_special(manager):
    grid = _grid_spec()
    special = _keyword_spec()
    manager.add_spec(grid, rebuild=False)
    manager.add_spec(special, rebuild=False)
    manager.rebuild()

    assert [e["track_id"] for e in manager.search(cell=("buddhist_meditative", "drone"))] == [grid["track_id"]]
    assert [e["track_id"] for e in manager.search(cell=("natural_sounds", "rain"))] == [special["track_id"]]


def test_cells_lists_every_distinct_cell(manager):
    manager.add_spec(_grid_spec(), rebuild=False)
    manager.add_spec(_keyword_spec(), rebuild=False)
    manager.rebuild()
    assert manager.cells() == [("buddhist_meditative", "drone"), ("natural_sounds", "rain")]


# --- pick -----------------------------------------------------------------------


def test_pick_returns_none_when_nothing_matches(manager):
    assert manager.pick(rendered=True) is None


def test_pick_excludes_when_theres_an_alternative(manager):
    a = build_signature("drone", "lofi", "relax", 60).spec()
    b = build_signature("melodic_instrument", "lofi", "relax", 60).spec()
    for spec in (a, b):
        manager.add_spec(spec, rebuild=False)
        manager.add_render(spec, b"\x00\x00", output_format="pcm_44100", provider="elevenlabs",
                            model_version="v1", license="x", generated_at="now", rebuild=False)
    manager.rebuild()

    picks = {manager.pick(rendered=True, exclude=a["track_id"])["track_id"] for _ in range(20)}
    assert picks == {b["track_id"]}


def test_pick_falls_back_to_full_pool_if_exclude_would_empty_it(manager):
    spec = _grid_spec()
    manager.add_spec(spec, rebuild=False)
    manager.add_render(spec, b"\x00\x00", output_format="pcm_44100", provider="elevenlabs",
                        model_version="v1", license="x", generated_at="now")

    entry = manager.pick(rendered=True, exclude=spec["track_id"])
    assert entry["track_id"] == spec["track_id"]  # only option, so exclude is overridden


def test_pick_unknown_strategy_raises(manager):
    manager.add_spec(_grid_spec(), rebuild=False)
    manager.add_render(_grid_spec(), b"\x00\x00", output_format="pcm_44100", provider="elevenlabs",
                        model_version="v1", license="x", generated_at="now")
    with pytest.raises(ValueError, match="unknown pick strategy"):
        manager.pick(rendered=True, strategy="least_recently_played")


# --- the catalog is a function of the spec tree --------------------------------


def test_rebuild_drops_a_spec_deleted_by_hand(manager):
    spec = _grid_spec()
    manager.add_spec(spec)
    assets.find_spec(spec["track_id"], root=manager.root).unlink()
    assert manager.rebuild()["count"] == 0


def test_rebuild_drops_a_cell_directory_deleted_by_hand(manager, tmp_path):
    manager.add_spec(_grid_spec(), rebuild=False)
    manager.add_spec(_keyword_spec(), rebuild=True)
    shutil.rmtree(tmp_path / "specs" / "natural_sounds")
    assert [e["kind"] for e in manager.rebuild()["tracks"]] == ["grid"]


def test_rebuild_relocates_a_spec_that_isnt_in_its_cell(manager, tmp_path):
    # The old flat layout, and any hand-moved spec: the cell is derived from the
    # record, so the file follows it rather than the other way round.
    spec = _grid_spec()
    stray = tmp_path / "specs" / f"{spec['track_id']}.json"
    stray.parent.mkdir(parents=True)
    stray.write_text(json.dumps(spec))

    manager.rebuild()

    assert not stray.exists()
    assert assets.find_spec(spec["track_id"], root=tmp_path) == assets.spec_path(spec, tmp_path)


def test_rebuild_renames_a_spec_whose_filename_isnt_its_track_id(manager, tmp_path):
    spec = _grid_spec()
    path = assets.spec_path(spec, tmp_path)
    path.parent.mkdir(parents=True)
    (path.parent / "copy-of-a-spec.json").write_text(json.dumps(spec))

    manager.rebuild()

    assert [p.name for p in (path.parent).iterdir()] == [f"{spec['track_id']}.json"]


def test_rebuild_prunes_the_empty_directories_a_delete_leaves_behind(manager, tmp_path):
    spec = _grid_spec()
    manager.add_spec(spec)
    assets.find_spec(spec["track_id"], root=manager.root).unlink()
    manager.rebuild()
    assert not (tmp_path / "specs" / "buddhist_meditative").exists()


def test_duplicate_track_ids_raise_instead_of_one_silently_winning(manager, tmp_path):
    spec = _grid_spec()
    manager.add_spec(spec)
    duplicate = tmp_path / "specs" / "elsewhere" / f"{spec['track_id']}.json"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_text(json.dumps(spec))

    with pytest.raises(ValueError, match="duplicate track_id"):
        manager.rebuild()


def test_an_unreadable_spec_raises_rather_than_being_skipped(manager, tmp_path):
    (tmp_path / "specs").mkdir(parents=True)
    (tmp_path / "specs" / "junk.json").write_text("{not json")
    with pytest.raises(ValueError, match="not a readable spec"):
        manager.rebuild()


def test_spec_ids_reads_disk_not_the_stale_catalog(manager):
    spec = _grid_spec()
    manager.add_spec(spec)
    assets.find_spec(spec["track_id"], root=manager.root).unlink()

    assert manager.spec_ids() == set()  # catalog.json still lists it; disk doesn't
    assert [e["track_id"] for e in manager.catalog()["tracks"]] == [spec["track_id"]]


def test_orphan_tracks_finds_audio_whose_spec_is_gone(manager):
    spec = _grid_spec()
    manager.add_spec(spec, rebuild=False)
    audio = manager.add_render(spec, b"\x00\x00", output_format="pcm_44100", provider="elevenlabs",
                                model_version="v1", license="x", generated_at="now")
    assert manager.orphan_tracks() == []

    assets.find_spec(spec["track_id"], root=manager.root).unlink()
    assert manager.orphan_tracks() == [audio]


# --- isolation from the real asset repo --------------------------------------


def test_manager_defaults_to_the_real_assets_dir():
    assert CategoryManager().root == assets.ASSETS_DIR


def test_manager_with_explicit_root_never_touches_the_real_assets_dir(manager, tmp_path):
    # Compared as a file listing rather than "this cell doesn't exist": the real repo
    # legitimately fills up with the same cells these tests use.
    before = set(assets.ASSETS_DIR.rglob("*"))

    manager.add_spec(_grid_spec(), rebuild=False)
    manager.add_render(_grid_spec(), b"\x00\x00", output_format="pcm_44100", provider="elevenlabs",
                        model_version="v1", license="x", generated_at="now")

    assert manager.root == tmp_path
    assert set(assets.ASSETS_DIR.rglob("*")) == before
