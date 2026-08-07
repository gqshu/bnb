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
    return build_signature(kw["substrate"], kw["style"], 60).spec()


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
    a = build_signature("drone", "lofi", 60).spec()
    other = build_signature("melodic_instrument", "lofi", 60).spec()
    for spec in (a, other):
        manager.add_spec(spec, rebuild=False)
    manager.rebuild()

    removed = manager.delete_cell(("lofi", "drone"))
    assert removed == 1
    remaining = {e["track_id"] for e in manager.search()}
    assert remaining == {other["track_id"]}
    assert not (manager.root / "specs" / "lofi" / "drone").exists()


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
    a = build_signature("drone", "lofi", 60).spec()
    b = build_signature("melodic_instrument", "lofi", 60).spec()
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


# --- isolation from the real asset repo --------------------------------------


def test_manager_defaults_to_the_real_assets_dir():
    assert CategoryManager().root == assets.ASSETS_DIR


def test_manager_with_explicit_root_never_touches_the_real_assets_dir(manager, tmp_path):
    manager.add_spec(_grid_spec(), rebuild=False)
    manager.add_render(_grid_spec(), b"\x00\x00", output_format="pcm_44100", provider="elevenlabs",
                        model_version="v1", license="x", generated_at="now")
    assert manager.root == tmp_path
    assert not (assets.ASSETS_DIR / "tracks" / "buddhist_meditative").exists()
