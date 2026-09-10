from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from bnb import master_sources


def _write_wav(path, seconds=0.1, sample_rate=8000, value=1.0, channels=2):
    """A tiny synthetic "movement" file. Always a real WAV underneath (``format="WAV"``)
    even when ``path`` has a non-.wav suffix (goldberg's fake movement names end in
    .mp3) — soundfile otherwise infers a lossy MP3 encoder from the extension, and a
    constant DC signal put through that comes back with audible ringing, not a
    constant, which would make this stand-in useless for an exact-equality check."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sample_rate)
    data = np.full((n, channels), value, dtype=np.float32)
    sf.write(str(path), data, sample_rate, format="WAV")
    return path


# --- manual_source_dir / staged_sources / _resolve_source ----------------------


def test_manual_source_dir_nests_by_group_then_keyword(tmp_path, monkeypatch):
    monkeypatch.setattr(master_sources, "MANUAL_SOURCES_DIR", tmp_path)
    assert master_sources.manual_source_dir("master", "gymnopedies") == tmp_path / "master" / "gymnopedies"


def test_staged_sources_returns_empty_for_a_cell_with_nothing_staged(tmp_path, monkeypatch):
    monkeypatch.setattr(master_sources, "MANUAL_SOURCES_DIR", tmp_path)
    assert master_sources.staged_sources("master", "gymnopedies") == []


def test_staged_sources_accepts_any_filename_but_only_audio_extensions(tmp_path, monkeypatch):
    monkeypatch.setattr(master_sources, "MANUAL_SOURCES_DIR", tmp_path)
    cell_dir = tmp_path / "master" / "gymnopedies"
    _write_wav(cell_dir / "z_recording.wav")
    _write_wav(cell_dir / "a_recording.wav")
    # A stray non-audio file in the same folder must not be picked up as a candidate
    # recording — it would crash trying to decode it.
    (cell_dir / "READ ME.txt").write_text("not audio, but still a file")

    found = master_sources.staged_sources("master", "gymnopedies")
    assert [p.name for p in found] == ["a_recording.wav", "z_recording.wav"]


def test_staged_sources_mixes_files_and_subfolders_in_sorted_order(tmp_path, monkeypatch):
    monkeypatch.setattr(master_sources, "MANUAL_SOURCES_DIR", tmp_path)
    cell_dir = tmp_path / "master" / "gymnopedies"
    _write_wav(cell_dir / "a_single_file.wav")
    (cell_dir / "b_movement_set").mkdir(parents=True)
    _write_wav(cell_dir / "b_movement_set" / "1.wav")

    found = master_sources.staged_sources("master", "gymnopedies")
    assert [p.name for p in found] == ["a_single_file.wav", "b_movement_set"]


def test_resolve_source_returns_a_file_as_itself(tmp_path):
    f = _write_wav(tmp_path / "recording.wav")
    assert master_sources._resolve_source(f) == [f]


def test_resolve_source_returns_a_subfolders_audio_files_sorted(tmp_path):
    folder = tmp_path / "movement_set"
    _write_wav(folder / "2.wav")
    _write_wav(folder / "1.wav")
    (folder / "notes.txt").write_text("not audio")
    assert [p.name for p in master_sources._resolve_source(folder)] == ["1.wav", "2.wav"]


# --- _concat -------------------------------------------------------------------


def test_concat_joins_audio_in_order(tmp_path):
    a = _write_wav(tmp_path / "a.wav", value=0.1)
    b = _write_wav(tmp_path / "b.wav", value=0.2)
    out = master_sources._concat([a, b], tmp_path / "out.wav")
    data, sr = sf.read(str(out), dtype="float32", always_2d=True)
    a_data, _ = sf.read(str(a), dtype="float32", always_2d=True)
    b_data, _ = sf.read(str(b), dtype="float32", always_2d=True)
    assert len(data) == len(a_data) + len(b_data)
    assert np.allclose(data[: len(a_data)], a_data, atol=1e-4)
    assert np.allclose(data[len(a_data):], b_data, atol=1e-4)


def test_concat_rejects_mismatched_sample_rate(tmp_path):
    a = _write_wav(tmp_path / "a.wav", sample_rate=8000)
    b = _write_wav(tmp_path / "b.wav", sample_rate=16000)
    with pytest.raises(ValueError, match="expected"):
        master_sources._concat([a, b], tmp_path / "out.wav")


# --- fetch_gymnopedies (manual staging) -----------------------------------------


SPEC = {
    "track_id": "master_gymnopedies_seed1",
    "group": "master",
    "keyword": "gymnopedies",
    "download": {"source_url": "https://musopen.org/music/8010-3-gymnopedies/"},
}


def test_fetch_gymnopedies_missing_source_raises_with_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr(master_sources, "MANUAL_SOURCES_DIR", tmp_path / "manual_sources")
    with pytest.raises(RuntimeError, match="musopen.org"):
        master_sources.fetch_gymnopedies(SPEC, tmp_path / "scratch")


def test_fetch_gymnopedies_single_staged_file_is_copied_not_moved(tmp_path, monkeypatch):
    manual = tmp_path / "manual_sources"
    monkeypatch.setattr(master_sources, "MANUAL_SOURCES_DIR", manual)
    # Any filename, dropped straight in the cell directory — no need to know the seed.
    staged = _write_wav(manual / "master" / "gymnopedies" / "complete.wav", value=0.5)

    out = master_sources.fetch_gymnopedies(SPEC, tmp_path / "scratch")

    assert out != staged
    assert staged.exists()  # the staged original is left in place
    data, _ = sf.read(str(out), dtype="float32", always_2d=True)
    assert np.allclose(data, 0.5, atol=1e-4)


def test_fetch_gymnopedies_indexes_loose_files_as_separate_variants(tmp_path, monkeypatch):
    # Three loose files staged side by side (e.g. three different recordings) are
    # three *variants* — the actual bug report this rewrite fixes. Each spec resolves
    # to exactly its own file, none of them concatenated with the others.
    manual = tmp_path / "manual_sources"
    monkeypatch.setattr(master_sources, "MANUAL_SOURCES_DIR", manual)
    cell_dir = manual / "master" / "gymnopedies"
    _write_wav(cell_dir / "a_recording.wav", value=0.1)
    _write_wav(cell_dir / "b_recording.wav", value=0.2)
    _write_wav(cell_dir / "c_recording.wav", value=0.3)

    for variant, expected in enumerate((0.1, 0.2, 0.3)):
        spec = {**SPEC, "track_id": f"master_gymnopedies_seed{variant}", "variant": variant}
        out = master_sources.fetch_gymnopedies(spec, tmp_path / "scratch")
        data, _ = sf.read(str(out), dtype="float32", always_2d=True)
        assert np.allclose(data, expected, atol=1e-4), f"variant {variant}"


def test_fetch_gymnopedies_concatenates_a_movement_subfolder(tmp_path, monkeypatch):
    # A recording that legitimately arrives as several movement files goes in its own
    # subfolder, one level under the cell dir — that's the one case still concatenated.
    manual = tmp_path / "manual_sources"
    monkeypatch.setattr(master_sources, "MANUAL_SOURCES_DIR", manual)
    cell_dir = manual / "master" / "gymnopedies" / "one_recording"
    _write_wav(cell_dir / "1.wav", value=0.1)
    _write_wav(cell_dir / "2.wav", value=0.2)
    _write_wav(cell_dir / "3.wav", value=0.3)

    out = master_sources.fetch_gymnopedies({**SPEC, "variant": 0}, tmp_path / "scratch")
    data, _ = sf.read(str(out), dtype="float32", always_2d=True)
    third = len(data) // 3
    assert np.allclose(data[:third], 0.1, atol=1e-4)
    assert np.allclose(data[third: 2 * third], 0.2, atol=1e-4)
    assert np.allclose(data[2 * third:], 0.3, atol=1e-4)


def test_fetch_gymnopedies_variant_beyond_staged_count_raises_with_instructions(tmp_path, monkeypatch):
    manual = tmp_path / "manual_sources"
    monkeypatch.setattr(master_sources, "MANUAL_SOURCES_DIR", manual)
    _write_wav(manual / "master" / "gymnopedies" / "a.wav")  # only variant 0 staged

    with pytest.raises(RuntimeError, match="variant 1 \\(1 staged so far\\)"):
        master_sources.fetch_gymnopedies({**SPEC, "variant": 1}, tmp_path / "scratch")


def test_fetch_gymnopedies_defaults_to_variant_zero_when_unset(tmp_path, monkeypatch):
    manual = tmp_path / "manual_sources"
    monkeypatch.setattr(master_sources, "MANUAL_SOURCES_DIR", manual)
    _write_wav(manual / "master" / "gymnopedies" / "only.wav", value=0.5)

    spec_without_variant = {k: v for k, v in SPEC.items() if k != "variant"}
    out = master_sources.fetch_gymnopedies(spec_without_variant, tmp_path / "scratch")
    data, _ = sf.read(str(out), dtype="float32", always_2d=True)
    assert np.allclose(data, 0.5, atol=1e-4)


# --- fetch_goldberg (archive.org) ------------------------------------


class _FakeResponse:
    def __init__(self, json_data=None, content=b""):
        self._json = json_data
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return self._json

    def iter_bytes(self):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


GOLDBERG_SPEC = {
    "track_id": "master_goldberg_seed1",
    "group": "master",
    "keyword": "goldberg",
    "download": {"source_url": "https://archive.org/download/The_Open_Goldberg_Variations-11823/"},
}


def test_fetch_goldberg_falls_back_to_archive_org_when_nothing_is_staged(tmp_path, monkeypatch):
    # goldberg is manual=True (prefers staged recordings) but, unlike gymnopedies,
    # still falls back to the complete archive.org performance for variant 0 when
    # nothing has been staged yet.
    monkeypatch.setattr(master_sources, "MANUAL_SOURCES_DIR", tmp_path / "manual_sources")
    monkeypatch.setattr(master_sources, "DOWNLOAD_CACHE_DIR", tmp_path / "download_cache")

    movement_files = [
        {"name": "02 - Var 1.mp3", "format": "VBR MP3"},
        {"name": "01 - Aria.mp3", "format": "VBR MP3"},
        {"name": "notes.txt", "format": "Text"},  # not audio, must be filtered out
    ]

    def fake_get(url, timeout=30.0):
        assert "archive.org/metadata" in url
        return _FakeResponse(json_data={"files": movement_files})

    written_wavs: dict[str, bytes] = {}

    def fake_stream(method, url, timeout=120.0, follow_redirects=True):
        assert method == "GET"
        # Return distinct tiny "mp3" bytes per movement; _download doesn't parse them,
        # it just writes bytes to disk, so the real decode happens in a second step below.
        return _FakeResponse(content=url.encode())

    monkeypatch.setattr(master_sources.httpx, "get", fake_get)
    monkeypatch.setattr(master_sources.httpx, "stream", fake_stream)

    # _download writes raw bytes (URLs, here) as "mp3s", which soundfile can't read —
    # so also stub _download to write real tiny wavs keyed by movement name, proving
    # fetch_goldberg calls it once per movement, in the right order, and
    # caches the result (a second fetch must not re-download).
    calls = []

    def fake_download(url, dest, *, timeout=120.0):
        calls.append(url)
        value = 0.1 if "Aria" in url else 0.2
        _write_wav(dest, value=value)

    monkeypatch.setattr(master_sources, "_download", fake_download)

    out = master_sources.fetch_goldberg(GOLDBERG_SPEC, tmp_path / "scratch")
    data, _ = sf.read(str(out), dtype="float32", always_2d=True)
    half = len(data) // 2
    # Alphabetical filename order: "01 - Aria.mp3" before "02 - Var 1.mp3".
    assert np.allclose(data[:half], 0.1, atol=1e-4)
    assert np.allclose(data[half:], 0.2, atol=1e-4)
    assert len(calls) == 2

    # A second fetch reuses the cached movements instead of downloading again.
    master_sources.fetch_goldberg(GOLDBERG_SPEC, tmp_path / "scratch2")
    assert len(calls) == 2


def test_fetch_goldberg_prefers_a_staged_recording_over_archive_org(tmp_path, monkeypatch):
    # "manual=True, prefer manual": once anything is staged, variant 0 resolves to it
    # instead of downloading the archive.org performance — even though archive.org
    # could also answer for variant 0.
    monkeypatch.setattr(master_sources, "MANUAL_SOURCES_DIR", tmp_path / "manual_sources")
    _write_wav(tmp_path / "manual_sources" / "master" / "goldberg" / "variation-1-remix.wav", value=0.5)

    def fail_if_called(*a, **k):
        raise AssertionError("archive.org should not be reached when a recording is staged")

    monkeypatch.setattr(master_sources.httpx, "get", fail_if_called)
    monkeypatch.setattr(master_sources.httpx, "stream", fail_if_called)

    out = master_sources.fetch_goldberg({**GOLDBERG_SPEC, "variant": 0}, tmp_path / "scratch")
    data, _ = sf.read(str(out), dtype="float32", always_2d=True)
    assert np.allclose(data, 0.5, atol=1e-4)


def test_fetch_goldberg_has_no_archive_org_fallback_past_variant_zero(tmp_path, monkeypatch):
    # There's only the one archive.org item — it can answer for variant 0 when
    # nothing's staged, but a further variant with nothing staged has no fallback and
    # must fail clearly rather than silently re-downloading the same performance.
    monkeypatch.setattr(master_sources, "MANUAL_SOURCES_DIR", tmp_path / "manual_sources")
    with pytest.raises(RuntimeError, match="variant 1"):
        master_sources.fetch_goldberg({**GOLDBERG_SPEC, "variant": 1}, tmp_path / "scratch")


# --- fetch dispatch --------------------------------------------------------------


def test_fetch_dispatches_on_keyword(tmp_path, monkeypatch):
    called = {}
    monkeypatch.setitem(master_sources.DOWNLOADERS, "gymnopedies", lambda spec, d: called.setdefault("ok", True) or d)
    master_sources.fetch(SPEC, tmp_path)
    assert called == {"ok": True}


def test_fetch_rejects_unknown_keyword(tmp_path):
    with pytest.raises(RuntimeError, match="no downloader"):
        master_sources.fetch({"keyword": "bogus"}, tmp_path)
