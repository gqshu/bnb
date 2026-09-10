"""The cloud build: what lands in the bucket, and what deliberately doesn't."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np  # noqa: E402  (path hack must precede this import)
import pytest  # noqa: E402
import soundfile as sf  # noqa: E402

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
    manifest = prep.build_manifest(entries, 96, prep.DEFAULT_CHUNK_MINUTES * 60)
    assert prep.unplayable_profiles(profiles, manifest) == []


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


# --- long-track chunking (the master group's real recordings) ------------------


def test_chunk_count_ships_short_or_untimed_tracks_whole():
    assert prep.chunk_count(None, 240) == 1  # unmeasured duration: nothing to split on
    assert prep.chunk_count(100, 240) == 1  # under one chunk
    assert prep.chunk_count(240, 240) == 1  # exactly one chunk


def test_chunk_count_splits_a_long_track_evenly():
    # A full Goldberg Variations recording, at the 4-minute default: not an exact
    # multiple, so this is also checking chunk_count rounds up rather than truncating
    # (a truncated last chunk would silently drop the tail of the recording).
    assert prep.chunk_count(80 * 60, 240) == 20
    assert prep.chunk_count(79 * 60, 240) == 20


def test_chunk_count_respects_the_min_chunk_size_floor():
    # At 96 kbps, 240s (the default --chunk-minutes) chunks are ~2.9 MB each — nowhere
    # near the 1 MB default floor, so a duration that would naturally split several ways
    # still does, unconstrained.
    assert (
        prep.chunk_count(80 * 60, 240, bitrate_kbps=96, min_chunk_bytes=1 * 1024 * 1024) == 20
    )


def test_chunk_count_shrinks_part_count_rather_than_produce_tiny_files():
    # A small --chunk-minutes (60s) at 96 kbps would naturally want 10 one-minute
    # parts (~0.7 MB each) — under a 1 MB floor. The count must shrink until each
    # part's average size clears the floor, not silently ship undersized files.
    n = prep.chunk_count(600, 60, bitrate_kbps=96, min_chunk_bytes=1 * 1024 * 1024)
    assert n < 10
    bytes_per_s = 96 * 1000 / 8
    assert (600 / n) * bytes_per_s >= 1 * 1024 * 1024


def test_chunk_count_floor_never_goes_below_one_part():
    # An extreme floor (larger than the whole track) must still ship the track — as
    # a single unchunked file — rather than producing zero parts.
    assert prep.chunk_count(600, 60, bitrate_kbps=96, min_chunk_bytes=1000 * 1024 * 1024) == 1


def test_chunk_ids_ships_a_single_part_under_the_bare_track_id():
    assert prep.chunk_ids("master_goldberg_seed1", 1) == ["master_goldberg_seed1"]


def test_chunk_ids_zero_pads_to_the_part_counts_width():
    assert prep.chunk_ids("t", 3) == ["t_part1", "t_part2", "t_part3"]
    assert prep.chunk_ids("t", 12)[0] == "t_part01"
    assert prep.chunk_ids("t", 12)[-1] == "t_part12"


def _entry(**overrides):
    base = {
        "track_id": "master_goldberg_seed1",
        "kind": "special",
        "group": "master",
        "keyword": "goldberg",
        "substrate": None,
        "style": None,
        "goal": None,
        "duration_s": 500,
        "loopable": False,
        "tags": [],
    }
    return {**base, **overrides}


def test_manifest_entries_for_expands_a_long_non_loopable_track_into_chunks():
    rows = prep.manifest_entries_for(_entry(), chunk_s=240)
    assert [r["id"] for r in rows] == [
        "master_goldberg_seed1_part1",
        "master_goldberg_seed1_part2",
        "master_goldberg_seed1_part3",
    ]
    assert [r["name"] for r in rows] == [
        f"{prep._bg_display_name(_entry())} · {i}/3" for i in (1, 2, 3)
    ]
    # Every chunk shares the source track's selection keys, so a goals/tags filter
    # still finds the whole recording, not just its first piece.
    assert {r["loopable"] for r in rows} == {False}
    assert all(r["tags"] == rows[0]["tags"] for r in rows)
    assert all(r["goals"] == rows[0]["goals"] for r in rows)


def test_manifest_entries_for_never_splits_a_loopable_track():
    rows = prep.manifest_entries_for(_entry(loopable=True, duration_s=10_000), chunk_s=240)
    assert len(rows) == 1
    assert rows[0]["id"] == "master_goldberg_seed1"


def test_manifest_entries_for_ships_a_short_master_track_whole():
    rows = prep.manifest_entries_for(_entry(duration_s=60), chunk_s=240)
    assert len(rows) == 1
    assert rows[0]["id"] == "master_goldberg_seed1"


def test_build_manifest_count_is_published_files_not_source_tracks():
    manifest = prep.build_manifest([_entry()], bitrate_kbps=96, chunk_s=240)
    assert manifest["count"] == 3 == len(manifest["tracks"])


# --- chunk_bounds --------------------------------------------------------------


def _tone(seconds, sample_rate=8000, amplitude=0.2, freq=220.0):
    t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def test_chunk_bounds_is_a_no_op_for_one_part():
    mono = _tone(1.0)
    assert prep.chunk_bounds(len(mono), 8000, mono, 1) == [(0, len(mono))]


def test_chunk_bounds_covers_the_signal_contiguously_with_no_gaps_or_overlaps():
    sample_rate = 8000
    mono = _tone(20.0, sample_rate)
    bounds = prep.chunk_bounds(len(mono), sample_rate, mono, 4)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == len(mono)
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))  # end of one == start of next
    assert all(start < end for start, end in bounds)  # never an empty chunk


def test_chunk_bounds_snaps_the_cut_to_a_nearby_quiet_gap():
    # A steady tone with one true silent gap placed half a second after the even
    # 2-way split point (well within the +/-2s search window). An even split would
    # land inside the tone; the cut must move to the gap instead.
    sample_rate = 8000
    tone_a = _tone(10.5, sample_rate)
    gap = np.zeros(int(0.3 * sample_rate))
    tone_b = _tone(9.2, sample_rate)
    mono = np.concatenate([tone_a, gap, tone_b])
    even_split = len(mono) // 2
    gap_start, gap_end = len(tone_a), len(tone_a) + len(gap)
    assert not (gap_start <= even_split <= gap_end)  # the split point is NOT already in the gap

    bounds = prep.chunk_bounds(len(mono), sample_rate, mono, 2)
    cut = bounds[0][1]
    assert gap_start <= cut <= gap_end


# --- transcode_track -------------------------------------------------------------


def _write_master_wav(path, seconds=6.0, sample_rate=8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    mono = _tone(seconds, sample_rate)
    data = np.stack([mono, mono], axis=1).astype(np.float32)
    sf.write(str(path), data, sample_rate)
    return path


class _Args:
    no_declick = True  # keep the synthetic tone's edges untouched
    no_loop_prep = False
    crossfade = 0.05
    click_sensitivity = 6.0
    peak_dbfs = -1.0
    bitrate = 96
    quality = 7  # fast; these are tiny throwaway files


def test_transcode_track_writes_one_file_per_chunk(tmp_path):
    src = _write_master_wav(tmp_path / "src.wav")
    audio_dir = tmp_path / "bg"
    ids = ["t_part1", "t_part2", "t_part3"]

    sizes, report = prep.transcode_track(
        src, audio_dir, ids, _Args(), loopable=False, max_chunk_bytes=10 * 1024 * 1024
    )

    assert len(sizes) == 3
    for chunk_id, size in zip(ids, sizes):
        written = audio_dir / f"{chunk_id}.mp3"
        assert written.exists()
        assert written.stat().st_size == size
        assert size > 0
    assert report.duration_s > 0


def test_transcode_track_ships_a_single_file_for_one_chunk(tmp_path):
    src = _write_master_wav(tmp_path / "src.wav")
    audio_dir = tmp_path / "bg"

    sizes, _ = prep.transcode_track(
        src, audio_dir, ["whole"], _Args(), loopable=False, max_chunk_bytes=10 * 1024 * 1024
    )

    assert len(sizes) == 1
    assert (audio_dir / "whole.mp3").exists()


def test_transcode_track_refuses_a_chunk_over_the_size_cap(tmp_path):
    src = _write_master_wav(tmp_path / "src.wav")
    with pytest.raises(ValueError, match="exceeds"):
        prep.transcode_track(
            src, tmp_path / "bg", ["t_part1", "t_part2"], _Args(), loopable=False, max_chunk_bytes=10
        )
