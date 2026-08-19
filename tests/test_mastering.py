"""Loop mastering: click repair and seam preparation.

Fixtures are synthesized rather than read from ``assets/tracks/``: the audio masters
are git-ignored, so a test that needs them fails on a fresh clone.
"""

import numpy as np
import pytest

from bnb import mastering

SAMPLE_RATE = 44_100


def bed(seconds=10, amplitude=0.3, seed=0, swell=True):
    """A stand-in for a healthy ambient render: slow tonal movement, some texture.

    The partials carry a phase offset and periods that don't divide the duration, so
    the fixture does *not* wrap onto itself — a bed whose last sample happens to meet its
    first loops cleanly by accident and would hide the seam this module exists to fix.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 0.13 * t) if swell else 1.0
    tone = np.sin(2 * np.pi * 110.37 * t + 0.7) + 0.7 * np.sin(2 * np.pi * 164.53 * t + 2.1)
    mono = amplitude * (envelope * tone / 1.7 + rng.normal(0, 0.03, len(t)))
    return np.stack([mono, mono * 0.95], axis=1)


def with_fade_out(audio, seconds=2.0):
    """The library's actual shape: full level at the top, faded to nothing at the end."""
    out = audio.copy()
    n = int(seconds * SAMPLE_RATE)
    out[-n:] *= np.linspace(1.0, 0.0, n)[:, None]
    return out


def crackle(seconds=10, seed=1, rate=40):
    """Impulsive content — a stand-in for fireplace or stream.

    Each event is a sharp attack decaying away, which is what a crackle *is*: it trips
    the detector just as a defect would. How often it happens is the only thing that
    tells the two apart.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    mono = 0.02 * rng.normal(0, 1, n)
    burst = np.exp(-np.arange(80) / 8.0) * np.cos(np.linspace(0, 12, 80))
    for pos in rng.integers(200, n - 200, int(rate * seconds)):
        mono[pos : pos + 80] += 0.4 * rng.normal(0, 1) * burst
    return np.stack([mono, mono], axis=1)


# ── click detection and repair ──────────────────────────────────────────────────


def test_a_one_sample_step_is_found():
    audio = bed()
    audio[SAMPLE_RATE * 4, :] += 0.25
    hits = mastering.find_clicks(audio[:, 0], SAMPLE_RATE)
    assert any(start - 3 <= SAMPLE_RATE * 4 <= end + 3 for start, end in hits)


def test_repair_removes_the_step_and_leaves_the_music():
    clean = bed()
    broken = clean.copy()
    broken[SAMPLE_RATE * 4, :] += 0.25
    fixed, clicks = mastering.repair_clicks(broken, SAMPLE_RATE)

    assert clicks.found == clicks.repaired == 2  # once per channel
    assert not mastering.find_clicks(fixed[:, 0], SAMPLE_RATE)
    # The repair is local: everything outside the click's few samples is untouched.
    away = np.r_[0 : SAMPLE_RATE * 4 - 10, SAMPLE_RATE * 4 + 10 : len(clean)]
    assert np.allclose(fixed[away], clean[away])


def test_crackle_survives_because_of_how_often_it_happens():
    """Fireplace crackle and stream splashes trip the detector — they are impulses. What
    saves them is the rate: content made of impulses produces hundreds a minute, where a
    real defect produces a handful, so the whole track is quarantined rather than fixed."""
    audio = crackle()
    fixed, clicks = mastering.repair_clicks(audio, SAMPLE_RATE)
    assert clicks.found and clicks.impulsive
    assert clicks.repaired == 0
    assert np.array_equal(fixed, audio)


def test_a_lone_click_in_an_otherwise_calm_track_is_still_repaired():
    """The rate floor must not become a blanket amnesty: one event in a minute is exactly
    the case the pass exists for."""
    audio = crackle(rate=1)
    audio[SAMPLE_RATE * 4, :] += 0.25
    _, clicks = mastering.repair_clicks(audio, SAMPLE_RATE)
    assert not clicks.impulsive
    assert clicks.repaired


def test_a_wide_burst_is_reported_but_not_interpolated():
    """Past a millisecond the thing detected is an attack, not a click — count it and
    leave it, rather than punching a hole where the transient was."""
    audio = bed()
    start = SAMPLE_RATE * 4
    audio[start : start + 300, :] += 0.6
    fixed, clicks = mastering.repair_clicks(audio, SAMPLE_RATE, max_click_ms=1.0)
    assert clicks.found and clicks.repaired == 0
    assert np.array_equal(fixed, audio)


def test_near_silence_never_reads_as_clicks():
    """The ratio test alone would fire on dither: the local mean collapses to nothing,
    so the absolute-jump floor is what keeps a quiet passage quiet."""
    rng = np.random.default_rng(3)
    quiet = np.stack([rng.normal(0, 1e-5, SAMPLE_RATE)] * 2, axis=1)
    assert not mastering.find_clicks(quiet[:, 0], SAMPLE_RATE)


# ── loop preparation ────────────────────────────────────────────────────────────


def test_trim_drops_the_fade_not_just_the_silence():
    """A tail at -45 dBFS is inaudible next to a head at -12 and loops as a hole all the
    same, so the gate is relative to the track's own level rather than a silence floor."""
    # A steady bed, so what the gate measures is the fade and not the track's own swell.
    audio = with_fade_out(bed(swell=False), seconds=4.0)
    trimmed, head, tail = mastering.trim_edges(audio, SAMPLE_RATE)
    assert len(trimmed) == len(audio) - int((head + tail) * SAMPLE_RATE)
    # The fade is linear over 4 s, so everything past -12 dB is its last second — and a
    # silence gate would have taken only the last 25 ms of it, which is the whole reason
    # the threshold is relative rather than absolute.
    _, _, by_silence = mastering.trim_edges(audio, SAMPLE_RATE, below_db=-60)
    assert tail == pytest.approx(1.0, abs=0.1)
    assert by_silence < 0.1


def test_trim_refuses_to_eat_a_track_with_long_quiet_ends():
    """A bed whose outer thirds sit well below its middle is a track we would be
    mangling, not fixing. Past the cap the gate falls back to a plain silence floor,
    which finds nothing to cut here, and the track comes back whole."""
    audio = bed()
    envelope = np.full((len(audio), 1), 0.05)
    third = len(audio) // 3
    envelope[third : 2 * third] = 1.0
    trimmed, head, tail = mastering.trim_edges(audio * envelope, SAMPLE_RATE)
    assert (head, tail) == (0.0, 0.0)
    assert len(trimmed) == len(audio)


def test_crossfade_makes_the_wrap_sample_continuous():
    audio = bed()
    looped = mastering.loop_crossfade(audio, SAMPLE_RATE, seconds=1.0)
    width = SAMPLE_RATE
    assert len(looped) == len(audio) - width
    # The last sample of the output is the original sample just before its first, so a
    # player wrapping end -> start sees two consecutive samples of the source.
    assert np.allclose(looped[0], audio[len(audio) - width])
    assert np.allclose(looped[-1], audio[len(audio) - width - 1])


def test_crossfade_is_capped_at_a_quarter_of_the_track():
    audio = bed(seconds=2)
    looped = mastering.loop_crossfade(audio, SAMPLE_RATE, seconds=10.0)
    assert len(looped) == len(audio) - len(audio) // 4


def test_peak_limiting_only_pulls_down():
    loud = bed(amplitude=0.9) * 1.4
    limited, gain_db = mastering.limit_peak(loud, -1.0)
    assert gain_db < 0
    assert np.abs(limited).max() == pytest.approx(10 ** (-1.0 / 20), rel=1e-6)

    quiet = bed(amplitude=0.2)
    same, gain_db = mastering.limit_peak(quiet, -1.0)
    assert gain_db == 0.0
    assert np.array_equal(same, quiet)


# ── the whole pass ──────────────────────────────────────────────────────────────


def test_mastering_fixes_the_seam_the_client_would_have_heard():
    """The defect this pass exists for: a master that fades out at the end and starts at
    full level lurches once per lap, because nothing in the cloud path crossfades."""
    audio = with_fade_out(bed(), seconds=2.0)
    out, report = mastering.master_for_loop(audio, SAMPLE_RATE)

    assert report.seam_level_db_before < -20  # tail far quieter than head
    assert abs(report.seam_level_db_after) < 3  # the two ends now match
    assert report.trimmed_tail_s > 0.25
    assert report.crossfade_s == pytest.approx(mastering.CROSSFADE_S)
    assert out.dtype == np.float32
    # And the wrap is continuous, not merely level-matched: the step across it is an
    # ordinary sample-to-sample move, not the lurch the untreated master had.
    assert report.seam_jump_after < 10 * float(np.median(np.abs(np.diff(out[:, 0]))))


def test_mastering_closes_a_seam_that_never_faded():
    """The other shape of the same defect: a master that simply stops mid-phrase. There
    is no level step to see, just a discontinuity where the last sample meets the first."""
    audio = bed()
    _, report = mastering.master_for_loop(audio, SAMPLE_RATE)
    assert report.seam_jump_before > 0.05
    assert report.seam_jump_after < 0.01


def test_a_clean_loopable_bed_is_left_close_to_alone():
    """Nothing to trim, nothing to repair — the pass should cost the crossfade and no
    more, so a track that was already fine doesn't get "fixed" into something else."""
    audio = bed(seconds=20)
    out, report = mastering.master_for_loop(audio, SAMPLE_RATE)
    assert (report.trimmed_head_s, report.trimmed_tail_s) == (0.0, 0.0)
    assert report.clicks_repaired == 0
    assert report.gain_db == 0.0
    assert len(out) == len(audio) - int(mastering.CROSSFADE_S * SAMPLE_RATE)


def test_switches_turn_each_stage_off():
    audio = with_fade_out(bed(), seconds=2.0)
    out, report = mastering.master_for_loop(audio, SAMPLE_RATE, declick=False, loop=False)
    assert len(out) == len(audio)
    assert report.clicks_found == 0
    assert report.crossfade_s == 0.0


def test_mono_input_is_accepted():
    out, report = mastering.master_for_loop(bed()[:, 0], SAMPLE_RATE)
    assert out.ndim == 2 and out.shape[1] == 1
    assert report.duration_s > 0
