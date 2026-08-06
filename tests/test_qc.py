"""Background-track integrity checks.

Fixtures are synthesized rather than read from ``assets/tracks/``: the audio masters
are git-ignored, so a test that needs them fails on a fresh clone.
"""

import numpy as np
import pytest
import soundfile as sf

from bnb import qc

SAMPLE_RATE = 44_100
SECONDS = 10


def write(path, samples, sample_rate=SAMPLE_RATE):
    sf.write(path, np.clip(samples, -1.0, 1.0).astype(np.float32), sample_rate)
    return path


def bed(seconds=SECONDS, amplitude=0.3, seed=0):
    """A stand-in for a healthy ambient render: slow tonal movement, some texture."""
    rng = np.random.default_rng(seed)
    t = np.arange(seconds * SAMPLE_RATE) / SAMPLE_RATE
    # Two detuned partials with a slow swell, plus light noise for texture.
    swell = 0.6 + 0.4 * np.sin(2 * np.pi * 0.1 * t)
    tone = np.sin(2 * np.pi * 110 * t) + 0.7 * np.sin(2 * np.pi * 164.5 * t)
    # Noise scales with amplitude — a fixed floor would dominate the quiet fixtures
    # and stop them from being quiet at all.
    mono = amplitude * (swell * tone / 1.7 + rng.normal(0, 0.03, len(t)))
    return np.stack([mono, mono * 0.95], axis=1)


@pytest.fixture
def healthy(tmp_path):
    return write(tmp_path / "healthy.wav", bed())


def test_a_normal_ambient_bed_passes(healthy):
    report = qc.check_track(healthy)
    assert report.verdict == "ok", (report.failures, report.warnings)
    assert report.ok


def test_metrics_are_reported_even_when_the_track_passes(healthy):
    metrics = qc.check_track(healthy).metrics
    assert metrics["sample_rate"] == SAMPLE_RATE
    assert metrics["channels"] == 2
    assert metrics["duration_s"] == pytest.approx(SECONDS, abs=0.1)
    assert metrics["peak"] > 0


def test_digital_silence_fails(tmp_path):
    report = qc.check_track(write(tmp_path / "silent.wav", np.zeros((SECONDS * SAMPLE_RATE, 2))))
    assert report.verdict == "fail"
    assert any("silent" in f for f in report.failures)


def test_a_constant_tone_fails_as_a_dead_buffer(tmp_path):
    """Loud but never changing — a stuck buffer or test tone, not music."""
    t = np.arange(SECONDS * SAMPLE_RATE) / SAMPLE_RATE
    tone = np.tile((0.5 * np.sin(2 * np.pi * 220 * t))[:, None], (1, 2))
    report = qc.check_track(write(tmp_path / "tone.wav", tone))

    assert report.verdict == "fail"
    assert any("dead constant level" in f for f in report.failures)


def test_clipped_audio_fails(tmp_path):
    t = np.arange(SECONDS * SAMPLE_RATE) / SAMPLE_RATE
    smashed = np.tile((3.0 * np.sin(2 * np.pi * 220 * t))[:, None], (1, 2))
    report = qc.check_track(write(tmp_path / "clipped.wav", smashed))

    assert report.verdict == "fail"
    assert any("clipped" in f for f in report.failures)


def test_mostly_silent_render_fails(tmp_path):
    audible = bed(seconds=2)
    padded = np.concatenate([audible, np.zeros((8 * SAMPLE_RATE, 2))])
    report = qc.check_track(write(tmp_path / "half.wav", padded))

    assert report.verdict == "fail"
    assert any("mostly silence" in f for f in report.failures)


def test_inaudible_render_fails(tmp_path):
    """The real failure mode found in the library: a track rendered ~35 dB too quiet."""
    report = qc.check_track(write(tmp_path / "tiny.wav", bed(amplitude=0.0015)))

    assert report.verdict == "fail"
    assert any("inaudible" in f for f in report.failures)


def test_short_render_fails(tmp_path):
    report = qc.check_track(write(tmp_path / "short.wav", bed(seconds=2)))
    assert report.verdict == "fail"
    assert any("too short" in f for f in report.failures)


def test_min_duration_is_configurable(tmp_path):
    path = write(tmp_path / "short.wav", bed(seconds=2))
    assert qc.check_track(path, min_duration_s=1.0).verdict == "ok"


def test_unreadable_file_fails_without_raising(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"not audio at all")
    report = qc.check_track(path)

    assert report.verdict == "fail"
    assert any("unreadable" in f for f in report.failures)


def test_white_noise_warns_but_does_not_fail(tmp_path):
    """A hissy render is still playable, and `noise_texture` is a real library
    substrate — so this is a listen-to-it flag, not a re-render order."""
    rng = np.random.default_rng(0)
    report = qc.check_track(write(tmp_path / "noise.wav", rng.normal(0, 0.2, (SECONDS * SAMPLE_RATE, 2))))

    assert report.verdict == "warn"
    assert any("noise-like" in w for w in report.warnings)


def test_quiet_but_recoverable_render_warns(tmp_path):
    report = qc.check_track(write(tmp_path / "quiet.wav", bed(amplitude=0.01)))
    assert report.verdict == "warn"
    assert any("quiet" in w for w in report.warnings)


def test_dc_offset_warns(tmp_path):
    report = qc.check_track(write(tmp_path / "dc.wav", bed() + 0.2))
    assert any("DC offset" in w for w in report.warnings)


def test_check_path_scans_a_directory(tmp_path):
    write(tmp_path / "a.wav", bed())
    write(tmp_path / "b.wav", np.zeros((SECONDS * SAMPLE_RATE, 2)))
    (tmp_path / "notes.txt").write_text("not audio")

    reports = qc.check_path(tmp_path)

    assert [r.path.name for r in reports] == ["a.wav", "b.wav"]  # sorted, non-audio skipped
    assert [r.verdict for r in reports] == ["ok", "fail"]


def test_check_path_accepts_a_single_file(healthy):
    assert len(qc.check_path(healthy)) == 1


def test_check_path_rejects_a_missing_target(tmp_path):
    with pytest.raises(FileNotFoundError):
        qc.check_path(tmp_path / "nope")


def test_report_serializes_for_json(healthy):
    payload = qc.check_track(healthy).as_dict()
    assert set(payload) == {"path", "verdict", "failures", "warnings", "metrics"}
    assert payload["verdict"] == "ok"
