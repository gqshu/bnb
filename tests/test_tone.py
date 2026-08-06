import numpy as np
import pytest
import soundfile as sf

from bnb.tone import (
    CARRIER_HZ,
    MAX_AM_BEAT_HZ,
    ISOCHRONIC_CARRIER_HZ,
    MAX_BEAT_HZ,
    SAMPLE_RATE,
    WAVEFORMS,
    _gate_envelope,
    load_background,
    render_am_music,
    render_binaural,
    render_isochronic,
    render_monaural,
    write_wav,
)

BEAT_HZ = 8.0
DURATION_S = 4.0
ISO_BEAT_HZ = 10.0
ISO_DURATION_S = 4.0


def dominant_hz(channel: np.ndarray, sample_rate: int) -> float:
    """Frequency of the strongest FFT bin in a channel."""
    spectrum = np.abs(np.fft.rfft(channel))
    freqs = np.fft.rfftfreq(len(channel), 1 / sample_rate)
    return float(freqs[np.argmax(spectrum)])


@pytest.fixture
def wav_path(tmp_path):
    """A rendered WAV file: constant 432 Hz left, 440 Hz right."""
    path = tmp_path / "constant_8hz.wav"
    write_wav(path, render_binaural(beat_hz=BEAT_HZ, duration_s=DURATION_S))
    return path


def test_wav_is_stereo_at_expected_rate_and_length(wav_path):
    samples, sample_rate = sf.read(wav_path)

    assert sample_rate == SAMPLE_RATE
    assert samples.shape == (round(DURATION_S * SAMPLE_RATE), 2)


def test_each_ear_holds_its_own_constant_frequency(wav_path):
    samples, sample_rate = sf.read(wav_path)
    left, right = samples[:, 0], samples[:, 1]

    # 4 s of audio gives 0.25 Hz FFT resolution, so an 8 Hz beat is well resolved.
    assert dominant_hz(left, sample_rate) == pytest.approx(CARRIER_HZ, abs=0.5)
    assert dominant_hz(right, sample_rate) == pytest.approx(CARRIER_HZ + BEAT_HZ, abs=0.5)


def test_ears_are_loudness_matched(wav_path):
    samples, _ = sf.read(wav_path)
    left_rms = np.sqrt(np.mean(samples[:, 0] ** 2))
    right_rms = np.sqrt(np.mean(samples[:, 1] ** 2))

    # An interaural imbalance lateralizes the image and breaks immersion.
    assert left_rms == pytest.approx(right_rms, rel=1e-3)


def test_signal_does_not_clip(wav_path):
    samples, _ = sf.read(wav_path)

    assert np.max(np.abs(samples)) < 1.0


@pytest.mark.parametrize("waveform", WAVEFORMS)
def test_every_waveform_keeps_its_per_ear_frequency(waveform, tmp_path):
    path = tmp_path / f"{waveform}.wav"
    write_wav(path, render_binaural(BEAT_HZ, DURATION_S, waveform=waveform))
    samples, sample_rate = sf.read(path)

    # Non-sine shapes carry harmonics, but the fundamental stays the loudest bin.
    assert dominant_hz(samples[:, 0], sample_rate) == pytest.approx(CARRIER_HZ, abs=0.5)
    assert dominant_hz(samples[:, 1], sample_rate) == pytest.approx(
        CARRIER_HZ + BEAT_HZ, abs=0.5
    )


def test_waveform_defaults_to_sine():
    assert np.array_equal(
        render_binaural(BEAT_HZ, 0.1),
        render_binaural(BEAT_HZ, 0.1, waveform="sine"),
    )


def test_sine_is_the_purest_waveform():
    """Sine should put essentially all its energy in the fundamental; the shapes with
    harmonics should not. This is the reason sine is the default."""
    fundamental_share = {}
    for waveform in WAVEFORMS:
        left = render_binaural(BEAT_HZ, DURATION_S, waveform=waveform)[:, 0]
        spectrum = np.abs(np.fft.rfft(left)) ** 2
        fundamental_share[waveform] = np.max(spectrum) / np.sum(spectrum)

    assert fundamental_share["sine"] > 0.99
    assert all(
        fundamental_share["sine"] > fundamental_share[w] for w in WAVEFORMS if w != "sine"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"beat_hz": MAX_BEAT_HZ},  # the offline renderer stays exclusive of the bound
        {"beat_hz": 0.0},
        {"beat_hz": BEAT_HZ, "carrier_hz": 1000.0},  # above the usable carrier range
        {"beat_hz": BEAT_HZ, "carrier_hz": 100.0},
        {"beat_hz": BEAT_HZ, "duration_s": 0.0},
        {"beat_hz": BEAT_HZ, "waveform": "noise"},
    ],
)
def test_rejects_parameters_outside_the_usable_range(kwargs):
    kwargs = {"duration_s": DURATION_S, **kwargs}

    with pytest.raises(ValueError):
        render_binaural(**kwargs)


# --- monaural ----------------------------------------------------------------


@pytest.fixture
def monaural_wav_path(tmp_path):
    """A rendered monaural WAV: the same summed signal in both channels."""
    path = tmp_path / "monaural_8hz.wav"
    write_wav(path, render_monaural(beat_hz=BEAT_HZ, duration_s=DURATION_S))
    return path


def test_monaural_channels_are_identical(monaural_wav_path):
    samples, _ = sf.read(monaural_wav_path)
    assert np.array_equal(samples[:, 0], samples[:, 1])


def test_monaural_spectrum_contains_both_tones(monaural_wav_path):
    samples, sample_rate = sf.read(monaural_wav_path)
    mono = samples[:, 0]
    spectrum = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(len(mono), 1 / sample_rate)
    noise_floor = np.median(spectrum)

    def bin_at(freq):
        return spectrum[np.argmin(np.abs(freqs - freq))]

    # Both symmetric tones should be strong peaks, not just one dominant carrier.
    assert bin_at(CARRIER_HZ - BEAT_HZ / 2) > 50 * noise_floor
    assert bin_at(CARRIER_HZ + BEAT_HZ / 2) > 50 * noise_floor


def test_monaural_signal_does_not_clip(monaural_wav_path):
    samples, _ = sf.read(monaural_wav_path)
    assert np.max(np.abs(samples)) < 1.0


def test_monaural_rejects_parameters_outside_the_usable_range():
    with pytest.raises(ValueError):
        render_monaural(beat_hz=BEAT_HZ, duration_s=DURATION_S, carrier_hz=1000.0)


# --- isochronic ----------------------------------------------------------------


@pytest.fixture
def isochronic_wav_path(tmp_path):
    """A rendered isochronic WAV: identical L/R, gated at ISO_BEAT_HZ."""
    path = tmp_path / "isochronic_10hz.wav"
    write_wav(path, render_isochronic(beat_hz=ISO_BEAT_HZ, duration_s=ISO_DURATION_S))
    return path


def test_isochronic_channels_are_identical(isochronic_wav_path):
    samples, _ = sf.read(isochronic_wav_path)
    assert np.array_equal(samples[:, 0], samples[:, 1])


def test_isochronic_gate_frequency_matches_beat_hz(isochronic_wav_path):
    samples, sample_rate = sf.read(isochronic_wav_path)
    envelope = np.abs(samples[:, 0])  # the gate rate shows up in the rectified envelope
    spectrum = np.abs(np.fft.rfft(envelope - envelope.mean()))
    freqs = np.fft.rfftfreq(len(envelope), 1 / sample_rate)
    assert freqs[np.argmax(spectrum)] == pytest.approx(ISO_BEAT_HZ, abs=0.5)


def test_isochronic_carrier_and_sidebands_present():
    samples = render_isochronic(beat_hz=ISO_BEAT_HZ, duration_s=ISO_DURATION_S)
    mono = samples[:, 0]
    spectrum = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(len(mono), 1 / SAMPLE_RATE)
    noise_floor = np.median(spectrum)

    def bin_at(freq):
        return spectrum[np.argmin(np.abs(freqs - freq))]

    assert bin_at(ISOCHRONIC_CARRIER_HZ) > 50 * noise_floor
    assert bin_at(ISOCHRONIC_CARRIER_HZ - ISO_BEAT_HZ) > 20 * noise_floor
    assert bin_at(ISOCHRONIC_CARRIER_HZ + ISO_BEAT_HZ) > 20 * noise_floor


def test_isochronic_signal_does_not_clip():
    samples = render_isochronic(beat_hz=ISO_BEAT_HZ, duration_s=ISO_DURATION_S)
    assert np.max(np.abs(samples)) < 1.0


def test_isochronic_depth_zero_means_no_gating():
    """depth=0 leaves env == 1 always, so the tone is effectively ungated."""
    samples = render_isochronic(beat_hz=ISO_BEAT_HZ, duration_s=ISO_DURATION_S, depth=0.0, amplitude=0.3)
    rms = np.sqrt(np.mean(samples[:, 0] ** 2))
    assert rms == pytest.approx(0.3 / np.sqrt(2), rel=0.05)


def test_gate_envelope_ramping_suppresses_harmonics():
    """doc §4.4/§7: a hard gate spreads energy across many harmonics of the beat
    frequency; the mandatory raised-cosine ramp should suppress them sharply.

    A duty=0.5 gate has no even harmonics, so this sums energy across a band of
    higher odd harmonics (5th-15th) rather than a single bin, which would be
    fragile against incidental near-zero crossings at any one harmonic."""
    n = 4096
    phi = (np.arange(n) / n) % 1.0  # one full gate cycle, densely sampled
    duty = 0.5

    hard = _gate_envelope(phi, duty, ramp_frac=0.0)
    ramped = _gate_envelope(phi, duty, ramp_frac=0.1)

    def higher_harmonic_energy(gate):
        spectrum = np.abs(np.fft.rfft(gate - gate.mean()))
        return sum(spectrum[k] ** 2 for k in range(5, 16, 2))

    assert higher_harmonic_energy(ramped) < 0.2 * higher_harmonic_energy(hard)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"carrier_hz": 50.0},  # below isochronic's usable range
        {"carrier_hz": 600.0},  # above isochronic's usable range
        {"depth": 1.5},
        {"duty": 0.05},
        {"ramp_ms": 20.0},
        {"beat_hz": 0.0},
        {"waveform": "noise"},
    ],
)
def test_isochronic_rejects_parameters_outside_the_usable_range(kwargs):
    kwargs = {"beat_hz": ISO_BEAT_HZ, "duration_s": ISO_DURATION_S, **kwargs}

    with pytest.raises(ValueError):
        render_isochronic(**kwargs)


# --- AM music (doc §6) ------------------------------------------------------

AM_BEAT_HZ = 40.0  # the gamma target; AM has no binaural-fusion ceiling to respect


def music_bed(seconds: float = 4.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """A stand-in for a rendered background: broadband, steady, non-silent."""
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.2, size=(round(seconds * sample_rate), 2))
    return np.clip(noise, -1.0, 1.0).astype(np.float32)


def envelope_hz(channel: np.ndarray, sample_rate: int) -> float:
    """Dominant frequency of the signal's amplitude envelope."""
    env = np.abs(channel)
    env = env - env.mean()
    return dominant_hz(env, sample_rate)


def test_am_music_modulates_at_the_beat_frequency():
    out = render_am_music(music_bed(), beat_hz=AM_BEAT_HZ, depth=1.0)
    assert envelope_hz(out[:, 0], SAMPLE_RATE) == pytest.approx(AM_BEAT_HZ, abs=0.5)


def test_am_music_depth_sets_the_trough_level():
    """The envelope spans [1 - depth, 1], so depth alone fixes how far the bed ducks."""
    bed = music_bed()
    for depth in (0.25, 0.5, 1.0):
        out = render_am_music(bed, beat_hz=10.0, depth=depth, modulator="sine")
        ratio = np.abs(out[:, 0]).max() and out[:, 0] / np.where(bed[:, 0] == 0, np.nan, bed[:, 0])
        assert np.nanmin(ratio) == pytest.approx(1 - depth, abs=0.02)
        assert np.nanmax(ratio) == pytest.approx(1.0, abs=0.02)


def test_am_music_depth_zero_returns_the_bed_untouched():
    bed = music_bed()
    out = render_am_music(bed, beat_hz=AM_BEAT_HZ, depth=0.0)
    assert np.allclose(out, bed, atol=1e-6)


def test_am_music_never_exceeds_the_beds_peak():
    """Envelope <= 1 everywhere, so modulation can only duck — it can't clip."""
    bed = music_bed()
    out = render_am_music(bed, beat_hz=AM_BEAT_HZ, depth=1.0, modulator="gate")
    assert np.max(np.abs(out)) <= np.max(np.abs(bed)) + 1e-6


def test_am_music_loops_a_short_bed_to_fill_the_duration():
    bed = music_bed(seconds=1.0)
    out = render_am_music(bed, beat_hz=10.0, duration_s=3.0)
    assert out.shape[0] == round(3.0 * SAMPLE_RATE)


def test_am_music_trims_a_long_bed():
    out = render_am_music(music_bed(seconds=4.0), beat_hz=10.0, duration_s=1.5)
    assert out.shape[0] == round(1.5 * SAMPLE_RATE)


def test_am_music_gate_modulator_dwells_at_the_extremes():
    """Same depth, different shape — this is the drive/pleasantness trade.

    A sine modulator sweeps continuously and only touches the trough instantaneously;
    the gate holds the music near-silent for a whole off-phase and near-full for the
    on-phase. That dwell is what makes the gate the more audible pulse (and the
    stronger drive), not a difference in total energy — a duty-0.5 gate actually
    carries *more* RMS than the sine, since sine spends most of its cycle mid-way.
    """
    bed = np.ones((SAMPLE_RATE, 2), dtype=np.float32)  # flat bed: output == envelope
    depth = 0.8
    sine = render_am_music(bed, beat_hz=10.0, depth=depth, modulator="sine")[:, 0]
    gate = render_am_music(bed, beat_hz=10.0, depth=depth, modulator="gate")[:, 0]

    near_trough = lambda env: float(np.mean(env < (1 - depth) + 0.01))
    near_peak = lambda env: float(np.mean(env > 0.99))

    # gate holds ~50% of the cycle at each extreme; sine passes through in ~7%.
    assert near_trough(gate) > 4 * near_trough(sine)
    assert near_peak(gate) > 4 * near_peak(sine)


def test_am_music_allows_gamma_above_the_binaural_ceiling():
    """Binaural fusion caps beats near 30-40 Hz; a physical envelope has no such limit."""
    out = render_am_music(music_bed(), beat_hz=MAX_AM_BEAT_HZ)
    assert out.shape[0] == music_bed().shape[0]

    with pytest.raises(ValueError, match="outside"):
        render_am_music(music_bed(), beat_hz=MAX_AM_BEAT_HZ + 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"beat_hz": 0.0},
        {"depth": 1.5},
        {"duty": 0.05},
        {"ramp_ms": 20.0},
        {"modulator": "noise"},
        {"duration_s": -1.0},
    ],
)
def test_am_music_rejects_parameters_outside_the_usable_range(kwargs):
    kwargs = {"beat_hz": AM_BEAT_HZ, **kwargs}
    with pytest.raises(ValueError):
        render_am_music(music_bed(), **kwargs)


def test_am_music_requires_stereo_bed():
    with pytest.raises(ValueError, match="stereo"):
        render_am_music(music_bed()[:, 0], beat_hz=AM_BEAT_HZ)


def test_load_background_resolves_an_asset_track_id():
    # assets/tracks/ is git-ignored (the masters are regenerable), so this only runs
    # where the library has actually been rendered.
    from bnb import assets

    track_id = "lofi_drone_seed47621"
    if not assets.has_track(track_id):
        pytest.skip(f"{track_id} not rendered locally")

    bed, sample_rate = load_background(track_id)
    assert sample_rate == 44_100
    assert bed.ndim == 2 and bed.shape[1] == 2


def test_load_background_reads_a_file_path_and_upmixes_mono(tmp_path):
    path = tmp_path / "mono.wav"
    sf.write(path, np.zeros(1000, dtype=np.float32) + 0.1, 22_050)
    bed, sample_rate = load_background(path)

    assert sample_rate == 22_050
    assert bed.shape == (1000, 2)
    assert np.allclose(bed[:, 0], bed[:, 1])


def test_load_background_resamples_when_asked(tmp_path):
    path = tmp_path / "bed.wav"
    sf.write(path, np.zeros((22_050, 2), dtype=np.float32), 22_050)
    bed, sample_rate = load_background(path, target_sample_rate=44_100)

    assert sample_rate == 44_100
    assert bed.shape[0] == 44_100


def test_load_background_rejects_an_unknown_source():
    with pytest.raises(FileNotFoundError):
        load_background("no_such_track_id")
