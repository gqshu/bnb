import numpy as np
import pytest
import soundfile as sf

from bnb.tone import (
    CARRIER_HZ,
    MAX_BEAT_HZ,
    SAMPLE_RATE,
    WAVEFORMS,
    render_binaural,
    write_wav,
)

BEAT_HZ = 8.0
DURATION_S = 4.0


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
