import numpy as np
from fastapi.testclient import TestClient

import bnb.stream as stream_mod
from bnb.server import app, engine
from bnb.stream import SHUFFLE, Beat, StreamEngine, to_int16_bytes


def dominant_hz(channel: np.ndarray, sample_rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(channel))
    freqs = np.fft.rfftfreq(len(channel), 1 / sample_rate)
    return float(freqs[np.argmax(spectrum)])


def test_beat_puts_the_offset_on_the_right_ear():
    eng = StreamEngine()
    eng.start(beat=Beat(carrier_hz=300.0, beat_hz=8.0), background_id=None, background_volume=1.0)
    frames = np.concatenate([eng.read(eng.sample_rate) for _ in range(2)])  # 2 s
    assert abs(dominant_hz(frames[:, 0], eng.sample_rate) - 300.0) < 1.0
    assert abs(dominant_hz(frames[:, 1], eng.sample_rate) - 308.0) < 1.0


def test_phase_is_continuous_across_chunks():
    # Reading 2n in one call must equal reading n twice — i.e. phase carries over.
    def render_flat(chunks):
        eng = StreamEngine()
        eng.start(beat=Beat(carrier_hz=440.0, beat_hz=6.0), background_id=None, background_volume=1.0)
        eng._amp = eng.beat.volume  # skip the fade-in ramp so only phase differs
        return np.concatenate([eng.read(n) for n in chunks])

    one_chunk = render_flat([2000])
    two_chunks = render_flat([1000, 1000])
    assert np.allclose(one_chunk, two_chunks, atol=1e-6)


def test_no_beat_no_background_is_silence():
    eng = StreamEngine()
    eng.start(beat=None, background_id=None, background_volume=1.0)
    assert np.all(eng.read(1000) == 0.0)


def test_to_int16_bytes_shape_and_clip():
    frames = np.array([[2.0, -2.0], [0.0, 1.0]], dtype=np.float32)  # first row over-range
    data = to_int16_bytes(frames)
    assert len(data) == 2 * 2 * 2  # 2 frames x 2 ch x 2 bytes
    pcm = np.frombuffer(data, dtype="<i2")
    assert pcm.max() <= 32767 and pcm.min() >= -32767


# --- API -------------------------------------------------------------------

client = TestClient(app)


def teardown_function():
    engine.stop()


def test_start_stop_and_state():
    body = {"beat": {"carrier_hz": 432, "beat_hz": 10, "volume": 0.3, "waveform": "sine"}}
    started = client.post("/api/stream/start", json=body).json()
    assert started["running"] is True
    assert started["beat"]["beat_hz"] == 10
    assert client.get("/api/stream").json()["running"] is True
    assert client.post("/api/stream/stop").json()["running"] is False


def test_patch_updates_beat_and_can_disable_it():
    client.post("/api/stream/start", json={"beat": {"beat_hz": 8}})
    updated = client.patch("/api/stream/spec", json={"beat": {"beat_hz": 4, "volume": 0.5}}).json()
    assert updated["beat"]["beat_hz"] == 4 and updated["beat"]["volume"] == 0.5
    assert client.patch("/api/stream/spec", json={"beat": None}).json()["beat"] is None


def _fake_library(monkeypatch, eng, tracks, frames=80):
    """Point an engine at a set of tiny in-memory 'rendered' tracks (no disk)."""
    monkeypatch.setattr(stream_mod.assets, "list_rendered", lambda: list(tracks))
    monkeypatch.setattr(eng, "_read_asset",
                        lambda tid: np.full((frames, 2), 0.1, dtype=np.float32))


def test_shuffle_picks_a_track_and_reports_the_flag(monkeypatch):
    eng = StreamEngine()
    _fake_library(monkeypatch, eng, ["a", "b", "c"])
    eng.start(beat=None, background_id=SHUFFLE, background_volume=1.0)
    snap = eng.snapshot()
    assert snap["shuffle"] is True
    assert snap["background_id"] in ("a", "b", "c")


def test_shuffle_hands_off_near_the_end_of_a_track(monkeypatch):
    eng = StreamEngine()
    _fake_library(monkeypatch, eng, ["a", "b"], frames=4000)
    eng.start(beat=None, background_id=SHUFFLE, background_volume=1.0)
    first = eng.background_id
    # Jump to just inside the final fade-out window and render one chunk: the
    # current track's tail should hand off (crossfade) to the *other* track.
    eng._bg_pos = eng._bg.shape[0] - 10
    eng.read(64)
    assert eng.background_id != first  # advanced to the other rendered track
    assert eng._bg_out is not None     # the outgoing track is now fading out
    assert eng.shuffle is True


def test_pinning_a_track_turns_shuffle_off(monkeypatch):
    eng = StreamEngine()
    _fake_library(monkeypatch, eng, ["a", "b"])
    eng.start(beat=None, background_id=SHUFFLE, background_volume=1.0)
    assert eng.shuffle is True
    eng.set_background("a")
    assert eng.shuffle is False and eng.background_id == "a"


def test_shuffle_over_the_api_reports_a_real_rendered_track():
    """End-to-end against the real asset repo — starts and reports a playable track."""
    snap = client.post("/api/stream/start",
                       json={"background_id": SHUFFLE}).json()
    from bnb import assets
    rendered = assets.list_rendered()
    if rendered:  # only assert when the repo actually has rendered audio
        assert snap["shuffle"] is True
        assert snap["background_id"] in rendered
    client.post("/api/stream/stop")


def test_start_rejects_unknown_background():
    res = client.post("/api/stream/start", json={"background_id": "does_not_exist"})
    assert res.status_code == 400


def test_beat_bounds_are_validated():
    assert client.post("/api/stream/start", json={"beat": {"beat_hz": 999}}).status_code == 422
    assert client.post("/api/stream/start", json={"beat": {"beat_hz": -1}}).status_code == 422


def test_zero_beat_is_accepted_as_the_sham_condition():
    """Δ=0 means carrier in both ears and no beat — the EEG pilot's sham arm."""
    started = client.post("/api/stream/start", json={"beat": {"beat_hz": 0}}).json()
    assert started["running"] is True and started["beat"]["beat_hz"] == 0
    # Both ears must be bit-identical: a beat would show up as a channel difference.
    chunk = engine.read(2048)
    assert np.array_equal(chunk[:, 0], chunk[:, 1])
    client.post("/api/stream/stop")


def test_stream_wav_when_stopped_is_just_a_header():
    engine.stop()
    data = client.get("/stream.wav").content
    assert data[:4] == b"RIFF" and len(data) == 44


def test_backgrounds_lists_specs():
    rows = client.get("/api/backgrounds").json()
    assert isinstance(rows, list)
    assert all({"track_id", "summary", "rendered"} <= row.keys() for row in rows)
