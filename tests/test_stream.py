import numpy as np
import pytest
from fastapi.testclient import TestClient

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
    entries = [{"track_id": tid, "rendered": True} for tid in tracks]
    monkeypatch.setattr(eng._categories, "search", lambda **_: list(entries))
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


def test_repeat_crossfades_a_pinned_track_into_its_own_loop(monkeypatch):
    """A pinned track loops forever, and the seam is crossfaded rather than spliced."""
    eng = StreamEngine()
    _fake_library(monkeypatch, eng, ["a", "b"], frames=4000)
    eng.start(beat=None, background_id="a", background_volume=1.0)
    assert eng.shuffle is False
    eng._bg_pos = eng._bg.shape[0] - 10  # just inside the final fade-out window
    eng.read(64)
    assert eng.background_id == "a"      # still the same track — repeat, not shuffle
    assert eng._bg_out is not None       # ...but its tail is crossfading out
    assert eng._bg_pos < 100             # ...while the head restarts from the top


def test_diotic_mode_puts_the_beat_in_both_ears_identically():
    """The ASSR control: both ears carry the same summed tones — a real acoustic beat,
    but no interaural difference, so the channels are bit-identical."""
    eng = StreamEngine()
    eng.start(beat=Beat(carrier_hz=400.0, beat_hz=40.0, mode="diotic"),
              background_id=None, background_volume=1.0)
    frames = eng.read(4096)
    assert np.array_equal(frames[:, 0], frames[:, 1])            # diotic: L == R
    # The summed tones beat at 40 Hz — the amplitude envelope is modulated, unlike a
    # single-tone dichotic ear, so the signal is not a pure carrier.
    assert np.ptp(np.abs(frames[:, 0])) > 0.1
    eng2 = StreamEngine()
    eng2.start(beat=Beat(carrier_hz=400.0, beat_hz=40.0, mode="dichotic"),
               background_id=None, background_volume=1.0)
    di = eng2.read(4096)
    assert not np.array_equal(di[:, 0], di[:, 1])                # dichotic: L != R
    eng.stop(); eng2.stop()


def test_beat_range_extends_to_40hz():
    """The stream accepts the full experimental Δ range (percept weakens past ~30)."""
    snap = client.post("/api/stream/start", json={"beat": {"beat_hz": 40}}).json()
    assert snap["beat"]["beat_hz"] == 40
    assert client.post("/api/stream/start",
                       json={"beat": {"beat_hz": 41}}).status_code == 422
    client.post("/api/stream/stop")


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


# --- monaural ----------------------------------------------------------------


def test_monaural_mode_matches_diotic_signal():
    """"monaural" is the product-facing name for the exact same summed-tone signal
    "diotic" already renders (doc §3) — not a rename, but the same DSP branch."""
    eng = StreamEngine()
    eng.start(beat=Beat(carrier_hz=400.0, beat_hz=40.0, mode="monaural"),
              background_id=None, background_volume=1.0)
    mono_frames = eng.read(4096)
    assert np.array_equal(mono_frames[:, 0], mono_frames[:, 1])  # monaural: L == R

    eng2 = StreamEngine()
    eng2.start(beat=Beat(carrier_hz=400.0, beat_hz=40.0, mode="diotic"),
               background_id=None, background_volume=1.0)
    diotic_frames = eng2.read(4096)
    assert np.array_equal(mono_frames, diotic_frames)
    eng.stop(); eng2.stop()


# --- isochronic ----------------------------------------------------------------


def test_isochronic_mode_gates_an_identical_signal_in_both_ears():
    eng = StreamEngine()
    eng.start(beat=Beat(carrier_hz=250.0, beat_hz=10.0, mode="isochronic", volume=0.5),
              background_id=None, background_volume=1.0)
    eng._amp = eng.beat.volume  # skip the fade-in ramp so the envelope is clean
    frames = eng.read(eng.sample_rate * 2)  # 2 s
    assert np.array_equal(frames[:, 0], frames[:, 1])  # isochronic: L == R (mono)

    envelope = np.abs(frames[:, 0])
    spectrum = np.abs(np.fft.rfft(envelope - envelope.mean()))
    freqs = np.fft.rfftfreq(len(envelope), 1 / eng.sample_rate)
    assert freqs[np.argmax(spectrum)] == pytest.approx(10.0, abs=0.5)
    eng.stop()


def test_isochronic_phase_is_continuous_across_chunks():
    def render_flat(chunks):
        eng = StreamEngine()
        eng.start(beat=Beat(carrier_hz=250.0, beat_hz=10.0, mode="isochronic"),
                  background_id=None, background_volume=1.0)
        eng._amp = eng.beat.volume
        return np.concatenate([eng.read(n) for n in chunks])

    one_chunk = render_flat([2000])
    two_chunks = render_flat([1000, 1000])
    assert np.allclose(one_chunk, two_chunks, atol=1e-6)


def test_isochronic_beat_rejects_background_at_start():
    eng = StreamEngine()
    with pytest.raises(ValueError):
        eng.start(beat=Beat(mode="isochronic"), background_id="anything", background_volume=1.0)


def test_isochronic_beat_rejects_background_via_set_background():
    eng = StreamEngine()
    eng.start(beat=Beat(mode="isochronic"), background_id=None, background_volume=1.0)
    with pytest.raises(ValueError):
        eng.set_background("anything")
    eng.stop()


def test_background_rejects_isochronic_via_set_beat(monkeypatch):
    eng = StreamEngine()
    _fake_library(monkeypatch, eng, ["a"])
    eng.start(beat=Beat(mode="dichotic"), background_id="a", background_volume=1.0)
    with pytest.raises(ValueError):
        eng.set_beat(Beat(mode="isochronic"))
    eng.stop()


def test_update_allows_switching_to_isochronic_while_clearing_background(monkeypatch):
    """A single combined update (new isochronic beat + background_id=None) must be
    judged by what it ends up as, not by which field a field-by-field application
    would apply first — that ordering bug would spuriously reject this."""
    eng = StreamEngine()
    _fake_library(monkeypatch, eng, ["a"])
    eng.start(beat=Beat(mode="dichotic"), background_id="a", background_volume=1.0)
    eng.update(beat=Beat(mode="isochronic"), background_id=None)
    assert eng.beat.mode == "isochronic"
    assert eng.background_id is None
    eng.stop()


def test_update_rejects_isochronic_combined_with_a_background():
    eng = StreamEngine()
    eng.start(beat=Beat(mode="dichotic"), background_id=None, background_volume=1.0)
    with pytest.raises(ValueError):
        eng.update(beat=Beat(mode="isochronic"), background_id="anything")
    eng.stop()


def test_isochronic_over_the_api_rejects_a_background():
    res = client.post(
        "/api/stream/start",
        json={"beat": {"mode": "isochronic", "carrier_hz": 250}, "background_id": "anything"},
    )
    assert res.status_code == 400
    client.post("/api/stream/stop")


def test_isochronic_carrier_range_is_validated_over_the_api():
    assert client.post(
        "/api/stream/start", json={"beat": {"mode": "isochronic", "carrier_hz": 600}}
    ).status_code == 422
    assert client.post(
        "/api/stream/start", json={"beat": {"mode": "dichotic", "carrier_hz": 150}}
    ).status_code == 422
    client.post("/api/stream/stop")


# --- am_music (focus goal: modulates the background itself, opposite constraint
#     from isochronic — a background is required, not forbidden) ----------------


def test_am_music_requires_background_at_start():
    eng = StreamEngine()
    with pytest.raises(ValueError, match="requires a background"):
        eng.start(beat=Beat(mode="am_music", beat_hz=10.0), background_id=None, background_volume=1.0)


def test_am_music_rejects_clearing_background_via_set_background(monkeypatch):
    eng = StreamEngine()
    _fake_library(monkeypatch, eng, ["a"], frames=200_000)
    eng.start(beat=Beat(mode="am_music", beat_hz=10.0), background_id="a", background_volume=1.0)
    with pytest.raises(ValueError, match="requires a background"):
        eng.set_background(None)
    eng.stop()


def test_am_music_rejected_via_set_beat_when_no_background(monkeypatch):
    eng = StreamEngine()
    eng.start(beat=Beat(mode="dichotic"), background_id=None, background_volume=1.0)
    with pytest.raises(ValueError, match="requires a background"):
        eng.set_beat(Beat(mode="am_music", beat_hz=10.0))
    eng.stop()


def test_update_rejects_am_music_without_a_background():
    eng = StreamEngine()
    eng.start(beat=Beat(mode="dichotic"), background_id=None, background_volume=1.0)
    with pytest.raises(ValueError, match="requires a background"):
        eng.update(beat=Beat(mode="am_music", beat_hz=10.0))
    eng.stop()


def test_update_allows_switching_to_am_music_while_setting_a_background(monkeypatch):
    eng = StreamEngine()
    _fake_library(monkeypatch, eng, ["a"], frames=200_000)
    eng.start(beat=Beat(mode="dichotic"), background_id=None, background_volume=1.0)
    eng.update(beat=Beat(mode="am_music", beat_hz=10.0), background_id="a")
    assert eng.beat.mode == "am_music"
    assert eng.background_id == "a"
    eng.stop()


def test_am_music_modulates_the_background(monkeypatch):
    eng = StreamEngine()
    _fake_library(monkeypatch, eng, ["a"], frames=200_000)  # bigger than the read, no looping
    eng.start(
        beat=Beat(mode="am_music", beat_hz=10.0, depth=1.0, modulator="sine"),
        background_id="a",
        background_volume=1.0,
    )
    frames = eng.read(eng.sample_rate * 2)  # 2 s
    assert np.array_equal(frames[:, 0], frames[:, 1])  # constant bg -> shared envelope in both ears

    signal = frames[:, 0] - frames[:, 0].mean()
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1 / eng.sample_rate)
    assert freqs[np.argmax(spectrum)] == pytest.approx(10.0, abs=0.5)
    eng.stop()


def test_am_music_no_beat_at_all_when_no_background_is_playing(monkeypatch):
    # A fresh engine with am_music set but background not yet loaded (e.g. mid-switch)
    # must render silence, never a bare unmodulated tone — there is no tone in this mode.
    eng = StreamEngine()
    eng.beat = Beat(mode="am_music", beat_hz=10.0)
    frames = eng.read(100)
    assert np.array_equal(frames, np.zeros_like(frames))


def test_am_music_phase_is_continuous_across_chunks(monkeypatch):
    def render_flat(chunks):
        eng = StreamEngine()
        _fake_library(monkeypatch, eng, ["a"], frames=200_000)
        eng.start(beat=Beat(mode="am_music", beat_hz=10.0, depth=1.0), background_id="a", background_volume=1.0)
        eng._fade_in = 1.0  # skip the background's own fade-in ramp so only AM phase differs
        return np.concatenate([eng.read(n) for n in chunks])

    one_chunk = render_flat([2000])
    two_chunks = render_flat([1000, 1000])
    assert np.allclose(one_chunk, two_chunks, atol=1e-6)


def test_am_music_over_the_api_requires_a_background():
    res = client.post("/api/stream/start", json={"beat": {"mode": "am_music", "beat_hz": 10}})
    assert res.status_code == 400
    client.post("/api/stream/stop")


def test_am_music_beat_range_extends_to_60hz_over_the_api():
    """am_music has no binaural-fusion ceiling, so it's allowed past tone.MAX_BEAT_HZ (40)."""
    assert client.post(
        "/api/stream/start", json={"beat": {"mode": "am_music", "beat_hz": 60}}
    ).status_code == 400  # valid Hz, rejected only for lacking a background
    assert client.post(
        "/api/stream/start", json={"beat": {"mode": "am_music", "beat_hz": 61}}
    ).status_code == 422  # over the am_music ceiling
    assert client.post(
        "/api/stream/start", json={"beat": {"mode": "dichotic", "beat_hz": 41}}
    ).status_code == 422  # the ordinary tone-mode ceiling is unchanged
    client.post("/api/stream/stop")


# ── Per-session API (concurrent independent streams) ──────────────────────────
def test_session_lifecycle_create_start_stop_delete():
    sid = client.post("/api/session").json()["session_id"]
    assert client.get(f"/api/session/{sid}").json()["running"] is False
    started = client.post(f"/api/session/{sid}/start", json={"beat": {"beat_hz": 8}}).json()
    assert started["running"] is True and started["beat"]["beat_hz"] == 8
    assert client.post(f"/api/session/{sid}/stop").json()["running"] is False
    assert client.delete(f"/api/session/{sid}").json()["ok"] is True
    # gone after delete
    assert client.get(f"/api/session/{sid}").status_code == 404


def test_sessions_are_independent():
    a = client.post("/api/session").json()["session_id"]
    b = client.post("/api/session").json()["session_id"]
    assert a != b
    client.post(f"/api/session/{a}/start", json={"beat": {"beat_hz": 6}})
    client.post(f"/api/session/{b}/start", json={"beat": {"beat_hz": 12}})
    assert client.get(f"/api/session/{a}").json()["beat"]["beat_hz"] == 6
    assert client.get(f"/api/session/{b}").json()["beat"]["beat_hz"] == 12
    # editing one leaves the other untouched
    client.patch(f"/api/session/{a}/spec", json={"beat": {"beat_hz": 4}})
    assert client.get(f"/api/session/{a}").json()["beat"]["beat_hz"] == 4
    assert client.get(f"/api/session/{b}").json()["beat"]["beat_hz"] == 12
    client.delete(f"/api/session/{a}")
    client.delete(f"/api/session/{b}")


def test_unknown_session_is_404():
    assert client.get("/api/session/nope").status_code == 404
    assert client.post("/api/session/nope/start", json={"beat": {"beat_hz": 8}}).status_code == 404
    assert client.patch("/api/session/nope/spec", json={"beat": {"beat_hz": 8}}).status_code == 404


def test_session_stream_endpoint_serves_wav_header():
    # Not started → the generator yields the WAV header then ends. Stream and read
    # just the first raw chunk (the header); don't consume the fake Content-Length.
    sid = client.post("/api/session").json()["session_id"]
    with client.stream("GET", f"/stream/{sid}.wav") as r:
        assert r.status_code == 200
        assert r.headers["content-type"] == "audio/wav"
        first = next(r.iter_raw())
        assert first[:4] == b"RIFF" and first[8:12] == b"WAVE"
    client.delete(f"/api/session/{sid}")


# ── MP3 stream + blank-spec default background ────────────────────────────────
def test_empty_spec_start_picks_a_natural_sounds_background():
    sid = client.post("/api/session").json()["session_id"]
    snap = client.post(f"/api/session/{sid}/start", json={}).json()
    assert snap["running"] is True
    assert snap["beat"] is None
    assert snap["background_id"] is not None
    assert snap["background_id"].startswith("natural_sounds_")
    client.post(f"/api/session/{sid}/stop")
    client.delete(f"/api/session/{sid}")


def test_session_mp3_endpoint_content_type():
    # Not started → empty body; stream so headers read without waiting on the body.
    sid = client.post("/api/session").json()["session_id"]
    with client.stream("GET", f"/stream/{sid}.mp3") as r:
        assert r.status_code == 200
        assert r.headers["content-type"] == "audio/mpeg"
    client.delete(f"/api/session/{sid}")


def test_stream_supports_range_requests_for_android():
    # Android's player opens with `Range: bytes=0-` and needs a 206 + Content-Range,
    # otherwise it errors out ("error player to stop"). Verify the handshake.
    sid = client.post("/api/session").json()["session_id"]
    with client.stream("GET", f"/stream/{sid}.mp3", headers={"Range": "bytes=0-"}) as r:
        assert r.status_code == 206
        assert r.headers["accept-ranges"] == "bytes"
        assert "content-range" in r.headers
    client.delete(f"/api/session/{sid}")


def test_mp3_encoding_pipeline_produces_frames():
    # Drive the same encode path the endpoint uses, but bounded (no open stream).
    import lameenc

    eng = StreamEngine()
    eng.start(beat=Beat(beat_hz=10), background_id=None, background_volume=1.0)
    enc = lameenc.Encoder()
    enc.set_bit_rate(128)
    enc.set_in_sample_rate(eng.sample_rate)
    enc.set_channels(2)
    enc.set_quality(5)
    out = b""
    for _ in range(5):
        out += bytes(enc.encode(to_int16_bytes(eng.read(int(eng.sample_rate * 0.2)))))
    out += bytes(enc.flush())
    eng.stop()
    assert len(out) > 0
    assert b"\xff" in out  # MP3 frame sync high byte


def test_random_background_returns_playable_track():
    res = client.get("/api/backgrounds/random").json()
    assert res["track_id"].startswith("natural_sounds_")
    assert res["url"] == f"/background/{res['track_id']}.mp3"
    assert res["name"]  # non-empty display name


def test_random_special_background_excludes_goal_incompatible_keywords(monkeypatch):
    """A keyword restricted to one goal (KeywordEntry.goals) must never be picked for
    a session of the other goal, even though the special-group spec itself carries no
    per-track goal (that's metadata on the taxonomy, not the render)."""
    from dataclasses import replace

    from bnb import server
    from bnb.background import SPECIAL_GROUPS, KeywordEntry

    group = SPECIAL_GROUPS["natural_sounds"]
    patched_keywords = dict(group.keywords)
    patched_keywords["cafe"] = KeywordEntry("Coffee shop murmur", goals=frozenset({"focus"}))
    monkeypatch.setitem(SPECIAL_GROUPS, "natural_sounds", replace(group, keywords=patched_keywords))

    fake_entries = [
        {"track_id": "natural_sounds_rain_seedX", "keyword": "rain", "group": "natural_sounds"},
        {"track_id": "natural_sounds_cafe_seedY", "keyword": "cafe", "group": "natural_sounds"},
    ]
    monkeypatch.setattr(
        server.categories,
        "search",
        lambda **kw: [e for e in fake_entries if kw.get("group") in (None, e["group"])],
    )

    relax_picks = {server._random_special_background("natural_sounds", "relax") for _ in range(20)}
    focus_picks = {server._random_special_background("natural_sounds", "focus") for _ in range(20)}
    assert relax_picks == {"natural_sounds_rain_seedX"}
    assert "natural_sounds_cafe_seedY" in focus_picks


def test_random_background_endpoint_accepts_a_goal_query_param():
    res = client.get("/api/backgrounds/random", params={"goal": "focus"})
    assert res.status_code == 200
    assert client.get("/api/backgrounds/random", params={"goal": "gamma"}).status_code == 422


def test_background_mp3_serves_complete_file():
    tid = client.get("/api/backgrounds/random").json()["track_id"]
    r = client.get(f"/background/{tid}.mp3")
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/mpeg"
    assert len(r.content) > 10000  # a real encoded file
    assert r.content[:2] == b"\xff\xfb"  # mp3 frame sync
