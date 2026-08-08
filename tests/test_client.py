import pytest
from fastapi.testclient import TestClient

from bnb.client import StreamClient, StreamClientError
from bnb.server import app, engine


@pytest.fixture
def client():
    # TestClient is an httpx.Client that runs the real app in-process, so the
    # control client can drive it without opening a socket.
    with StreamClient(client=TestClient(app)) as c:
        yield c
    engine.stop()


def test_list_backgrounds(client):
    # Reads the real asset repo, which is legitimately empty mid-replan (deleting specs
    # is how a prompt change is rolled out), so assert the shape, not the count — the
    # same guard test_stream.py uses for the same reason.
    rows = client.list_backgrounds()
    assert all({"track_id", "summary", "rendered"} <= r.keys() for r in rows)


def test_get_background_reports_selection(client):
    client.stop()
    assert client.get_background()["background_id"] is None
    client.start(beat={"beat_hz": 8})
    assert client.get_background()["background_id"] is None  # started without a background


def test_set_background_starts_when_stopped(client):
    client.stop()
    state = client.set_background(None, beat_hz=10, volume=0.4)
    assert state["running"] is True
    assert state["beat"]["beat_hz"] == 10 and state["beat"]["volume"] == 0.4


def test_set_beat_volume_merges_over_current(client):
    client.start(beat={"carrier_hz": 400, "beat_hz": 6, "volume": 0.3, "waveform": "sine"})
    state = client.set_beat_volume(0.7)
    # only volume changes; frequency/carrier/waveform are preserved
    assert state["beat"]["volume"] == 0.7
    assert state["beat"]["beat_hz"] == 6 and state["beat"]["carrier_hz"] == 400


def test_set_beat_frequency_merges_over_current(client):
    client.start(beat={"carrier_hz": 400, "beat_hz": 6, "volume": 0.3, "waveform": "square"})
    state = client.set_beat_frequency(9.5)
    assert state["beat"]["beat_hz"] == 9.5
    assert state["beat"]["volume"] == 0.3 and state["beat"]["waveform"] == "square"


def test_beat_volume_defaults_when_no_beat(client):
    client.start(beat=None)  # no beat active
    state = client.set_beat_volume(0.5)
    assert state["beat"]["volume"] == 0.5  # created from defaults


def test_error_surfaces_detail(client):
    client.stop()
    with pytest.raises(StreamClientError):
        client.set_background("does_not_exist", beat_hz=10)  # unknown background -> 400


def test_isochronic_rejects_a_background(client):
    """doc §5: isochronic must never be mixed with a background — the pulsing is the
    entrainment signal, and a background reduces its effective modulation depth."""
    client.stop()
    with pytest.raises(StreamClientError):
        client.start(beat={"mode": "isochronic", "carrier_hz": 250}, background_id="anything")
