"""A thin control client for the live stream service (src/bnb/server.py).

Wraps the HTTP endpoints so you can drive the single stream from Python (or the
`scripts/control.py` CLI) instead of hand-rolling requests. The server's PATCH
replaces the whole beat object, so the volume/frequency helpers here read the
current beat, change one field, and send it back — the merge lives client-side.
"""

from __future__ import annotations

from typing import Any

import httpx

from .server import PORT
from .tone import CARRIER_HZ

DEFAULT_BASE_URL = f"http://127.0.0.1:{PORT}"


class StreamClientError(RuntimeError):
    """A non-2xx response from the stream service."""


class StreamClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        # `client` injection lets tests drive an in-process ASGI app.
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout)

    def __enter__(self) -> StreamClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, body: dict | None = None) -> Any:
        res = self._client.request(method, path, json=body)
        if res.is_error:
            try:
                detail = res.json().get("detail", res.text)
            except Exception:
                detail = res.text
            raise StreamClientError(f"{res.status_code} {method} {path}: {detail}")
        return res.json()

    # --- state -------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """The full stream state (running, beat, background_id, volumes)."""
        return self._request("GET", "/api/stream")

    def stop(self) -> dict[str, Any]:
        return self._request("POST", "/api/stream/stop")

    # 1. list background track meta
    def list_backgrounds(self) -> list[dict[str, Any]]:
        """Every background track with its summary and whether it's rendered/playable."""
        return self._request("GET", "/api/backgrounds")

    # 2. get current background
    def get_background(self) -> dict[str, Any]:
        """The currently selected background: its id, volume, and catalog meta (if any)."""
        state = self.get_state()
        bid = state["background_id"]
        meta = None
        if bid is not None:
            meta = next((b for b in self.list_backgrounds() if b["track_id"] == bid), None)
        return {"background_id": bid, "background_volume": state["background_volume"], "meta": meta}

    # 3. start or change the background with a given beat + volume combo
    def set_background(
        self,
        background_id: str | None,
        *,
        beat_hz: float | None = None,
        volume: float | None = None,
        carrier_hz: float = CARRIER_HZ,
        waveform: str = "sine",
        background_volume: float | None = None,
    ) -> dict[str, Any]:
        """Point the stream at ``background_id`` with a beat. Starts the stream if it's
        stopped, otherwise changes it live (the background crossfades)."""
        beat = None
        if beat_hz is not None:
            beat = {
                "carrier_hz": carrier_hz,
                "beat_hz": beat_hz,
                "volume": 0.3 if volume is None else volume,
                "waveform": waveform,
            }
        if not self.get_state()["running"]:
            return self.start(
                beat=beat,
                background_id=background_id,
                background_volume=1.0 if background_volume is None else background_volume,
            )
        fields: dict[str, Any] = {"background_id": background_id}
        if beat is not None:
            fields["beat"] = beat
        if background_volume is not None:
            fields["background_volume"] = background_volume
        return self._request("PATCH", "/api/stream/spec", fields)

    def start(
        self,
        *,
        beat: dict[str, Any] | None = None,
        background_id: str | None = None,
        background_volume: float = 1.0,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"background_volume": background_volume}
        if beat is not None:
            body["beat"] = beat
        if background_id is not None:
            body["background_id"] = background_id
        return self._request("POST", "/api/stream/start", body)

    # 4. change the beat volume
    def set_beat_volume(self, volume: float) -> dict[str, Any]:
        beat = self._current_beat()
        beat["volume"] = volume
        return self._request("PATCH", "/api/stream/spec", {"beat": beat})

    # 5. change the beat frequency
    def set_beat_frequency(self, beat_hz: float) -> dict[str, Any]:
        beat = self._current_beat()
        beat["beat_hz"] = beat_hz
        return self._request("PATCH", "/api/stream/spec", {"beat": beat})

    def _current_beat(self) -> dict[str, Any]:
        """The live beat as a mutable dict, or a sensible default if none is set."""
        beat = self.get_state().get("beat")
        if beat:
            return dict(beat)
        return {"carrier_hz": CARRIER_HZ, "beat_hz": 10.0, "volume": 0.3, "waveform": "sine"}
