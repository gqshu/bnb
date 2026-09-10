"""Fetch audio for the ``master`` group's download keywords (bnb.background).

A keyword whose :class:`~bnb.background.KeywordEntry` sets ``download`` skips the
generative render path entirely — ``scripts/render_background.py`` calls
:func:`fetch` here instead of a provider. This is the "download" half of that split;
the "prompt" half (spiegel, glass, clayderman) goes through the
existing generative pipeline unchanged.

Two different fetch mechanics, one per download keyword, because the two sources
work differently:

* ``goldberg`` is a public archive.org item — a plain, unauthenticated
  download. Movements are cached under :data:`DOWNLOAD_CACHE_DIR` (source_url +
  identifier) so a re-render (or a retry after a partial failure) doesn't re-pull
  ~80 MB of MP3s it already has.
* ``gymnopedies`` sits behind a Musopen login, so nothing here can fetch it
  unattended (:attr:`DownloadSource.manual`). Instead it expects the file(s) staged
  by hand under :data:`MANUAL_SOURCES_DIR`, and only assembles them.

Both return one continuous audio file (they're one-track-per-keyword, not
one-track-per-movement — see the module's own design discussion), left for the
caller to run through ``bnb.qc`` and hand to ``CategoryManager.attach_render`` exactly
like a generative render's output.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

import httpx
import numpy as np
import soundfile as sf

from bnb.assets import ASSETS_DIR

MANUAL_SOURCES_DIR = ASSETS_DIR / "manual_sources"
"""Where a source that needs a logged-in download (:attr:`DownloadSource.manual`) is
staged by hand before a render can pick it up. Under ``assets/``, so it's git-ignored
like everything else in the asset repository (§ ``bnb.assets`` module docstring)."""

DOWNLOAD_CACHE_DIR = ASSETS_DIR / "download_cache"
"""Raw per-movement downloads, kept across runs so replanning/re-rendering a keyword
doesn't re-fetch its whole source every time. Derived, like ``tracks/`` — safe to
delete to force a fresh download."""


def _download(url: str, dest: Path, *, timeout: float = 120.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)
    tmp.rename(dest)


def _concat(paths: list[Path], out_path: Path) -> Path:
    """Concatenate audio files back-to-back into one wav.

    Every input must share a sample rate and channel count. A mismatch is a real
    source-quality problem (movements ripped at different settings) worth surfacing,
    not silently resampling around.
    """
    arrays = []
    sample_rate: int | None = None
    channels: int | None = None
    for path in paths:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if sample_rate is None:
            sample_rate, channels = sr, data.shape[1]
        elif sr != sample_rate or data.shape[1] != channels:
            raise ValueError(
                f"{path.name}: {sr}Hz/{data.shape[1]}ch, expected {sample_rate}Hz/"
                f"{channels}ch (from {paths[0].name}) — re-export the movements at a "
                f"matching sample rate/channel count"
            )
        arrays.append(data)
    combined = np.concatenate(arrays, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), combined, sample_rate)
    return out_path


def _archive_org_identifier(source_url: str) -> str:
    return source_url.rstrip("/").rsplit("/", 1)[-1]


def _archive_org_movement_files(identifier: str) -> list[str]:
    """The item's individual movement filenames, alphabetical (which is track order
    for a release numbered "01", "02", ... "30"), from the VBR MP3 file set every
    archive.org audio item carries."""
    resp = httpx.get(f"https://archive.org/metadata/{identifier}", timeout=30.0)
    resp.raise_for_status()
    files = resp.json().get("files", [])
    names = sorted(f["name"] for f in files if f.get("format") == "VBR MP3")
    if not names:
        raise RuntimeError(f"no VBR MP3 files found in archive.org item {identifier!r}")
    return names


def fetch_goldberg(spec: dict[str, Any], scratch_dir: Path) -> Path:
    """Download every movement of the Open Goldberg Variations from archive.org and
    concatenate them into one continuous recording of the complete work."""
    download = spec["download"]
    identifier = _archive_org_identifier(download["source_url"])
    filenames = _archive_org_movement_files(identifier)

    movement_dir = DOWNLOAD_CACHE_DIR / identifier
    movement_paths = []
    for name in filenames:
        dest = movement_dir / name
        if not dest.exists():
            _download(f"https://archive.org/download/{identifier}/{name}", dest)
        movement_paths.append(dest)

    return _concat(movement_paths, scratch_dir / f"{spec['track_id']}.wav")


def fetch_gymnopedies(spec: dict[str, Any], scratch_dir: Path) -> Path:
    """Assemble the 3 Gymnopedies from a manually-staged Musopen download.

    Musopen requires a logged-in account, so this can't be fetched with a plain HTTP
    GET (see ``DownloadSource.manual``) — the caller has to place the file(s) at
    ``MANUAL_SOURCES_DIR / "<track_id>*"`` first: either one combined file, or one per
    movement (concatenated here in filename order).
    """
    staged = sorted(
        p for p in MANUAL_SOURCES_DIR.glob(f"{spec['track_id']}*") if p.is_file()
    )
    if not staged:
        raise RuntimeError(
            f"no manually-staged source for {spec['track_id']!r}. Download the 3 "
            f"Gymnopedies from {spec['download']['source_url']} (a free Musopen "
            f"account is required), then place the file at "
            f"{MANUAL_SOURCES_DIR / spec['track_id']}.<ext> — or one file per "
            f"movement (e.g. '..._1.mp3', '..._2.mp3', '..._3.mp3'), concatenated in "
            f"filename order"
        )
    if len(staged) == 1:
        dest = scratch_dir / f"{spec['track_id']}{staged[0].suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged[0], dest)
        return dest
    return _concat(staged, scratch_dir / f"{spec['track_id']}.wav")


DOWNLOADERS: dict[str, Callable[[dict[str, Any], Path], Path]] = {
    "goldberg": fetch_goldberg,
    "gymnopedies": fetch_gymnopedies,
}


def fetch(spec: dict[str, Any], scratch_dir: Path) -> Path:
    """Produce the audio for one ``render_method: "download"`` spec, dispatched on its
    keyword. Raises ``RuntimeError`` (a missing staged file, an empty archive.org
    listing, a sample-rate mismatch) or an ``httpx`` error on network failure —
    callers treat either the way a failed generative render is treated."""
    keyword = spec["keyword"]
    downloader = DOWNLOADERS.get(keyword)
    if downloader is None:
        raise RuntimeError(f"no downloader registered for master keyword {keyword!r}")
    return downloader(spec, scratch_dir)
