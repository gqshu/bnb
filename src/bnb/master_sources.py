"""Fetch audio for the ``master`` group's download keywords (bnb.background).

A keyword whose :class:`~bnb.background.KeywordEntry` sets ``download`` skips the
generative render path entirely — ``scripts/render_background.py`` calls
:func:`fetch` here instead of a provider. This is the "download" half of that split;
the "prompt" half (spiegel, glass, clayderman) goes through the
existing generative pipeline unchanged.

Both download keywords are ``manual=True`` — every candidate recording is whatever
gets staged by hand under :func:`manual_source_dir`, resolved by index
(:func:`staged_sources`, :func:`_fetch_manual_variant`): more than one staged
recording means more than one playable variant, not one recording split into movements
(a recording that legitimately arrives as several movement files gets its own
subfolder instead — see :func:`staged_sources`). They differ in what happens on a
miss:

* ``gymnopedies`` sits behind a Musopen login, so nothing here can fetch it
  unattended (:attr:`DownloadSource.manual`) — a variant with nothing staged simply
  fails, with instructions.
* ``goldberg`` additionally falls back, for variant 0 only, to downloading the
  complete performance from its one public archive.org item (a plain, unauthenticated
  download — movements cached under :data:`DOWNLOAD_CACHE_DIR` so a re-render doesn't
  re-pull ~80 MB of MP3s it already has). Variant 1+ has no such fallback: there's
  only the one archive.org item, so a second variant can only come from staging
  another recording.

Every fetch here returns one continuous audio file (one per *variant*, not
necessarily one per keyword), left for the caller to run through ``bnb.qc`` and hand
to ``CategoryManager.attach_render`` exactly like a generative render's output.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

import httpx
import numpy as np
import soundfile as sf

from bnb import qc
from bnb.assets import ASSETS_DIR

MANUAL_SOURCES_DIR = ASSETS_DIR / "manual_sources"
"""Where a source that needs a logged-in download (:attr:`DownloadSource.manual`) is
staged by hand before a render can pick it up. Under ``assets/``, so it's git-ignored
like everything else in the asset repository (§ ``bnb.assets`` module docstring)."""


def manual_source_dir(group: str, keyword: str) -> Path:
    """Where a download keyword's manually-staged candidates belong: one directory per
    (group, keyword) cell, same nesting as ``assets/specs/`` and ``assets/tracks/``
    (``bnb.assets.cell_dir``) — the cell is the natural key, not a track_id/seed, which
    the person staging a file has no reason to know or keep in sync."""
    return MANUAL_SOURCES_DIR / group / keyword


def staged_sources(group: str, keyword: str) -> list[Path]:
    """Every manually-staged *candidate recording* for one cell, sorted by name.

    Each entry is one independent take, indexed by a spec's ``variant`` (0 = the first
    one here, 1 = the second, ...) — see :func:`fetch_gymnopedies`. An entry is either
    a single audio file (used as-is) or a subfolder (its contents concatenated, for the
    one case a recording legitimately arrives as several movement files rather than
    one — see :func:`_resolve_source`). Three loose files staged side by side are
    therefore three *different* recordings, not three movements of one; put movement
    files in their own subfolder if that's what they are.

    Filtered to known audio extensions (``bnb.qc.AUDIO_SUFFIXES``) plus any directory,
    so a stray README/.DS_Store dropped in the same folder isn't picked up as a candidate.
    """
    cell_dir = manual_source_dir(group, keyword)
    if not cell_dir.is_dir():
        return []
    return sorted(
        p for p in cell_dir.iterdir()
        if p.is_dir() or (p.is_file() and p.suffix.lower() in qc.AUDIO_SUFFIXES)
    )


def _resolve_source(entry: Path) -> list[Path]:
    """One staged candidate's underlying audio file(s), in the order :func:`_concat`
    should join them: the file itself, or every audio file inside it (sorted) if it's
    a per-movement subfolder."""
    if entry.is_file():
        return [entry]
    return sorted(
        p for p in entry.iterdir() if p.is_file() and p.suffix.lower() in qc.AUDIO_SUFFIXES
    )


def _fetch_manual_variant(spec: dict[str, Any], scratch_dir: Path) -> Path | None:
    """Resolve ``spec``'s variant from what's manually staged for its cell, or
    ``None`` if nothing is staged at that index yet — shared by every ``manual=True``
    download keyword (:func:`fetch_gymnopedies`, :func:`fetch_goldberg`), which differ
    only in what they do about a miss: gymnopedies has nothing else to try, goldberg
    falls back to archive.org.
    """
    variant = spec.get("variant", 0)
    sources = staged_sources(spec["group"], spec["keyword"])
    if variant >= len(sources):
        return None
    files = _resolve_source(sources[variant])
    if len(files) == 1:
        dest = scratch_dir / f"{spec['track_id']}{files[0].suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(files[0], dest)
        return dest
    return _concat(files, scratch_dir / f"{spec['track_id']}.wav")


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


def _fetch_full_archive_org_recording(spec: dict[str, Any], scratch_dir: Path) -> Path:
    """Download every movement of the Open Goldberg Variations from archive.org and
    concatenate them into one continuous recording of the complete work — the
    variant-0 fallback :func:`fetch_goldberg` uses when nothing has been staged."""
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


def fetch_goldberg(spec: dict[str, Any], scratch_dir: Path) -> Path:
    """Prefer a manually-staged recording (any variant — short remixes and alternate
    performances of individual variations are exactly what turned up staged in
    practice), falling back to the complete archive.org performance only for variant 0
    and only when nothing has been staged: there's just the one archive.org item, so
    it can't answer for variant 1+ the way a further staged recording can.
    """
    resolved = _fetch_manual_variant(spec, scratch_dir)
    if resolved is not None:
        return resolved
    variant = spec.get("variant", 0)
    if variant != 0:
        cell_dir = manual_source_dir(spec["group"], spec["keyword"])
        sources = staged_sources(spec["group"], spec["keyword"])
        raise RuntimeError(
            f"no manually-staged source for {spec['group']}:{spec['keyword']} variant "
            f"{variant} ({len(sources)} staged so far), and archive.org only has the "
            f"one complete performance (that's variant 0's fallback, not variant "
            f"{variant}'s). Stage another recording at {cell_dir}/"
        )
    return _fetch_full_archive_org_recording(spec, scratch_dir)


def fetch_gymnopedies(spec: dict[str, Any], scratch_dir: Path) -> Path:
    """Resolve one manually-staged recording of the 3 Gymnopedies.

    Musopen requires a logged-in account, so this can't be fetched with a plain HTTP
    GET (see ``DownloadSource.manual``) — the caller has to stage it by hand under
    ``manual_source_dir(spec["group"], spec["keyword"])`` first. ``spec["variant"]``
    (:attr:`~bnb.background.KeywordSignature.variant`) picks which staged candidate
    this spec resolves to (:func:`staged_sources`, via :func:`_fetch_manual_variant`) —
    several independently-staged recordings are several variants of the keyword, not
    movements of one, which is why this indexes rather than concatenates everything found.
    """
    resolved = _fetch_manual_variant(spec, scratch_dir)
    if resolved is not None:
        return resolved
    variant = spec.get("variant", 0)
    cell_dir = manual_source_dir(spec["group"], spec["keyword"])
    sources = staged_sources(spec["group"], spec["keyword"])
    raise RuntimeError(
        f"no manually-staged source for {spec['group']}:{spec['keyword']} variant "
        f"{variant} ({len(sources)} staged so far). Download another recording of "
        f"the 3 Gymnopedies from {spec['download']['source_url']} (a free Musopen "
        f"account is required), then place it at {cell_dir}/ — a single file for "
        f"one variant, or a subfolder of per-movement files for one variant "
        f"assembled from several"
    )


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
