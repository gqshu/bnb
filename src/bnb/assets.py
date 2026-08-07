"""The background-media asset repository.

A tagged store organized as a **cell tree**, one subdirectory per category cell, so
all the wav files (and specs) for a category sit together and can be browsed, bulk-
deleted, or `ls`'d as a unit. A cell is ``(style, substrate)`` for the grid taxonomy
or ``(group, keyword)`` for a special one (``bnb.background.SPECIAL_GROUPS``) —
either way, :func:`spec_cell` reads it straight off the spec. Layout::

    assets/
      specs/<cell-1>/<cell-2>/<track_id>.json    per-track metadata record (§3)
      tracks/<cell-1>/<cell-2>/<track_id>.wav    rendered audio master (git-ignored)
      catalog.json                               generated index of compact descriptors

``track_id`` stays the one flat, globally-unique identifier everywhere outside this
module (the stream engine, the API, the CLIs) — only its on-disk location is nested.
Lookups by bare track_id (:func:`find_spec`, :func:`find_track`) scan the tree; at
library scale (hundreds of tracks, not millions) that's simpler than maintaining a
separate id -> path index, and ``catalog.json`` already caches the resolved paths for
the hot selection path. The selection workflow reads only ``catalog.json``; it never
has to open every spec. Regenerate it with :func:`rebuild_catalog` after writing specs.

Every function here takes an optional ``root`` (default :data:`ASSETS_DIR`) so callers
— chiefly ``bnb.catalog.CategoryManager`` — can point a whole session at a different
asset repository, e.g. a ``tmp_path`` in tests.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
SPECS_DIR = ASSETS_DIR / "specs"
TRACKS_DIR = ASSETS_DIR / "tracks"
CATALOG_PATH = ASSETS_DIR / "catalog.json"


def specs_dir(root: Path = ASSETS_DIR) -> Path:
    return root / "specs"


def tracks_dir(root: Path = ASSETS_DIR) -> Path:
    return root / "tracks"


def catalog_path(root: Path = ASSETS_DIR) -> Path:
    return root / "catalog.json"


def spec_cell(spec: dict[str, Any]) -> tuple[str, str]:
    """The (outer, inner) cell key a spec's files nest under.

    Grid specs (``kind: "grid"``, the substrate x style taxonomy) nest under
    ``(style, substrate)``; special specs (``kind: "special"``, e.g. natural_sounds)
    nest under ``(group, keyword)``. Older grid specs with no ``kind`` key default
    to grid.
    """
    if spec.get("kind", "grid") == "grid":
        return (spec["style"], spec["substrate"])
    return (spec["group"], spec["keyword"])


def cell_dir(base: Path, spec: dict[str, Any]) -> Path:
    outer, inner = spec_cell(spec)
    return base / outer / inner


def spec_path(spec: dict[str, Any], root: Path = ASSETS_DIR) -> Path:
    """Where ``spec`` belongs on disk, given its cell. Requires the full spec (not
    just a track_id) since the cell can't be recovered from the id alone; use
    :func:`find_spec` to look an existing spec up by id."""
    return cell_dir(specs_dir(root), spec) / f"{spec['track_id']}.json"


def track_path(spec: dict[str, Any], output_format: str, root: Path = ASSETS_DIR) -> Path:
    """Where ``spec``'s rendered audio belongs on disk, given its cell."""
    return cell_dir(tracks_dir(root), spec) / f"{spec['track_id']}.{audio_extension(output_format)}"


def audio_extension(output_format: str) -> str:
    """File extension for an output format (e.g. ``pcm_44100`` -> ``wav``); ``wav`` and
    other bare extensions (Stable Audio always writes wav) pass through unchanged."""
    return "wav" if output_format.startswith("pcm") else output_format.split("_", 1)[0]


def find_spec(track_id: str, root: Path = ASSETS_DIR) -> Path | None:
    """The spec file for a track_id, wherever its cell put it, or None."""
    d = specs_dir(root)
    matches = sorted(d.rglob(f"{track_id}.json")) if d.exists() else []
    return matches[0] if matches else None


def find_track(track_id: str, root: Path = ASSETS_DIR) -> Path | None:
    """The rendered audio file for a track, whatever format it was saved in, or None."""
    d = tracks_dir(root)
    matches = sorted(d.rglob(f"{track_id}.*")) if d.exists() else []
    return matches[0] if matches else None


def has_spec(track_id: str, root: Path = ASSETS_DIR) -> bool:
    return find_spec(track_id, root) is not None


def has_track(track_id: str, root: Path = ASSETS_DIR) -> bool:
    return find_track(track_id, root) is not None


def write_spec(metadata: dict[str, Any], root: Path = ASSETS_DIR) -> Path:
    """Write a §3 metadata record to its cell directory."""
    path = spec_path(metadata, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    return path


def load_spec(track_id: str, root: Path = ASSETS_DIR) -> dict[str, Any]:
    path = find_spec(track_id, root)
    if path is None:
        raise FileNotFoundError(f"no spec for track_id {track_id!r}")
    return json.loads(path.read_text())


def list_specs(root: Path = ASSETS_DIR) -> list[str]:
    """Every track_id with a spec on disk, sorted."""
    d = specs_dir(root)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.rglob("*.json"))


def list_rendered(root: Path = ASSETS_DIR) -> list[str]:
    """Every track_id that has rendered audio on disk (i.e. is playable), sorted."""
    return [tid for tid in list_specs(root) if has_track(tid, root)]


def write_pcm_wav(path: Path, pcm: bytes, sample_rate: int, channels: int = 2) -> None:
    """Wrap raw signed-16-bit little-endian PCM in a WAV container.

    ElevenLabs ``pcm_*`` output is headerless S16LE; music is stereo, hence the
    ``channels=2`` default. Override if a first real render proves otherwise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def record_render(
    spec: dict[str, Any],
    *,
    provider: str,
    model_version: str,
    output_format: str,
    license: str,
    audio_file: str,
    generated_at: str,
    watermark: str | None = None,
) -> dict[str, Any]:
    """Fill a spec's ``render`` block after audio has been produced."""
    spec["render"] = {
        "provider": provider,
        "model_version": model_version,
        "output_format": output_format,
        "license": license,
        "generated_at": generated_at,
        "audio_file": audio_file,
        "watermark": watermark,
    }
    return spec


def _summary(spec: dict[str, Any]) -> str:
    """A one-line human-readable descriptor for the catalog."""
    instruments = ", ".join(spec.get("instrumentation", [])) or "—"
    if spec.get("kind", "grid") == "special":
        return f"{spec['group']} × {spec['keyword']} — {instruments}"
    rf = spec.get("requested_features") or {}
    bits = [b for b in (rf.get("energy"), rf.get("register"), rf.get("texture_density")) if b]
    return f"{spec['style']} × {spec['substrate']} — {instruments}; {', '.join(bits)}"


def catalog_entry(spec: dict[str, Any], root: Path = ASSETS_DIR) -> dict[str, Any]:
    """The compact, selectable descriptor for one track (what the workflow reads)."""
    audio = find_track(spec["track_id"], root)
    spec_file = find_spec(spec["track_id"], root)
    render = spec.get("render") or {}
    return {
        "track_id": spec["track_id"],
        "kind": spec.get("kind", "grid"),
        "summary": _summary(spec),
        "substrate": spec.get("substrate"),
        "style": spec.get("style"),
        "group": spec.get("group"),
        "keyword": spec.get("keyword"),
        "seed": spec.get("seed"),
        "duration_s": spec.get("duration_s"),
        "instrumentation": spec.get("instrumentation", []),
        "requested_features": spec.get("requested_features"),
        "measured_features": spec.get("measured_features"),
        "rendered": audio is not None,
        "provider": render.get("provider"),
        "spec": str(spec_file.relative_to(root)) if spec_file else None,
        "audio": str(audio.relative_to(root)) if audio else None,
    }


def rebuild_catalog(root: Path = ASSETS_DIR) -> dict[str, Any]:
    """Scan ``specs/`` and (re)write ``catalog.json``; return the catalog dict."""
    entries = [catalog_entry(load_spec(tid, root), root) for tid in list_specs(root)]
    catalog = {
        "count": len(entries),
        "tracks": entries,
    }
    root.mkdir(parents=True, exist_ok=True)
    catalog_path(root).write_text(json.dumps(catalog, indent=2) + "\n")
    return catalog
