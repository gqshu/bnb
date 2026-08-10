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

``catalog.json`` is *derived* state, never authored: the spec tree is the source of
truth, so removing a spec file or a whole cell directory by hand is the supported way
to drop tracks, and the next rebuild simply won't see them.

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


def spec_files(root: Path = ASSETS_DIR) -> list[Path]:
    """Every spec file under ``specs/``, wherever in the cell tree it sits, sorted."""
    d = specs_dir(root)
    return sorted(d.rglob("*.json")) if d.exists() else []


def track_files(root: Path = ASSETS_DIR) -> dict[str, Path]:
    """Every rendered audio file under ``tracks/``, keyed by track_id.

    One scan, so the catalog rebuild resolves audio for the whole library without
    an rglob per track. Ties (same id, two formats) resolve like :func:`find_track`:
    first in sorted order wins.
    """
    d = tracks_dir(root)
    index: dict[str, Path] = {}
    if d.exists():
        for path in sorted(d.rglob("*.*")):
            index.setdefault(path.stem, path)
    return index


def load_all_specs(root: Path = ASSETS_DIR) -> dict[str, tuple[Path, dict[str, Any]]]:
    """Every spec on disk, keyed by track_id — the single scan the catalog is built from.

    Raises ``ValueError`` on anything that would make the catalog disagree with the
    disk rather than quietly picking a winner: an unreadable or track_id-less record,
    or two files claiming the same track_id.
    """
    specs: dict[str, tuple[Path, dict[str, Any]]] = {}
    problems: list[str] = []
    for path in spec_files(root):
        rel = path.relative_to(root)
        try:
            spec = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            problems.append(f"{rel}: not a readable spec ({exc})")
            continue
        track_id = spec.get("track_id") if isinstance(spec, dict) else None
        if not track_id:
            problems.append(f"{rel}: no track_id — not a spec; move or delete it")
            continue
        if track_id in specs:
            problems.append(f"{rel}: duplicate track_id {track_id!r} (also {specs[track_id][0].relative_to(root)}); delete one")
            continue
        specs[track_id] = (path, spec)
    if problems:
        raise ValueError("inconsistent spec tree:\n  " + "\n  ".join(problems))
    return specs


def list_specs(root: Path = ASSETS_DIR) -> list[str]:
    """Every track_id with a spec on disk, sorted."""
    return sorted(load_all_specs(root))


def list_rendered(root: Path = ASSETS_DIR) -> list[str]:
    """Every track_id that has rendered audio on disk (i.e. is playable), sorted."""
    tracks = track_files(root)
    return [tid for tid in list_specs(root) if tid in tracks]


def orphan_tracks(root: Path = ASSETS_DIR) -> list[Path]:
    """Rendered audio whose spec is gone — the residue of a manual spec delete.

    Invisible to the catalog (which is spec-derived), but worth surfacing: seeds are
    deterministic, so replanning the same cell regenerates the *same* track_id and
    would adopt the stale audio as its render.
    """
    specs = set(load_all_specs(root))
    return [path for tid, path in sorted(track_files(root).items()) if tid not in specs]


def normalize_layout(root: Path = ASSETS_DIR) -> list[tuple[Path, Path]]:
    """Move every spec that isn't at :func:`spec_path` into its cell directory.

    Covers both ways a spec can drift from the layout — a stale directory (e.g. the
    old flat ``specs/<track_id>.json``) and a filename that isn't ``<track_id>.json``
    — since the canonical path fixes both. Returns the (from, to) pairs moved.
    """
    moved: list[tuple[Path, Path]] = []
    for track_id, (path, spec) in load_all_specs(root).items():
        target = spec_path(spec, root)
        if path == target:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)
        moved.append((path, target))
    prune_empty_dirs(specs_dir(root))
    return moved


def prune_empty_dirs(base: Path) -> None:
    """Remove directories under ``base`` that hold no files at any depth.

    Deleting specs (a file, or a whole cell) is the supported way to drop tracks, so
    the empty cell directories left behind are expected debris, not state."""
    if not base.exists():
        return
    for d in sorted(base.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()


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
    seed: int | None = None,
    qc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fill a spec's ``render`` block after audio has been produced.

    ``seed`` is the seed that actually produced this audio. It is normally the spec's
    own, but a render retried past a QC failure uses a fresh one, and then the spec's
    seed no longer describes the file on disk — so provenance records what was used.
    ``qc`` carries the integrity verdict that let the render through (:mod:`bnb.qc`).
    """
    spec["render"] = {
        "provider": provider,
        "model_version": model_version,
        "output_format": output_format,
        "license": license,
        "generated_at": generated_at,
        "audio_file": audio_file,
        "watermark": watermark,
        "seed": spec.get("seed") if seed is None else seed,
        "qc": qc,
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
    return _entry(spec, find_spec(spec["track_id"], root), find_track(spec["track_id"], root), root)


def _entry(
    spec: dict[str, Any], spec_file: Path | None, audio: Path | None, root: Path
) -> dict[str, Any]:
    """:func:`catalog_entry` with both file lookups already resolved — what the
    rebuild uses, so a whole-library scan costs two directory walks, not two per track."""
    render = spec.get("render") or {}
    return {
        "track_id": spec["track_id"],
        "kind": spec.get("kind", "grid"),
        "summary": _summary(spec),
        "substrate": spec.get("substrate"),
        "style": spec.get("style"),
        "goal": spec.get("goal", "relax") if spec.get("kind", "grid") == "grid" else spec.get("goal"),
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
    """Scan ``specs/`` and (re)write ``catalog.json``; return the catalog dict.

    The catalog is a pure function of the spec tree: whatever is on disk at this
    moment is exactly what lands in it, so deleting a spec file (or a whole cell
    directory) is all it takes to drop a track — nothing else caches the fact that it
    existed. The scan first normalizes the tree (:func:`normalize_layout`) and refuses
    to index a state it can't read unambiguously (:func:`load_all_specs` raises), so a
    stale catalog can't outlive the specs it describes.
    """
    normalize_layout(root)
    tracks = track_files(root)
    entries = [
        _entry(spec, path, tracks.get(track_id), root)
        for track_id, (path, spec) in sorted(load_all_specs(root).items())
    ]
    catalog = {
        "count": len(entries),
        "tracks": entries,
    }
    root.mkdir(parents=True, exist_ok=True)
    catalog_path(root).write_text(json.dumps(catalog, indent=2) + "\n")
    return catalog
