"""The background-media asset repository.

A flat, tagged store — not a category tree — because the consumer (the contextual
bandit, docs/background_music.md §5) selects on categorical tags (substrate, style)
plus a continuous MER vector, which a directory hierarchy can't express. Layout::

    assets/
      specs/<track_id>.json    per-track metadata record (§3); the source of truth
      tracks/<track_id>.wav    the rendered audio master (git-ignored: large, costs credits)
      catalog.json             generated index of compact descriptors for selection

The selection workflow reads only ``catalog.json``; it never has to open every spec.
Regenerate the catalog with :func:`rebuild_catalog` after writing specs.
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


def spec_path(track_id: str) -> Path:
    return SPECS_DIR / f"{track_id}.json"


def track_path(track_id: str, output_format: str) -> Path:
    return TRACKS_DIR / f"{track_id}.{audio_extension(output_format)}"


def audio_extension(output_format: str) -> str:
    """File extension for an ElevenLabs ``output_format`` (e.g. ``pcm_44100`` -> ``wav``)."""
    return "wav" if output_format.startswith("pcm") else output_format.split("_", 1)[0]


def find_track(track_id: str) -> Path | None:
    """The rendered audio file for a track, whatever format it was saved in, or None."""
    matches = sorted(TRACKS_DIR.glob(f"{track_id}.*")) if TRACKS_DIR.exists() else []
    return matches[0] if matches else None


def has_spec(track_id: str) -> bool:
    return spec_path(track_id).exists()


def has_track(track_id: str) -> bool:
    return find_track(track_id) is not None


def write_spec(metadata: dict[str, Any]) -> Path:
    """Write a §3 metadata record to ``specs/<track_id>.json``."""
    SPECS_DIR.mkdir(parents=True, exist_ok=True)
    path = spec_path(metadata["track_id"])
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    return path


def load_spec(track_id: str) -> dict[str, Any]:
    return json.loads(spec_path(track_id).read_text())


def list_specs() -> list[str]:
    """Every track_id with a spec on disk, sorted."""
    if not SPECS_DIR.exists():
        return []
    return sorted(p.stem for p in SPECS_DIR.glob("*.json"))


def list_rendered() -> list[str]:
    """Every track_id that has rendered audio on disk (i.e. is playable), sorted."""
    return [tid for tid in list_specs() if has_track(tid)]


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
    rf = spec.get("requested_features", {})
    instruments = ", ".join(spec.get("instrumentation", [])) or "—"
    bits = [b for b in (rf.get("energy"), rf.get("register"), rf.get("texture_density")) if b]
    return f"{spec['style']} × {spec['substrate']} — {instruments}; {', '.join(bits)}"


def catalog_entry(spec: dict[str, Any]) -> dict[str, Any]:
    """The compact, selectable descriptor for one track (what the workflow reads)."""
    audio = find_track(spec["track_id"])
    render = spec.get("render") or {}
    return {
        "track_id": spec["track_id"],
        "summary": _summary(spec),
        "substrate": spec["substrate"],
        "style": spec["style"],
        "seed": spec.get("seed"),
        "duration_s": spec.get("duration_s"),
        "instrumentation": spec.get("instrumentation", []),
        "requested_features": spec.get("requested_features", {}),
        "measured_features": spec.get("measured_features"),
        "rendered": audio is not None,
        "provider": render.get("provider"),
        "spec": f"specs/{spec['track_id']}.json",
        "audio": f"tracks/{audio.name}" if audio else None,
    }


def rebuild_catalog() -> dict[str, Any]:
    """Scan ``specs/`` and (re)write ``catalog.json``; return the catalog dict."""
    entries = [catalog_entry(load_spec(tid)) for tid in list_specs()]
    catalog = {
        "count": len(entries),
        "tracks": entries,
    }
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2) + "\n")
    return catalog
