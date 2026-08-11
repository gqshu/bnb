"""The category manager: add/delete/search/pick over the background asset repository.

``bnb.assets`` is the low-level, cell-aware file layer (spec/track paths, raw
read/write, the catalog rebuild). This is the one stateful handle callers actually
hold onto — the stream engine, the demo API, and the plan/render CLIs all go through
a :class:`CategoryManager` instead of poking ``assets`` functions directly, so the
"where does this cell live on disk" and "how do I pick a track" logic exists in one
place.

Defaults to the real ``assets/`` repo; pass ``root`` (e.g. a ``tmp_path``) to point a
manager at another one, which is what makes this testable without touching the real
asset repo.
"""

from __future__ import annotations

import json
import random
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from bnb import assets


class CategoryManager:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else assets.ASSETS_DIR

    # --- write --------------------------------------------------------------

    def add_spec(self, spec: dict[str, Any], *, rebuild: bool = True) -> Path:
        """Write a spec into its cell directory."""
        path = assets.write_spec(spec, root=self.root)
        if rebuild:
            self.rebuild()
        return path

    def add_render(
        self,
        spec: dict[str, Any],
        data: bytes,
        *,
        output_format: str,
        provider: str,
        model_version: str,
        license: str,
        generated_at: str,
        watermark: str | None = None,
        seed: int | None = None,
        qc: dict[str, Any] | None = None,
        rebuild: bool = True,
    ) -> Path:
        """Write in-memory audio ``data`` for ``spec`` into its cell, fill the spec's
        ``render`` block, and persist the spec. For a provider that returns raw bytes
        (e.g. ElevenLabs). Returns the audio path."""
        path = assets.track_path(spec, output_format, root=self.root)
        if output_format.startswith("pcm"):
            sample_rate = int(output_format.split("_", 1)[1])
            assets.write_pcm_wav(path, data, sample_rate)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self._record(
            spec,
            path,
            provider=provider,
            model_version=model_version,
            output_format=output_format,
            license=license,
            generated_at=generated_at,
            watermark=watermark,
            seed=seed,
            qc=qc,
            rebuild=rebuild,
        )
        return path

    def attach_render(
        self,
        spec: dict[str, Any],
        source: Path,
        *,
        provider: str,
        model_version: str,
        license: str,
        generated_at: str,
        output_format: str | None = None,
        watermark: str | None = None,
        seed: int | None = None,
        qc: dict[str, Any] | None = None,
        rebuild: bool = True,
    ) -> Path:
        """Move an already-written audio file (e.g. Stable Audio, which renders
        straight to a path) into ``spec``'s cell directory. Returns the new path.

        ``output_format`` defaults to the source file's bare extension; pass it
        explicitly to record a more specific requested format (e.g. ElevenLabs'
        ``pcm_44100``, which the extension alone (``wav``) can't reconstruct).
        """
        output_format = output_format or source.suffix.lstrip(".")
        path = assets.track_path(spec, output_format, root=self.root)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), path)
        self._record(
            spec,
            path,
            provider=provider,
            model_version=model_version,
            output_format=output_format,
            license=license,
            generated_at=generated_at,
            watermark=watermark,
            seed=seed,
            qc=qc,
            rebuild=rebuild,
        )
        return path

    def _record(self, spec: dict[str, Any], path: Path, *, rebuild: bool, **render_kwargs: Any) -> None:
        assets.record_render(spec, audio_file=str(path.relative_to(self.root)), **render_kwargs)
        assets.write_spec(spec, root=self.root)
        if rebuild:
            self.rebuild()

    # --- tags -----------------------------------------------------------------

    def add_tag(self, track_ids: Iterable[str], tag: str, *, rebuild: bool = True) -> dict[str, list[str]]:
        """Add ``tag`` to each track's spec. Returns each track's resulting tag list."""
        return self._edit_tags(track_ids, tag, add=True, rebuild=rebuild)

    def remove_tag(self, track_ids: Iterable[str], tag: str, *, rebuild: bool = True) -> dict[str, list[str]]:
        """Remove ``tag`` from each track's spec (a tag it doesn't carry is a no-op).
        Returns each track's resulting tag list."""
        return self._edit_tags(track_ids, tag, add=False, rebuild=rebuild)

    def _edit_tags(
        self, track_ids: Iterable[str], tag: str, *, add: bool, rebuild: bool
    ) -> dict[str, list[str]]:
        """Apply one tag edit across several tracks, then rebuild once.

        Every id is resolved to a spec *before* anything is written, so a batch naming
        one unknown track leaves the whole library untouched rather than half-tagged.
        Raises ``FileNotFoundError`` (from :func:`assets.load_spec`) for an unknown id.
        """
        specs = [assets.load_spec(track_id, root=self.root) for track_id in track_ids]
        out: dict[str, list[str]] = {}
        for spec in specs:
            tags = set(spec.get("tags") or [])
            tags.add(tag) if add else tags.discard(tag)
            assets.set_tags(spec, tags)
            assets.write_spec(spec, root=self.root)
            out[spec["track_id"]] = spec["tags"]
        if rebuild:
            self.rebuild()
        return out

    def tags(self) -> list[str]:
        """Every distinct tag in use across the library, sorted."""
        return sorted({tag for e in self.catalog()["tracks"] for tag in e.get("tags", [])})

    # --- delete ---------------------------------------------------------------

    def delete(self, track_id: str, *, rebuild: bool = True) -> bool:
        """Remove a track's spec and audio (whichever exist). Returns whether
        anything was found to remove."""
        spec_file = assets.find_spec(track_id, root=self.root)
        audio_file = assets.find_track(track_id, root=self.root)
        if spec_file is None and audio_file is None:
            return False
        for f in (spec_file, audio_file):
            if f is not None:
                f.unlink()
        if rebuild:
            self.rebuild()
        return True

    def delete_cell(self, cell: tuple[str, str]) -> int:
        """Remove every track in one cell (specs + audio); return how many were removed."""
        track_ids = {e["track_id"] for e in self.search(cell=cell)}
        for track_id in track_ids:
            self.delete(track_id, rebuild=False)
        outer, inner = cell
        for base in (assets.specs_dir(self.root), assets.tracks_dir(self.root)):
            shutil.rmtree(base / outer / inner, ignore_errors=True)
        self.rebuild()
        return len(track_ids)

    # --- read / search / pick --------------------------------------------------

    def catalog(self) -> dict[str, Any]:
        """The current catalog.json, rebuilding it first if it doesn't exist yet."""
        path = assets.catalog_path(self.root)
        if not path.exists():
            return self.rebuild()
        return json.loads(path.read_text())

    def spec_ids(self) -> set[str]:
        """Every track_id with a spec on disk *right now*.

        Disk truth rather than :meth:`catalog`, so a spec deleted by hand since the
        last rebuild counts as gone — which is what "delete a spec to replan it"
        relies on."""
        return set(assets.list_specs(self.root))

    def orphan_tracks(self) -> list[Path]:
        """Rendered audio left behind by a deleted spec (see :func:`assets.orphan_tracks`)."""
        return assets.orphan_tracks(self.root)

    @staticmethod
    def _cell_of(entry: dict[str, Any]) -> tuple[str, str]:
        if entry["kind"] == "special":
            return (entry["group"], entry["keyword"])
        return (entry["style"], entry["substrate"])

    def cells(self) -> list[tuple[str, str]]:
        """Every distinct cell with at least one spec, sorted."""
        return sorted({self._cell_of(e) for e in self.catalog()["tracks"]})

    def search(
        self,
        *,
        substrate: str | None = None,
        style: str | None = None,
        goal: str | None = None,
        group: str | None = None,
        keyword: str | None = None,
        kind: str | None = None,
        cell: tuple[str, str] | None = None,
        rendered: bool | None = None,
        provider: str | None = None,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        """Filter the catalog by any combination of tags; unset filters are ignored.

        ``goal`` only ever matches grid entries (``relax``/``focus``) — special-group
        entries carry no goal (§ :func:`bnb.assets.catalog_entry`), so a goal filter
        naturally excludes them rather than needing a ``kind`` filter alongside it.

        ``tag`` matches one hand-applied tag (§ :meth:`add_tag`); unlike the taxonomy
        fields it's a membership test, since a track carries a list of them. It's what
        makes a curated set selectable — ``pick(tag="warm-bed", rendered=True)``.
        """
        entries = self.catalog()["tracks"]
        if cell is not None:
            entries = [e for e in entries if self._cell_of(e) == tuple(cell)]
        if substrate is not None:
            entries = [e for e in entries if e["substrate"] == substrate]
        if style is not None:
            entries = [e for e in entries if e["style"] == style]
        if goal is not None:
            entries = [e for e in entries if e.get("goal") == goal]
        if group is not None:
            entries = [e for e in entries if e["group"] == group]
        if keyword is not None:
            entries = [e for e in entries if e["keyword"] == keyword]
        if kind is not None:
            entries = [e for e in entries if e["kind"] == kind]
        if rendered is not None:
            entries = [e for e in entries if e["rendered"] == rendered]
        if provider is not None:
            entries = [e for e in entries if e["provider"] == provider]
        if tag is not None:
            entries = [e for e in entries if tag in e.get("tags", [])]
        return entries

    def pick(
        self,
        *,
        exclude: str | None = None,
        strategy: Literal["random"] = "random",
        **filters: Any,
    ) -> dict[str, Any] | None:
        """Select one catalog entry matching ``filters`` (same keys as :meth:`search`).

        ``exclude`` drops one track_id from the pool first (e.g. "don't repeat the
        track that's currently playing"), falling back to the full pool if that would
        leave nothing to choose from.
        """
        pool = self.search(**filters)
        if exclude is not None:
            pool = [e for e in pool if e["track_id"] != exclude] or pool
        if not pool:
            return None
        if strategy == "random":
            return random.choice(pool)
        raise ValueError(f"unknown pick strategy {strategy!r}")

    def rebuild(self) -> dict[str, Any]:
        return assets.rebuild_catalog(self.root)
