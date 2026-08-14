"""Prepare a cloud-storage-sized copy of the app's content, for manual upload.

    uv run scripts/prepare_cloud_assets.py                 # everything, into run/cloud_assets
    uv run scripts/prepare_cloud_assets.py --limit 5       # a small batch, to eyeball first
    uv run scripts/prepare_cloud_assets.py --catalog-only  # re-emit the catalogues, skip audio

This produces everything the mini program reads at runtime, so that a session needs no
backend at all:

    <out>/
      bg/
        manifest.json            the background catalogue    (see `build_manifest`)
        profiles.json            the mode-profile catalogue  (see `build_profiles`)
        <track_id>.mp3           one file per rendered track
        profile/<id>.jpg         one card background per profile that has art

Upload ``bg/`` to the **root** of the WeChat cloud storage bucket (云开发控制台 → 存储)
and point the mini program's ``CLOUD_FILE_PREFIX`` at that root. The output directory is a
literal mirror of the bucket, so there is one thing to drag and nothing to rearrange. It
lives under ``run/`` by default, which is gitignored — these are derived artifacts,
regenerable from the sources at any time.

The rendered masters are 60 s stereo 16-bit WAVs — about 10.6 MB each, 1.2 GB for the
whole library. That's fine for a service that encodes on demand (``/background/{id}.mp3``
does exactly that, cached per track) but hopeless as something to upload by hand and pull
down over mobile data, hence the transcode.

**The manifest is the point of this script**, more than the transcoding is. Moving track
selection to the client means the client needs the taxonomy, and the taxonomy has a shape
that only makes sense inside the backend: grid tracks carry a per-track ``goal`` field
while special-group tracks carry none, and their goal compatibility lives on the *keyword*
definition instead (``background.KeywordEntry.goals``) — which is why the server needs
both `_goal_compatible_pool` and `_special_pool` to answer one question. Rather than ship
that split to the client, this resolves it here: every track comes out with a flat
``goals`` list and a flat ``type``, so the client's filter is one predicate over one array
and the grid/special distinction disappears entirely. The taxonomy stays owned by Python;
only its resolved output crosses the wire.

``profiles.json`` goes out in the same envelope ``GET /api/profiles`` serves, so the
client parses one shape either way. The only thing that changes is ``image``: the served
form is a route (``/profile/<file>``, resolved against the backend's host) and the cloud
form is a bucket-relative path (``bg/profile/<file>``) the client turns into a fileID.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import lameenc
import numpy as np
import soundfile as sf

from bnb import assets
from bnb.background import SPECIAL_GROUPS
from bnb.catalog import CategoryManager
from bnb.profiles import list_profiles, profiles_image_dir
from bnb.stream import to_int16_bytes

# The display names the app shows for a background. Imported rather than restated so the
# cloud manifest and the server's own `/api/backgrounds/random` can never disagree about
# what a track is called — it's a private helper, but it's the single source of truth for
# this mapping and a copy here would drift the first time a style is renamed.
from bnb.server import _bg_display_name

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "run" / "cloud_assets"
DEFAULT_BITRATE_KBPS = 96  # stereo; ~720 KB per 60 s track, ~15x smaller than the master
DEFAULT_QUALITY = 5  # lameenc: 0=best/slowest … 9=fastest. Matches the service's setting.
# Everything the app reads lives under one directory in the bucket, so the output tree is
# a literal mirror of what gets uploaded: drop `bg/` at the bucket root and you're done.
# Every path *written into* the catalogues stays relative to the bucket root (not to this
# directory), which is what lets the client resolve a fileID by plain concatenation with
# CLOUD_FILE_PREFIX — one rule for audio, card art, and the catalogues alike.
BUCKET_ROOT = "bg"
AUDIO_SUBDIR = BUCKET_ROOT  # the mp3s sit directly in it
PROFILE_SUBDIR = f"{BUCKET_ROOT}/profile"
MANIFEST_NAME = f"{BUCKET_ROOT}/manifest.json"
PROFILES_NAME = f"{BUCKET_ROOT}/profiles.json"
MANIFEST_VERSION = 1


def resolve_goals(entry: dict[str, Any]) -> list[str]:
    """Which goals a catalog entry is compatible with, as a flat list.

    This is the whole grid/special reconciliation (see the module docstring). A grid
    track was rendered *for* one goal and carries it per-track. A special-group track
    carries no goal at all — the same rainfall suits relaxing and masking-for-focus, so
    it's rendered once — and its compatibility is metadata on the group's keyword entry.
    Both collapse to the same list here.

    A group or keyword the taxonomy doesn't know is reported as compatible with nothing,
    so it drops out of every client pool rather than 500ing something later.
    """
    if entry.get("kind") == "special":
        group = SPECIAL_GROUPS.get(entry.get("group") or "")
        if group is None:
            return []
        keyword = group.keywords.get(entry.get("keyword") or "")
        return [] if keyword is None else sorted(keyword.goals)
    goal = entry.get("goal")
    return [goal] if goal else []


def resolve_type(entry: dict[str, Any]) -> str | None:
    """The track's filterable ``type``: its substrate (grid) or its group (special).

    Deliberately the same axis ``GET /api/backgrounds/random?type=`` accepts and the same
    one a profile's ``spec.type`` names, so a profile authored against the server keeps
    working unchanged when the client is reading from cloud storage.
    """
    return entry.get("substrate") or entry.get("group")


def manifest_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """One catalog entry reduced to what the client actually selects on."""
    track_id = entry["track_id"]
    return {
        "id": track_id,
        "file": f"{AUDIO_SUBDIR}/{track_id}.mp3",
        "name": _bg_display_name(entry),
        "goals": resolve_goals(entry),
        "type": resolve_type(entry),
    }


def build_manifest(entries: list[dict[str, Any]], bitrate_kbps: int) -> dict[str, Any]:
    """The client-facing catalogue: presentation and selection keys only.

    Everything the backend keeps for its own purposes — prompts, seeds, requested and
    measured features, provider, spec paths — stays out. The client selects on ``goals``
    and ``type``, displays ``name``, and fetches ``file``; shipping the rest would just be
    a second copy of the catalog to keep in sync.
    """
    return {
        "version": MANIFEST_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "bitrate_kbps": bitrate_kbps,
        "count": len(entries),
        "tracks": [manifest_entry(e) for e in entries],
    }


def build_profiles(out: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """The mode-profile catalogue, with card art copied into the output tree.

    Read through :func:`bnb.profiles.list_profiles`, so the same validation that guards
    ``GET /api/profiles`` guards the upload: a card with a duplicate id, too many badges,
    or a ``spec.goal``/``spec.type`` the taxonomy doesn't know fails here rather than
    shipping to a bucket where nothing checks it again.

    ``image`` is rewritten from the served route form (``/profile/<file>``, which only
    means anything relative to the backend's host) to a bucket-relative path
    (``profile/<file>``). A profile naming art that isn't on disk has its ``image``
    dropped rather than carried over broken — the client already falls back to
    ``gradient``, so the card still renders. Returns the profiles and any such misses.
    """
    profiles = list_profiles()
    missing: list[str] = []
    for profile in profiles:
        image = profile.get("image")
        if not image:
            continue
        filename = Path(image).name
        src = profiles_image_dir() / filename
        if not src.is_file():
            missing.append(f"{profile.get('id')}: {filename}")
            profile.pop("image", None)
            continue
        dst = out / PROFILE_SUBDIR / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        profile["image"] = f"{PROFILE_SUBDIR}/{filename}"
    return profiles, missing


def encode_mp3(src: Path, dst: Path, bitrate_kbps: int, quality: int) -> int:
    """Transcode one rendered WAV master to MP3. Returns the bytes written.

    Mono sources are widened to stereo rather than encoded as mono, so every file in the
    bucket has the same channel count and the client never has to special-case one.
    """
    data, sample_rate = sf.read(str(src), dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate_kbps)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(2)
    encoder.set_quality(quality)
    mp3 = bytes(encoder.encode(to_int16_bytes(data))) + bytes(encoder.flush())
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(mp3)
    return len(mp3)


def human(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024 or unit == "GB":
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} GB"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcode the rendered background library for WeChat cloud storage."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--bitrate", type=int, default=DEFAULT_BITRATE_KBPS, help="MP3 bitrate in kbps")
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY, help="lameenc quality, 0=best … 9=fastest")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N tracks (for a trial run)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-encode tracks whose MP3 already exists (default: skip them, so a "
        "interrupted run resumes instead of starting over)",
    )
    parser.add_argument(
        "--catalog-only",
        action="store_true",
        help="rewrite manifest.json and profiles.json (and re-copy card art) without "
        "touching any audio",
    )
    args = parser.parse_args()

    categories = CategoryManager()
    entries = sorted(categories.search(rendered=True), key=lambda e: e["track_id"])
    if args.limit is not None:
        entries = entries[: args.limit]
    if not entries:
        print("no rendered tracks in the catalog — nothing to prepare", file=sys.stderr)
        return 1

    out: Path = args.out
    # The catalogues live inside BUCKET_ROOT too, so create it up front — a --catalog-only
    # run into a fresh directory encodes nothing and would otherwise have nowhere to write.
    (out / BUCKET_ROOT).mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    missing: list[str] = []
    total_bytes = 0

    if not args.catalog_only:
        for i, entry in enumerate(entries, 1):
            track_id = entry["track_id"]
            dst = out / AUDIO_SUBDIR / f"{track_id}.mp3"
            if dst.exists() and not args.force:
                total_bytes += dst.stat().st_size
                skipped += 1
                continue
            src = assets.find_track(track_id)
            if src is None:
                # The catalog says rendered but the audio isn't on disk. Report it at the
                # end rather than aborting — one absent master shouldn't cost you the
                # other 116 transcodes.
                missing.append(track_id)
                continue
            size = encode_mp3(src, dst, args.bitrate, args.quality)
            total_bytes += size
            written += 1
            print(f"[{i}/{len(entries)}] {track_id}  {human(size)}")

    # The manifest lists what's actually in the bucket, so a track whose master was
    # missing is left out of it too — otherwise the client would pick an id it can't
    # download and fail at play time instead of never offering it.
    publishable = [e for e in entries if e["track_id"] not in set(missing)]
    manifest = build_manifest(publishable, args.bitrate)
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    # Profiles are validated on read, so a bad hand-edit to assets/profiles.json aborts
    # the run here — before anything reaches a bucket that won't validate it again.
    try:
        profiles, missing_art = build_profiles(out)
    except (OSError, ValueError) as exc:
        print(f"\nprofiles config error: {exc}", file=sys.stderr)
        return 1
    (out / PROFILES_NAME).write_text(
        json.dumps({"profiles": profiles}, ensure_ascii=False, indent=2)
    )

    print()
    print(f"output      {out}")
    print(f"manifest    {MANIFEST_NAME} — {len(publishable)} tracks")
    print(f"profiles    {PROFILES_NAME} — {len(profiles)} cards")
    if not args.catalog_only:
        print(f"encoded     {written} new, {skipped} already present")
        print(f"total size  {human(total_bytes)} at {args.bitrate} kbps")
    if missing:
        print(f"\nWARNING: {len(missing)} track(s) marked rendered but with no audio on disk:", file=sys.stderr)
        for track_id in missing:
            print(f"  {track_id}", file=sys.stderr)
        print("These are excluded from the manifest.", file=sys.stderr)
    if missing_art:
        print(f"\nWARNING: {len(missing_art)} profile(s) name card art that isn't on disk:", file=sys.stderr)
        for item in missing_art:
            print(f"  {item}", file=sys.stderr)
        print("Their 'image' was dropped; the client falls back to the gradient.", file=sys.stderr)

    print()
    print(f"Next: upload '{out / BUCKET_ROOT}' to the ROOT of the WeChat cloud storage")
    print(f"bucket (云开发控制台 → 存储), so it lands as '{BUCKET_ROOT}/' there. Then set")
    print("CLOUD_ENV and CLOUD_FILE_PREFIX in the mini program's services/config.ts to")
    print("the environment id and the bucket root's cloud:// prefix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
