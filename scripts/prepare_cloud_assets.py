"""Prepare a cloud-storage-sized copy of the app's content, for manual upload.

    uv run scripts/prepare_cloud_assets.py                 # everything, into run/cloud_assets
    uv run scripts/prepare_cloud_assets.py --limit 5       # a small batch, to eyeball first
    uv run scripts/prepare_cloud_assets.py --catalog-only  # re-emit the catalogues, skip audio

This produces everything the mini program reads at runtime, so that a session needs no
backend at all:

    <out>/
      mastering_report.json      what the audio pass measured and fixed (NOT uploaded)
      bg/
        manifest.json            the background catalogue    (see `build_manifest`)
        profiles.json            the mode-profile catalogue  (see `build_profiles`)
        <track_id>.mp3           one file per rendered track
        profile/<id>.jpg         one card background per profile that has art

Upload ``bg/`` to the **root** of the WeChat cloud storage bucket (云开发控制台 → 存储)
and point the mini program's ``CLOUD_FILE_PREFIX`` at that root. The output directory is a
literal mirror of the bucket, so there is one thing to drag and nothing to rearrange. It
lives under ``run/`` by default, which is gitignored — these are derived artifacts,
regenerable from the sources at any time. Only ``bg/`` gets uploaded; the report sits
outside it deliberately.

The rendered masters are 60 s stereo 16-bit WAVs — about 10.6 MB each. That's fine for a
service that encodes on demand (``/background/{id}.mp3`` does exactly that, cached per
track) but hopeless as something to upload by hand and pull down over mobile data, hence
the transcode.

**Transcoding is not the whole audio pass.** The client loops each MP3 forever
(``audio.ts``: ``src.loop = true``) with nothing crossfading the wrap, so a master that
fades out at 60 s and starts at full level lurches audibly once a minute — which is most
of this library. :mod:`bnb.mastering` fixes that (trim the fade, fold the tail back over
the head) and repairs true sample-scale clicks on the way past; this script only chooses
the settings and reports what happened. The encoder is left at its original bitrate: the
artifacts people hear are seams, not bits. Encoder *effort* is raised, though — it costs
build time rather than bytes, and a batch job has build time to spare.

**The manifest is the point of this script**, more than the transcoding is. Moving track
selection to the client means the client needs the taxonomy, and the taxonomy has a shape
that only makes sense inside the backend: grid tracks carry a per-track ``goal`` field
while special-group tracks carry none, and their goal compatibility lives on the *keyword*
definition instead (``background.KeywordEntry.goals``) — which is why the server needs
both `_goal_compatible_pool` and `_special_pool` to answer one question. Rather than ship
that split to the client, this resolves it here: every track comes out with a flat
``goals`` list and a flat ``tags`` list — its two taxonomy axes, ``[substrate, style]`` for
a grid track and ``[group, keyword]`` for a special one — so the client's filter is one
predicate over two arrays and the grid/special distinction disappears entirely. The
taxonomy stays owned by Python; only its resolved output crosses the wire.

``profiles.json`` is the authored ``assets/profiles.json``, copied out in the same
envelope ``GET /api/profiles`` serves so the client parses one shape either way. Two
things change on the way. ``image`` is rewritten from the served route form
(``/profile/<file>``, which only means anything relative to the backend's host) to a
bucket-relative path (``bg/profile/<file>``) the client turns into a fileID. And only
``source: community`` cards go: the mini program ships the personal cards (the manual
panel, the EEG-driven preset) built in, so publishing them again would duplicate them in
the grid.

The cards are read through :func:`bnb.profiles.list_profiles`, so everything that guards
the endpoint guards the upload too. One check belongs to the build alone, because it is
the only place both halves are in the same room: a card's ``spec`` names a ``goal`` and a
``soundscape`` list, and :func:`unplayable_profiles` asks the finished manifest whether any
track actually satisfies them. A card that no track can answer is a dead tile in the grid —
it looks fine right up until someone taps it.
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

from bnb import assets, mastering
from bnb.background import SPECIAL_GROUPS, soundscape_selects, soundscape_tags
from bnb.catalog import CategoryManager
from bnb.mastering import MasterReport
from bnb.profiles import list_profiles, profiles_image_dir
from bnb.stream import to_int16_bytes

# The display names the app shows for a background. Imported rather than restated so the
# cloud manifest and the server's own `/api/backgrounds/random` can never disagree about
# what a track is called — it's a private helper, but it's the single source of truth for
# this mapping and a copy here would drift the first time a style is renamed.
from bnb.server import _bg_display_name

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "run" / "cloud_assets"
DEFAULT_BITRATE_KBPS = 96  # stereo; ~720 KB per 60 s track, ~15x smaller than the master
DEFAULT_QUALITY = 2
"""lameenc effort: 0=best/slowest … 9=fastest. The service uses 5 because it encodes on
the request path; this is a batch job run once per library, so it buys the better psycho-
acoustic search for nothing that matters. Same bitrate, same file size, fewer artifacts
on the hiss-like beds (rain, stream, noise_texture) where 96 kbps is worked hardest."""

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
REPORT_NAME = "mastering_report.json"  # sibling of BUCKET_ROOT, so it is never uploaded
# 2: per-track ``type`` (one substrate-or-group string) became ``tags`` (both taxonomy
# axes), so a client can filter on style and on group keywords too. Clients written
# against v1 fall back to reading ``tags[0]`` as the old type.
MANIFEST_VERSION = 2


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




def manifest_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """One catalog entry reduced to what the client actually selects on."""
    track_id = entry["track_id"]
    return {
        "id": track_id,
        "file": f"{AUDIO_SUBDIR}/{track_id}.mp3",
        "name": _bg_display_name(entry),
        "goals": resolve_goals(entry),
        # Both axes, in the taxonomy's own vocabulary — the exact strings a profile's
        # ``spec.soundscape`` keywords are built from (§ :func:`bnb.background.soundscape_tags`).
        "tags": soundscape_tags(entry),
    }


def build_manifest(entries: list[dict[str, Any]], bitrate_kbps: int) -> dict[str, Any]:
    """The client-facing catalogue: presentation and selection keys only.

    Everything the backend keeps for its own purposes — prompts, seeds, requested and
    measured features, provider, spec paths — stays out. The client selects on ``goals``
    and ``tags``, displays ``name``, and fetches ``file``; shipping the rest would just be
    a second copy of the catalog to keep in sync.
    """
    return {
        "version": MANIFEST_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "bitrate_kbps": bitrate_kbps,
        "count": len(entries),
        "tracks": [manifest_entry(e) for e in entries],
    }


def playable_tracks(manifest: dict[str, Any]) -> list[tuple[list[str], set[str]]]:
    """Every published track as ``(tags, goals)`` — what a card's spec has to hit.

    Read off the manifest rather than the catalog so it describes what is *in the bucket*:
    a track whose master went missing was already dropped from the manifest, and so it
    can't vouch for a card here either.
    """
    return [(track["tags"], set(track["goals"])) for track in manifest["tracks"]]


def unplayable_profiles(profiles: list[dict[str, Any]], manifest: dict[str, Any]) -> list[str]:
    """Cards whose ``spec`` no published track can answer — a dead tile in the grid.

    Selection is the client's job now, and the client's whole filter is
    ``goals.includes(goal)`` plus an optional ``soundscape`` match. So a card asking for a
    combination the manifest doesn't hold isn't merely unlucky: it throws at play time
    (``cloud.ts``: "没有 goal=… soundscape=… 的背景"). Reported rather than fatal — the fix
    is usually to render the missing track, not to delete the card.

    The whole list has to miss for a card to be dead: a soundscape is a union, so one
    keyword with tracks behind it carries the card even if the others name nothing yet.
    """
    tracks = playable_tracks(manifest)
    dead: list[str] = []
    for profile in profiles:
        spec = profile.get("spec")
        if not spec:
            continue
        goal, soundscape = spec.get("goal"), spec.get("soundscape")
        ok = any(
            goal in goals and soundscape_selects(tags, soundscape) for tags, goals in tracks
        )
        if not ok:
            where = f"goal={goal}" + (f" soundscape={soundscape}" if soundscape else "")
            dead.append(f"{profile.get('id')}: no track with {where}")
    return dead


def build_profiles(out: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """The community mode-profile catalogue, with card art copied into the output tree.

    Read through :func:`bnb.profiles.list_profiles`, so the same validation that guards
    ``GET /api/profiles`` guards the upload: a card with a duplicate id, an unknown
    ``source``, a ``spec.goal``/``spec.soundscape`` the taxonomy doesn't know, a ``spec.mode``
    the app can't build, or a gradient that contradicts its goal fails here rather than
    shipping to a bucket where nothing checks it again.

    Only ``source: community`` cards are published — the mini program ships the personal
    ones built in.

    ``image`` is rewritten from the served route form (``/profile/<file>``) to a
    bucket-relative path (``profile/<file>``). A profile naming art that isn't on disk has
    its ``image`` dropped rather than carried over broken — the client already falls back
    to ``gradient``, so the card still renders. Returns the profiles and any such misses.
    """
    profiles = [p for p in list_profiles() if p.get("source") == "community"]

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


def transcode(src: Path, dst: Path, args: argparse.Namespace) -> tuple[int, MasterReport]:
    """Master one rendered WAV for looping and encode it. Returns ``(bytes, report)``.

    Mono sources are widened to stereo rather than encoded as mono, so every file in the
    bucket has the same channel count and the client never has to special-case one.
    """
    data, sample_rate = sf.read(str(src), dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    data, report = mastering.master_for_loop(
        data,
        sample_rate,
        declick=not args.no_declick,
        loop=not args.no_loop_prep,
        crossfade_s=args.crossfade,
        click_z=args.click_sensitivity,
        peak_dbfs=args.peak_dbfs,
    )
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(args.bitrate)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(2)
    encoder.set_quality(args.quality)
    mp3 = bytes(encoder.encode(to_int16_bytes(data))) + bytes(encoder.flush())
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(mp3)
    return len(mp3), report


def stray_mp3s(out: Path, published: set[str]) -> list[Path]:
    """MP3s sitting in the output tree that the manifest no longer lists.

    They matter because the output directory is what gets dragged into the bucket: a
    track dropped from the taxonomy leaves its audio behind, and every later upload
    carries the corpse along. The client never asks for them, so this is dead weight
    rather than a bug — but it is dead weight that grows.
    """
    audio_dir = out / AUDIO_SUBDIR
    if not audio_dir.is_dir():
        return []
    return sorted(p for p in audio_dir.glob("*.mp3") if p.stem not in published)


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
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY, help="lameenc effort, 0=best … 9=fastest")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N tracks (for a trial run)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-encode tracks whose MP3 already exists (default: skip them, so a "
        "interrupted run resumes instead of starting over). Needed after changing any "
        "of the audio settings below — an existing file is never re-examined.",
    )
    parser.add_argument(
        "--catalog-only",
        action="store_true",
        help="rewrite manifest.json and profiles.json (and re-copy card art) without "
        "touching any audio",
    )

    audio = parser.add_argument_group(
        "audio",
        "The mastering pass (bnb.mastering). Defaults are what the library ships with; "
        "the switches exist to A/B a suspect track, not for routine use.",
    )
    audio.add_argument(
        "--no-loop-prep",
        action="store_true",
        help="don't trim the edges or crossfade the loop seam — encode the master as-is",
    )
    audio.add_argument(
        "--crossfade",
        type=float,
        default=mastering.CROSSFADE_S,
        help=f"seconds of tail folded back over the head (default: {mastering.CROSSFADE_S:g})",
    )
    audio.add_argument("--no-declick", action="store_true", help="skip click detection and repair")
    audio.add_argument(
        "--click-sensitivity",
        type=float,
        default=mastering.CLICK_Z,
        help=f"how far above its neighbourhood an impulse must sit to count as a click "
        f"(default: {mastering.CLICK_Z:g}; lower catches more, and eventually eats real "
        f"transients like fireplace crackle)",
    )
    audio.add_argument(
        "--peak-dbfs",
        type=float,
        default=mastering.PEAK_DBFS,
        help=f"ceiling applied before encoding (default: {mastering.PEAK_DBFS:g})",
    )
    audio.add_argument(
        "--prune",
        action="store_true",
        help="delete MP3s in the output tree that the manifest no longer lists (they are "
        "only reported otherwise). Refused alongside --limit, where every unprocessed "
        "track looks stale.",
    )
    args = parser.parse_args()

    if args.prune and args.limit is not None:
        print("--prune with --limit would delete the tracks --limit skipped; refusing", file=sys.stderr)
        return 2

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
    reports: dict[str, dict[str, Any]] = {}

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
            size, report = transcode(src, dst, args)
            reports[track_id] = report.as_dict()
            total_bytes += size
            written += 1
            notes = f"  ({', '.join(report.notes)})" if report.notes else ""
            print(f"[{i}/{len(entries)}] {track_id}  {human(size)}{notes}")

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
    dead_cards = unplayable_profiles(profiles, manifest)

    if reports:
        (out / REPORT_NAME).write_text(
            json.dumps(
                {
                    "generated_at": manifest["generated_at"],
                    "settings": {
                        "bitrate_kbps": args.bitrate,
                        "quality": args.quality,
                        "declick": not args.no_declick,
                        "click_sensitivity": args.click_sensitivity,
                        "loop_prep": not args.no_loop_prep,
                        "crossfade_s": args.crossfade,
                        "peak_dbfs": args.peak_dbfs,
                    },
                    "tracks": reports,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    strays = stray_mp3s(out, {e["track_id"] for e in publishable})
    if strays and args.prune:
        for path in strays:
            path.unlink()

    print()
    print(f"output      {out}")
    print(f"manifest    {MANIFEST_NAME} — {len(publishable)} tracks")
    print(f"profiles    {PROFILES_NAME} — {len(profiles)} community cards")
    if not args.catalog_only:
        print(f"encoded     {written} new, {skipped} already present")
        print(f"total size  {human(total_bytes)} at {args.bitrate} kbps")
    if reports:
        repaired = sum(r["clicks_repaired"] for r in reports.values())
        touched = sum(1 for r in reports.values() if r["clicks_repaired"])
        impulsive = [t for t, r in reports.items() if r["clicks_impulsive"]]
        trimmed = sum(r["trimmed_head_s"] + r["trimmed_tail_s"] for r in reports.values())
        print(f"mastering   {repaired} click(s) repaired across {touched} track(s); "
              f"{trimmed:.0f}s of silence and fade trimmed")
        if impulsive:
            # Not a warning: these are rain, fire and running water. Naming them is how
            # you'd notice a track that landed here by mistake.
            print(f"            {len(impulsive)} track(s) read as impulsive content, "
                  f"declick skipped: {', '.join(sorted(impulsive)[:3])}"
                  + (f" (+{len(impulsive) - 3} more)" if len(impulsive) > 3 else ""))
        print(f"            per-track detail in {REPORT_NAME} (not uploaded)")
    if skipped and not args.force:
        print("            (skipped tracks kept whatever settings they were encoded with; "
              "use --force after changing any audio option)")
    if strays:
        verb = "deleted" if args.prune else "left in place"
        print(f"stale       {len(strays)} MP3(s) not in the manifest, {verb}"
              + ("" if args.prune else " — pass --prune to remove them"))
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
    if dead_cards:
        print(f"\nWARNING: {len(dead_cards)} profile(s) name a combination no track can play:", file=sys.stderr)
        for item in dead_cards:
            print(f"  {item}", file=sys.stderr)
        print("They ship anyway, but the grid tile throws when tapped — render the "
              "missing track or retarget the card.", file=sys.stderr)

    print()
    print(f"Next: upload '{out / BUCKET_ROOT}' to the ROOT of the WeChat cloud storage")
    print(f"bucket (云开发控制台 → 存储), so it lands as '{BUCKET_ROOT}/' there. Then set")
    print("CLOUD_ENV and CLOUD_FILE_PREFIX in the mini program's services/config.ts to")
    print("the environment id and the bucket root's cloud:// prefix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
