"""Render background specs into audio via a provider (paid or, later, local).

Reads render-independent specs written by scripts/plan_background.py and turns
them into audio masters, filling each spec's ``render`` block. Rendering is the
paid / heavyweight half, kept separate so specs can be previewed for free first.

    export ELEVENLABS_API_KEY=...

    uv run scripts/render_background.py                  # render every spec still missing audio
    uv run scripts/render_background.py --dry-run        # list what would render, no API call
    uv run scripts/render_background.py buddhist_meditative_drone_seed81657   # render specific track(s)
    uv run scripts/render_background.py --force buddhist_meditative_drone_seed81657   # re-render existing
    uv run scripts/render_background.py --output-format mp3_44100_192 --model-id music_v2

Default is credit-safe and idempotent: specs that already have audio are skipped
unless you pass --force. Provider is pluggable (--provider); only ElevenLabs is
wired today, but the same specs can drive a self-hosted model later.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from typing import Any, Callable

from bnb import assets

DEFAULT_OUTPUT_FORMAT = "pcm_44100"  # -> WAV master (§2); mp3_*/opus_* also accepted
DEFAULT_MODEL_ID = "music_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "track_ids",
        nargs="*",
        help="specific specs to render (default: every spec missing audio)",
    )
    parser.add_argument("--provider", default="elevenlabs", choices=sorted(PROVIDERS))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-format", default=DEFAULT_OUTPUT_FORMAT)
    parser.add_argument("--force", action="store_true", help="re-render audio even if the track already has it")
    parser.add_argument("--dry-run", action="store_true", help="list what would render, skip the API call")
    return parser.parse_args()


def render_elevenlabs(spec: dict[str, Any], *, output_format: str, model_id: str) -> bytes:
    """Call Eleven Music with the spec's composition plan and seed.

    ``force_instrumental`` is *not* passed: the API rejects it alongside a
    composition plan (422), and the plan's empty section ``lines`` already force a
    wordless, instrumental render (guardrail §6).
    """
    from elevenlabs import ElevenLabs

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise SystemExit("ELEVENLABS_API_KEY is not set")

    client = ElevenLabs(api_key=api_key)
    stream = client.music.compose(
        composition_plan=spec["composition_plan"],
        model_id=model_id,
        output_format=output_format,
        seed=spec["seed"],
    )
    return stream if isinstance(stream, (bytes, bytearray)) else b"".join(stream)


# provider name -> (render callable, license string for the render block)
PROVIDERS: dict[str, tuple[Callable[..., bytes], str]] = {
    "elevenlabs": (render_elevenlabs, "elevenlabs-eleven-music"),
}


def write_audio(track_id: str, data: bytes, output_format: str) -> str:
    """Write the audio master; return its repo-relative path."""
    path = assets.track_path(track_id, output_format)
    if output_format.startswith("pcm"):
        sample_rate = int(output_format.split("_", 1)[1])
        assets.write_pcm_wav(path, bytes(data), sample_rate)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(data))
    return f"tracks/{path.name}"


def target_ids(args: argparse.Namespace) -> list[str]:
    available = set(assets.list_specs())
    if not available:
        raise SystemExit("no specs to render; run scripts/plan_background.py first")
    if args.track_ids:
        missing = [t for t in args.track_ids if t not in available]
        if missing:
            raise SystemExit(f"no spec for: {', '.join(missing)}")
        return args.track_ids
    return assets.list_specs()


def main() -> None:
    args = parse_args()
    render, license = PROVIDERS[args.provider]

    rendered = skipped = planned = 0
    for track_id in target_ids(args):
        if assets.has_track(track_id) and not args.force:
            skipped += 1
            print(f"skip (audio)   {track_id}")
            continue

        spec = assets.load_spec(track_id)
        if args.dry_run:
            planned += 1
            print(f"planned        {track_id}  ({spec['substrate']} x {spec['style']})")
            continue

        data = render(spec, output_format=args.output_format, model_id=args.model_id)
        audio_file = write_audio(track_id, data, args.output_format)
        assets.record_render(
            spec,
            provider=args.provider,
            model_version=args.model_id,
            output_format=args.output_format,
            license=license,
            audio_file=audio_file,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        assets.write_spec(spec)
        rendered += 1
        print(f"rendered       {track_id}  -> {audio_file}")

    catalog = assets.rebuild_catalog()
    if args.dry_run:
        print(f"\ndry run: {planned} would render, {skipped} skipped; catalog: {catalog['count']} tracks")
    else:
        print(f"\n{rendered} rendered, {skipped} skipped; catalog: {catalog['count']} tracks")


if __name__ == "__main__":
    main()
