"""Generate sample background-soundscape tracks via ElevenLabs Eleven Music.

Renders the down-regulation media library described in docs/background_music.md.
Each track is a (substrate, style) signature -> one audio file plus a JSON
sidecar carrying the §3 metadata record. Output lands in run/background/
(git-ignored) so you can put headphones on and ear-check the beds.

    export ELEVENLABS_API_KEY=...

    uv run --extra media scripts/generate_background.py                 # curated sample set, 60 s each
    uv run --extra media scripts/generate_background.py --list          # print the catalog, render nothing
    uv run --extra media scripts/generate_background.py --dry-run       # write prompts/metadata, no API call
    uv run --extra media scripts/generate_background.py buddhist_meditative:drone neutral:noise_texture
    uv run --extra media scripts/generate_background.py --duration 90 --output-format mp3_44100_192

These samples are for the Stage 1 provider bake-off, not the shipped library:
the WAV master + objective-feature extraction pipeline (measured MER) lands in
Stage 2. ElevenLabs is the structured-control alternative to self-hosted Stable
Audio; the composition-plan surface is what we exercise here.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from bnb.background import (
    SAMPLE_PAIRS,
    STYLES,
    SUBSTRATES,
    Signature,
    build_signature,
    sample_signatures,
)

RUN_DIR = Path(__file__).resolve().parent.parent / "run" / "background"

PROVIDER = "elevenlabs-eleven-music"
LICENSE = "elevenlabs-eleven-music"  # per ElevenLabs terms; not the Stable Audio community-owned tier
MODEL_ID = "music_v1"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_192"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "pairs",
        nargs="*",
        metavar="STYLE:SUBSTRATE",
        help="pairs to render (default: the curated sample set)",
    )
    parser.add_argument("--duration", type=int, default=60, help="seconds per track (default: 60)")
    parser.add_argument("--out-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-format", default=DEFAULT_OUTPUT_FORMAT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write the prompt/metadata JSON only, skip the API call",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the substrate/style catalog and the default sample set, then exit",
    )
    return parser.parse_args()


def resolve_signatures(pairs: list[str], duration_s: int) -> list[Signature]:
    """Turn ``style:substrate`` CLI args into signatures (or the default set)."""
    if not pairs:
        return sample_signatures(duration_s)
    signatures = []
    for pair in pairs:
        if ":" not in pair:
            raise SystemExit(f"expected STYLE:SUBSTRATE, got {pair!r}")
        style, substrate = pair.split(":", 1)
        signatures.append(build_signature(substrate, style, duration_s))
    return signatures


def print_catalog() -> None:
    print("Substrates (Axis A):")
    for name in SUBSTRATES:
        print(f"  {name}")
    print("\nStyles (Axis B):")
    for name in STYLES:
        print(f"  {name}")
    print("\nDefault sample set (style:substrate):")
    for style, substrate in SAMPLE_PAIRS:
        print(f"  {style}:{substrate}")


def audio_extension(output_format: str) -> str:
    """File extension for an ElevenLabs output_format string (e.g. mp3_44100_192)."""
    return "wav" if output_format.startswith("pcm") else output_format.split("_", 1)[0]


def render_audio(sig: Signature, *, model_id: str, output_format: str) -> bytes:
    """Call Eleven Music with the signature's composition plan and return audio bytes."""
    from elevenlabs import ElevenLabs

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise SystemExit("ELEVENLABS_API_KEY is not set")

    client = ElevenLabs(api_key=api_key)
    # The composition plan fixes the length (one loop section), so we don't also
    # pass music_length_ms. Empty section lines keep the render instrumental.
    audio = client.music.compose(
        composition_plan=sig.composition_plan,
        model_id=model_id,
        output_format=output_format,
    )
    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio)
    return b"".join(audio)


def main() -> None:
    args = parse_args()

    if args.list:
        print_catalog()
        return

    signatures = resolve_signatures(args.pairs, args.duration)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for sig in signatures:
        generated_at = datetime.now(timezone.utc).isoformat()
        metadata = sig.metadata(
            provider=PROVIDER,
            model_version=args.model_id,
            generated_at=generated_at,
            output_format=args.output_format,
            license=LICENSE,
        )

        if not args.dry_run:
            data = render_audio(sig, model_id=args.model_id, output_format=args.output_format)
            audio_path = args.out_dir / f"{sig.track_id}.{audio_extension(args.output_format)}"
            audio_path.write_bytes(data)
            metadata["file"] = audio_path.name

        meta_path = args.out_dir / f"{sig.track_id}.json"
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")

        status = "planned" if args.dry_run else "rendered"
        print(f"{status}  {sig.track_id}  ({sig.substrate.name} x {sig.style.name}, {sig.duration_s}s)")

    if args.dry_run:
        print(f"\ndry run: wrote {len(signatures)} metadata file(s) to {args.out_dir}, no audio generated")


if __name__ == "__main__":
    main()
