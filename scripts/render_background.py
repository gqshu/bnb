"""Render background specs into audio via a provider: Stable Audio 3 (self-hosted,
out-of-process — see src/bnb/stable_audio.py; the default) or ElevenLabs (paid, hosted).

Reads render-independent specs written by scripts/plan_background.py and turns
them into audio masters, filling each spec's ``render`` block. Rendering is the
paid / heavyweight half, kept separate so specs can be previewed for free first.

    uv run scripts/render_background.py                  # render everything missing audio (Stable Audio 3)
    uv run scripts/render_background.py --dry-run        # preflight + list what would render, no render
    uv run scripts/render_background.py buddhist_meditative_drone_seed81657   # render specific track(s)
    uv run scripts/render_background.py --force buddhist_meditative_drone_seed81657   # re-render existing
    uv run scripts/render_background.py --sa3-backend mlx --sa3-model medium
    uv run scripts/render_background.py --sa3-cfg 5      # turn on guidance

    export ELEVENLABS_API_KEY=...
    uv run scripts/render_background.py --provider elevenlabs
    uv run scripts/render_background.py --provider elevenlabs --output-format mp3_44100_192 --model-id music_v1

Every run — including --dry-run — starts with a preflight check that the selected
provider is actually usable (SA3 CLI discoverable / API key set) and fails fast with
setup instructions if not, before touching any spec.

Default is credit-safe and idempotent: specs that already have audio are skipped
unless you pass --force. Each render lands in its spec's category cell directory
(assets/tracks/<cell>/, see src/bnb/assets.py) via the CategoryManager.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from bnb import assets, stable_audio
from bnb.background import composition_plan_for_model, prompt_for_provider
from bnb.catalog import CategoryManager

DEFAULT_PROVIDER = "stable_audio"  # self-hosted: no per-track API cost once the checkout is set up
DEFAULT_OUTPUT_FORMAT = "pcm_44100"  # elevenlabs only -> WAV master (§2); mp3_*/opus_* also accepted
DEFAULT_MODEL_ID = "music_v2"  # elevenlabs only: newest Eleven Music model; music_v1 available via --model-id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "track_ids",
        nargs="*",
        help="specific specs to render (default: every spec missing audio)",
    )
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, choices=sorted(PROVIDERS))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="elevenlabs: composition model")
    parser.add_argument("--output-format", default=DEFAULT_OUTPUT_FORMAT, help="elevenlabs: output format")
    parser.add_argument("--sa3-model", default=stable_audio.DEFAULT_MODEL, help="stable_audio: model name")
    parser.add_argument("--sa3-backend", default=stable_audio.DEFAULT_BACKEND, choices=stable_audio.BACKENDS)
    parser.add_argument("--sa3-steps", type=int, help="stable_audio: default depends on checkpoint type")
    parser.add_argument("--sa3-cfg", type=float, help="stable_audio: guidance scale, turns negative_prompt on")
    parser.add_argument("--force", action="store_true", help="re-render audio even if the track already has it")
    parser.add_argument("--dry-run", action="store_true", help="list what would render, skip the API/model call")
    return parser.parse_args()


def render_elevenlabs(spec: dict[str, Any], *, output_format: str, model_id: str) -> Path:
    """Call Eleven Music with the model-appropriate composition plan and seed, and
    write the result to a scratch file for the caller to place in the repo.

    The plan shape differs per model (v1 sections vs. v2 chunks), so we translate
    the stored spec via ``composition_plan_for_model``. ``force_instrumental`` is
    *not* passed: the API rejects it alongside a composition plan (422), and the
    plan itself forces a wordless render (no lyric lines + ``vocals`` negative, §6).
    """
    from elevenlabs import ElevenLabs

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise SystemExit("ELEVENLABS_API_KEY is not set")

    client = ElevenLabs(api_key=api_key)
    stream = client.music.compose(
        composition_plan=composition_plan_for_model(spec, model_id),
        model_id=model_id,
        output_format=output_format,
        seed=spec["seed"],
    )
    data = stream if isinstance(stream, (bytes, bytearray)) else b"".join(stream)

    fd, tmp = tempfile.mkstemp(suffix=f".{assets.audio_extension(output_format)}")
    os.close(fd)
    tmp_path = Path(tmp)
    if output_format.startswith("pcm"):
        sample_rate = int(output_format.split("_", 1)[1])
        assets.write_pcm_wav(tmp_path, bytes(data), sample_rate)
    else:
        tmp_path.write_bytes(bytes(data))
    return tmp_path


def render_stable_audio(
    spec: dict[str, Any], *, model: str, backend: str, steps: int | None, cfg: float | None
) -> Path:
    """Render with the self-hosted Stable Audio 3 checkout (out-of-process; see
    src/bnb/stable_audio.py for why). Uses the SA3-adapted prompt (AudioSparx tags),
    not the bare stored prompt — ElevenLabs gets structure from composition_plan,
    SA3 gets it from the tag vocabulary instead.
    """
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sa3_spec = {**spec, "prompt": prompt_for_provider(spec, "stable_audio")}
    return stable_audio.render(sa3_spec, Path(tmp), model=model, backend=backend, steps=steps, cfg=cfg)


# provider name -> (render callable, license string for the render block)
PROVIDERS: dict[str, tuple[Callable[..., Path], str]] = {
    "elevenlabs": (render_elevenlabs, "elevenlabs-eleven-music"),
    "stable_audio": (render_stable_audio, "stable-audio-3-community"),
}


def provider_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.provider == "elevenlabs":
        return {"output_format": args.output_format, "model_id": args.model_id}
    return {"model": args.sa3_model, "backend": args.sa3_backend, "steps": args.sa3_steps, "cfg": args.sa3_cfg}


def model_version(args: argparse.Namespace) -> str:
    if args.provider == "elevenlabs":
        return args.model_id
    return f"{args.sa3_model}:{args.sa3_backend}"


def record_output_format(args: argparse.Namespace) -> str:
    """The format string to record in the render block — ElevenLabs' exact requested
    format (e.g. ``pcm_44100``, which a bare ``.wav`` extension can't reconstruct);
    Stable Audio always writes plain wav, so the extension already says everything."""
    return args.output_format if args.provider == "elevenlabs" else "wav"


def describe(spec: dict[str, Any]) -> str:
    if spec.get("kind", "grid") == "special":
        return f"{spec['group']}:{spec['keyword']}"
    return f"{spec['substrate']} x {spec['style']}"


def check_provider_ready(args: argparse.Namespace) -> str:
    """Verify the selected provider is actually usable and return a short readiness
    description to print. Raises ``SystemExit`` with setup instructions if not — run
    unconditionally (dry-run included) so a misconfigured engine fails before any
    spec is touched, not partway through a batch or silently (a --dry-run where every
    track is already rendered would otherwise never call the provider at all).
    """
    if args.provider == "elevenlabs":
        if not os.environ.get("ELEVENLABS_API_KEY"):
            raise SystemExit("ELEVENLABS_API_KEY is not set; export it, or use --provider stable_audio")
        return f"elevenlabs, model {args.model_id}"

    try:
        cli = stable_audio.require_cli(args.sa3_backend)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    return f"stable_audio, {args.sa3_model} on {args.sa3_backend} ({cli})"


def target_ids(args: argparse.Namespace, manager: CategoryManager) -> list[str]:
    available = {entry["track_id"] for entry in manager.search()}
    if not available:
        raise SystemExit("no specs to render; run scripts/plan_background.py first")
    if args.track_ids:
        missing = [t for t in args.track_ids if t not in available]
        if missing:
            raise SystemExit(f"no spec for: {', '.join(missing)}")
        return args.track_ids
    return sorted(available)


def main() -> None:
    args = parse_args()
    render, license = PROVIDERS[args.provider]
    readiness = check_provider_ready(args)
    print(f"provider       {readiness}\n")

    manager = CategoryManager()
    rendered_ids = {e["track_id"] for e in manager.search(rendered=True)}

    rendered = skipped = planned = 0
    for track_id in target_ids(args, manager):
        if track_id in rendered_ids and not args.force:
            skipped += 1
            print(f"skip (audio)   {track_id}")
            continue

        spec = assets.load_spec(track_id, root=manager.root)
        if args.dry_run:
            planned += 1
            print(f"planned        {track_id}  ({describe(spec)})")
            continue

        tmp_path = render(spec, **provider_kwargs(args))
        audio_path = manager.attach_render(
            spec,
            tmp_path,
            provider=args.provider,
            model_version=model_version(args),
            license=license,
            generated_at=datetime.now(timezone.utc).isoformat(),
            output_format=record_output_format(args),
            rebuild=False,
        )
        rendered += 1
        print(f"rendered       {track_id}  -> {audio_path.relative_to(manager.root)}")

    catalog = manager.rebuild()
    if args.dry_run:
        print(f"\ndry run: {planned} would render, {skipped} skipped; catalog: {catalog['count']} tracks")
    else:
        print(f"\n{rendered} rendered, {skipped} skipped; catalog: {catalog['count']} tracks")


if __name__ == "__main__":
    main()
