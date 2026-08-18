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

    uv run scripts/render_background.py --max-retry 5    # try harder on a failing cell
    uv run scripts/render_background.py --no-qc          # keep whatever comes back
    uv run scripts/render_background.py --no-worker      # one process per track

    export ELEVENLABS_API_KEY=...
    uv run scripts/render_background.py --provider elevenlabs --dry-run   # verify the key, render nothing
    uv run scripts/render_background.py --provider elevenlabs
    uv run scripts/render_background.py --provider elevenlabs --output-format mp3_44100_192 --model-id music_v1

Every run — including --dry-run — starts with a preflight check that the selected
provider is actually usable (SA3 CLI discoverable / API key accepted by the API) and
fails fast with setup instructions if not, before touching any spec. For elevenlabs the
key is *verified*, not just read: --dry-run is exactly when you want a bad credential to
surface, before any credits are committed.

Default is credit-safe and idempotent: specs whose audio file is actually on disk are
skipped unless you pass --force. "Actually on disk" is checked per run rather than read
off the catalog's `rendered` flag, so a master that was deleted or lost since the last
rebuild is re-rendered and its spec refilled, instead of being skipped forever. Each render lands in its spec's category cell directory
(assets/tracks/<cell>/, see src/bnb/assets.py) via the CategoryManager.

Three things make a batch cheaper, safer and more accurate than a loop of one-off renders:

*Send each spec to a checkpoint that can render it.* The default (medium, on MLX) covers
the whole taxonomy, so one engine renders the run. It matters when you trade down: only
some checkpoints can render sound effects, and small-music can't — asked for rain it
returns a bass drone. Special cells therefore route to --sa3-sfx-model independently of
--sa3-model, and a run needing two checkpoints loads them in turn rather than quietly
rendering half the library on the wrong one.

*Load the model once where that's possible.* A checkpoint load can dominate a render, so
the batch runs against one resident model when the backend supports it
(bnb.stable_audio.Worker; torch only, i.e. --sa3-backend torch --sa3-model small-music).
The default MLX path reloads per track and absorbs it — ~1s of the 7.5s a 60s medium
track takes. --no-worker opts out where it does apply.

*Check every track before it enters the library.* A generative render fails bluntly
and at random — silent, clipped, a dead constant buffer — so each result goes through
bnb.qc immediately, while the model is still loaded, and a failure is re-rendered on
the spot with a fresh seed (--max-retry, default 3). A track that never passes is left
unrendered rather than shipped broken, and the run exits non-zero.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, NamedTuple

from bnb import assets, qc, stable_audio
from bnb.background import composition_plan_for_model, prompt_for_provider, retry_seed
from bnb.catalog import CategoryManager

DEFAULT_MAX_RETRY = 3  # QC-failed renders to redo before giving up on a track

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
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, choices=sorted(LICENSES))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="elevenlabs: composition model")
    parser.add_argument("--output-format", default=DEFAULT_OUTPUT_FORMAT, help="elevenlabs: output format")
    parser.add_argument("--sa3-model", default=stable_audio.DEFAULT_MODEL, help="stable_audio: model name")
    parser.add_argument("--sa3-backend", default=stable_audio.DEFAULT_BACKEND, choices=stable_audio.BACKENDS)
    parser.add_argument(
        "--sa3-sfx-model",
        default=stable_audio.DEFAULT_SFX_MODEL,
        help=f"stable_audio: checkpoint for special cells (default: {stable_audio.DEFAULT_SFX_MODEL})",
    )
    parser.add_argument(
        "--sa3-sfx-backend",
        default=None,
        choices=stable_audio.BACKENDS,
        help="stable_audio: backend for special cells (default: follow --sa3-backend)",
    )
    parser.add_argument("--sa3-steps", type=int, help="stable_audio: default depends on checkpoint type")
    parser.add_argument("--sa3-cfg", type=float, help="stable_audio: guidance scale, turns negative_prompt on")
    parser.add_argument("--force", action="store_true", help="re-render audio even if the track already has it")
    parser.add_argument("--dry-run", action="store_true", help="list what would render, skip the API/model call")

    quality = parser.add_argument_group("quality gate (bnb.qc)")
    quality.add_argument(
        "--max-retry",
        type=int,
        default=DEFAULT_MAX_RETRY,
        metavar="N",
        help=f"re-render up to N times when a track fails QC (default: {DEFAULT_MAX_RETRY}; 0 to never retry)",
    )
    quality.add_argument("--no-qc", action="store_true", help="keep every render, checked or not")
    quality.add_argument(
        "--no-worker",
        action="store_true",
        help="stable_audio: one process per track instead of one resident model",
    )
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


def sa3_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """``spec`` with the SA3-adapted prompt (AudioSparx tags) in place of the stored one.

    ElevenLabs gets its structure from composition_plan; SA3 gets it from the tag
    vocabulary instead. Both SA3 paths — the per-track CLI and the worker — go through
    here, so they render the same prompt.
    """
    return {**spec, "prompt": prompt_for_provider(spec, "stable_audio")}


def render_stable_audio(
    spec: dict[str, Any], *, model: str, backend: str, steps: int | None, cfg: float | None
) -> Path:
    """Render with the self-hosted Stable Audio 3 checkout, one process per track
    (out-of-process; see src/bnb/stable_audio.py for why)."""
    return stable_audio.render(
        sa3_spec(spec), scratch_wav(), model=model, backend=backend, steps=steps, cfg=cfg
    )


# provider name -> the license string recorded in the render block
LICENSES: dict[str, str] = {
    "elevenlabs": "elevenlabs-eleven-music",
    "stable_audio": "stable-audio-3-community",
}


class Engine(NamedTuple):
    """One (checkpoint, backend) pair — what a batch of specs is rendered by."""

    model: str
    backend: str

    def __str__(self) -> str:
        return f"{self.model} on {self.backend}"


def engine_for(spec: dict[str, Any], args: argparse.Namespace) -> Engine:
    """The checkpoint a spec has to be rendered by.

    Special cells are field recordings, and ``small-music`` has *no* SFX capability
    (docs/guides/prompting.md, "Model Compatibility") — it rendered the first rain
    tracks as bass drones. So they go to an SFX-capable checkpoint while the grid
    stays on the music one, and a run that needs both simply loads both in turn.
    """
    if spec.get("kind") == "special":
        # The *model* is a content decision and differs per cell; the *backend* is a
        # property of the machine (which runtime exists here), so it follows the run's
        # unless overridden — otherwise moving to a CUDA box would need two flags, and
        # forgetting one sends half the library at a runtime that isn't installed.
        return Engine(args.sa3_sfx_model, args.sa3_sfx_backend or args.sa3_backend)
    return Engine(args.sa3_model, args.sa3_backend)


def group_by_engine(
    track_ids: list[str], args: argparse.Namespace, manager: CategoryManager
) -> dict[Engine, list[str]]:
    """Partition the batch by the checkpoint each spec needs, order preserved.

    One entry becomes one model load; ElevenLabs has no checkpoint to choose, so it
    stays a single group.
    """
    groups: dict[Engine, list[str]] = {}
    for track_id in track_ids:
        spec = assets.load_spec(track_id, root=manager.root)
        engine = Engine("elevenlabs", args.model_id) if args.provider == "elevenlabs" else engine_for(spec, args)
        groups.setdefault(engine, []).append(track_id)
    return groups


def per_track_renderer(args: argparse.Namespace, engine: Engine) -> Callable[[dict[str, Any]], Path]:
    """The one-process-per-render path for the selected provider."""
    if args.provider == "elevenlabs":
        return lambda spec: render_elevenlabs(spec, **provider_kwargs(args))
    return lambda spec: render_stable_audio(
        spec, model=engine.model, backend=engine.backend, steps=args.sa3_steps, cfg=args.sa3_cfg
    )


@contextmanager
def render_session(args: argparse.Namespace, engine: Engine) -> Iterator[Callable[[dict[str, Any]], Path]]:
    """Yield ``render(spec) -> Path`` for the whole batch, holding any reusable engine open.

    Model loading dominates a self-hosted render, so where the backend allows it the
    batch runs against one resident model (:class:`stable_audio.Worker`) instead of a
    fresh process per track — which matters twice over now that a QC failure re-renders.
    Everything else (ElevenLabs' HTTP calls, the MLX CLI) is per-track either way, and
    yields the plain one-shot callable.
    """
    per_track = per_track_renderer(args, engine)

    if args.provider != "stable_audio" or args.no_worker or not stable_audio.supports_worker(engine.backend):
        yield per_track
        return

    try:
        worker = stable_audio.Worker(
            model=engine.model, backend=engine.backend, steps=args.sa3_steps, cfg=args.sa3_cfg
        ).start()
    except stable_audio.WorkerError as exc:
        # A missing/unusable worker is a performance regression, not a failure: the
        # per-track CLI renders exactly the same audio, just reloading each time.
        print(f"note: no resident model ({exc}); falling back to one process per track\n")
        yield per_track
        return

    print(f"model          {engine} loaded once, reused for every track in this group\n")
    try:
        yield lambda spec: worker.render(sa3_spec(spec), scratch_wav())
    finally:
        worker.close()


def scratch_wav() -> Path:
    """An empty temp file for a provider that renders straight to a path."""
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    return Path(tmp)


def render_checked(
    render: Callable[[dict[str, Any]], Path], spec: dict[str, Any], *, max_retry: int, check: bool
) -> tuple[Path | None, dict[str, Any]]:
    """Render ``spec``, and re-render it while the result fails QC.

    Generative engines fail bluntly and occasionally — silent, clipped, a dead constant
    buffer (bnb.qc) — and the cheapest moment to catch that is here, while the model is
    already loaded and the track hasn't entered the library yet. Each retry uses a fresh
    seed, since the same seed would mostly reproduce the same broken audio.

    Returns the audio to keep and its QC record, or ``(None, record)`` if every attempt
    failed; failed attempts are deleted rather than left in the scratch directory.
    """
    attempts = max(1, max_retry + 1)
    for attempt in range(attempts):
        seed = spec["seed"] if attempt == 0 else retry_seed(spec["track_id"], attempt)
        path = render({**spec, "seed": seed})
        record: dict[str, Any] = {"attempts": attempt + 1, "seed": seed}

        if not check:
            return path, {**record, "verdict": "unchecked"}

        report = qc.check_track(path)
        record |= {"verdict": report.verdict, "warnings": report.warnings, "metrics": report.metrics}
        if report.ok:
            return path, record

        path.unlink(missing_ok=True)
        remaining = attempts - attempt - 1
        reasons = "; ".join(report.failures)
        tail = f"retrying with a new seed ({remaining} left)" if remaining else "giving up"
        print(f"  qc fail      attempt {attempt + 1}/{attempts}: {reasons} — {tail}")

    return None, record


def provider_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """ElevenLabs' render kwargs. Stable Audio takes its checkpoint from the run's
    :class:`Engine` instead, since that varies per spec within a single run."""
    return {"output_format": args.output_format, "model_id": args.model_id}


def model_version(args: argparse.Namespace, engine: Engine) -> str:
    if args.provider == "elevenlabs":
        return args.model_id
    return f"{engine.model}:{engine.backend}"


def record_output_format(args: argparse.Namespace) -> str:
    """The format string to record in the render block — ElevenLabs' exact requested
    format (e.g. ``pcm_44100``, which a bare ``.wav`` extension can't reconstruct);
    Stable Audio always writes plain wav, so the extension already says everything."""
    return args.output_format if args.provider == "elevenlabs" else "wav"


def describe(spec: dict[str, Any]) -> str:
    if spec.get("kind", "grid") == "special":
        return f"{spec['group']}:{spec['keyword']}"
    return f"{spec['substrate']} x {spec['style']}"


def _redact(api_key: str) -> str:
    """Enough of a key to tell *which* one is loaded, never enough to use."""
    return f"...{api_key[-4:]}" if len(api_key) >= 8 else "(too short to be a key)"


def _api_error_detail(body: Any) -> tuple[str | None, str | None]:
    """``(status, message)`` out of an ElevenLabs error body, which nests both under
    ``detail`` — e.g. ``{"detail": {"status": "invalid_api_key", "message": "..."}}``.

    Worth parsing rather than printing raw, because ``status`` is what separates a key
    the API *rejected* from a key it *recognised and then refused for this endpoint*.
    """
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        return detail.get("status"), detail.get("message")
    if isinstance(detail, str):
        return None, detail
    return None, None


def check_elevenlabs_credential(model_id: str) -> str:
    """Verify ``ELEVENLABS_API_KEY`` actually authenticates — not merely that it is set.

    Checking only that the variable is non-empty passes a typo'd, rotated or revoked key,
    and the run then dies on the first *paid* call, partway into a batch, with a raw 401
    from inside the SDK. This spends one cheap GET (the account's subscription record) to
    settle it up front, which is the whole point of a preflight and the reason it runs on
    ``--dry-run`` too: a dry run is exactly when you want to learn the credential is wrong,
    before committing credits.

    The quota counters come back from the same call, so they're worth printing — but they
    are the account-wide character counters, which is not a music-render cost estimate;
    they say the account is live and roughly how much room is on it, nothing finer.

    A *restricted* key that cannot read the account is explicitly not a failure here — see
    the ``missing_permissions`` branch. The probe is the cheapest authenticated GET
    available, not a permission the render itself needs.
    """
    raw_key = os.environ.get("ELEVENLABS_API_KEY") or ""
    # `export ELEVENLABS_API_KEY=$(cat key.txt)` leaves a trailing newline, which the API
    # rejects as an invalid key — indistinguishable, from the error, from a wrong one.
    api_key = raw_key.strip()
    if not api_key:
        raise SystemExit("ELEVENLABS_API_KEY is not set; export it, or use --provider stable_audio")
    whitespace_note = " [note: surrounding whitespace was trimmed]" if api_key != raw_key else ""

    try:
        from elevenlabs import ElevenLabs
        from elevenlabs.core.api_error import ApiError
    except ImportError as exc:  # the SDK is an optional extra (pyproject [media])
        raise SystemExit(
            f"--provider elevenlabs needs the elevenlabs SDK, which is not installed ({exc}). "
            "Install it with `uv sync --extra media`, or use --provider stable_audio."
        ) from exc

    try:
        subscription = ElevenLabs(api_key=api_key).user.subscription.get()
    except ApiError as exc:
        status, message = _api_error_detail(exc.body)
        if status == "missing_permissions":
            # Not a bad key. ElevenLabs keys can be *restricted* to chosen scopes, and the
            # API can only name the scope it wants after recognising the key — so this
            # proves authentication succeeded. Only the account read was refused, and
            # nothing about rendering needs it. Failing here would reject a perfectly good
            # music-generation key, which is precisely the wrong answer for a preflight.
            return (
                f"elevenlabs, model {model_id} (key {_redact(api_key)} accepted; account not "
                f"readable, so quota is unknown — {message or 'the key lacks user_read'})"
                f"{whitespace_note}"
            )
        if exc.status_code in (401, 403):
            raise SystemExit(
                f"ELEVENLABS_API_KEY was rejected by the API ({exc.status_code}"
                f"{f', {status}' if status else ''}): {message or exc.body}\n"
                f"The variable is set (ending {_redact(api_key)}), so this is a wrong, rotated "
                f"or revoked key rather than a missing one — check the key in the ElevenLabs "
                f"dashboard.{whitespace_note}"
            ) from exc
        raise SystemExit(
            f"could not verify ELEVENLABS_API_KEY: the API returned {exc.status_code}: "
            f"{message or exc.body}"
        ) from exc
    except Exception as exc:  # network/DNS/TLS — a render would fail the same way
        raise SystemExit(
            f"could not reach the ElevenLabs API to verify ELEVENLABS_API_KEY: {exc!r}"
        ) from exc

    used, limit = subscription.character_count, subscription.character_limit
    quota = f"{used:,}/{limit:,} characters used" if limit else f"{used:,} characters used"
    return (
        f"elevenlabs, model {model_id} (key {_redact(api_key)} verified; "
        f"tier {subscription.tier}, {quota}){whitespace_note}"
    )


def check_provider_ready(args: argparse.Namespace, engines: Iterable[Engine] = ()) -> str:
    """Verify every engine the run needs is actually usable, and return a short
    readiness description to print. Raises ``SystemExit`` with setup instructions if
    not — run unconditionally (dry-run included) so a misconfigured engine fails before
    any spec is rendered, not partway through a batch or silently (a --dry-run where
    every track is already rendered would otherwise never call the provider at all).
    """
    if args.provider == "elevenlabs":
        return check_elevenlabs_credential(args.model_id)

    engines = list(engines) or [Engine(args.sa3_model, args.sa3_backend)]
    described = []
    for engine in engines:
        try:
            cli = stable_audio.require_cli(engine.backend)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        described.append(f"{engine} ({cli})")
    return "stable_audio, " + "; ".join(described)


def check_engines_can_render(
    args: argparse.Namespace, groups: dict[Engine, list[str]], manager: CategoryManager
) -> None:
    """Refuse to render special cells on a checkpoint with no SFX capability.

    The failure is silent otherwise — small-music happily returns 60 seconds of
    plausible-sounding *music* when asked for rain — so the only place to catch it is
    before the render, by checking the model against what it was trained to do.
    """
    if args.provider != "stable_audio":
        return
    for engine, track_ids in groups.items():
        if stable_audio.supports_sfx(engine.model):
            continue
        special = [t for t in track_ids if assets.load_spec(t, root=manager.root).get("kind") == "special"]
        if special:
            raise SystemExit(
                f"{engine.model} cannot render sound effects, but {len(special)} special "
                f"cell(s) are targeted (e.g. {special[0]}).\n"
                f"  pass --sa3-sfx-model with one of: {', '.join(stable_audio.SFX_MODELS)}"
            )


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


def select_todo(
    targets: Iterable[str], manager: CategoryManager, *, force: bool = False
) -> tuple[list[str], int]:
    """Split ``targets`` into what still needs rendering, and count how many of those are
    *orphans* — tracks the catalog calls rendered whose audio file is not actually there.

    The catalog's ``rendered`` flag is a snapshot from its last rebuild, so it goes stale
    the moment a master is deleted, moved or lost — and a track skipped on a stale flag is
    skipped *forever*: never re-rendered, while its spec still advertises a render whose
    file is gone. So the skip decision asks the filesystem, and the flag is kept only to
    report the discrepancy. Nothing further is needed to repair the spec:
    :func:`assets.record_render` replaces the whole ``render`` block, so re-rendering
    refills it.
    """
    catalog_rendered = {e["track_id"] for e in manager.search(rendered=True)}
    todo: list[str] = []
    orphaned = 0
    for track_id in targets:
        if not force and assets.find_track(track_id, root=manager.root) is not None:
            print(f"skip (audio)   {track_id}")
            continue
        if not force and track_id in catalog_rendered:
            orphaned += 1
            print(f"re-render      {track_id}  (catalog says rendered, audio file is missing)")
        todo.append(track_id)
    return todo, orphaned


def main() -> None:
    args = parse_args()
    license = LICENSES[args.provider]
    # The credential check needs nothing from the catalog, so it runs before the scan: a
    # bad key then fails in about a second, instead of underneath a screenful of skip
    # lines. SA3's engine preflight genuinely does depend on what is in the batch (special
    # cells route to a different checkpoint), so that half stays below, after grouping.
    readiness = check_elevenlabs_credential(args.model_id) if args.provider == "elevenlabs" else None
    manager = CategoryManager()

    targets = target_ids(args, manager)
    todo, orphaned = select_todo(targets, manager, force=args.force)
    skipped = len(targets) - len(todo)
    repair = f", {orphaned} missing audio" if orphaned else ""

    # Preflight against the engines this batch actually needs — which depends on what
    # is in it, since special cells route to a different checkpoint than the grid.
    groups = group_by_engine(todo, args, manager)
    check_engines_can_render(args, groups, manager)
    if readiness is None:
        readiness = check_provider_ready(args, groups)
    print(f"provider       {readiness}")
    print(f"quality gate   {'off' if args.no_qc else f'bnb.qc, up to {args.max_retry} re-render(s)'}\n")

    if args.dry_run:
        for engine, track_ids in groups.items():
            for track_id in track_ids:
                spec = assets.load_spec(track_id, root=manager.root)
                print(f"planned        {track_id}  ({describe(spec)}, {engine})")
        catalog = manager.rebuild()
        print(
            f"\ndry run: {len(todo)} would render{repair}, {skipped} skipped; "
            f"catalog: {catalog['count']} tracks"
        )
        return

    rendered = failed = 0
    # One session per engine: with a resident model this is where each load is paid.
    for engine, track_ids in groups.items():
        with render_session(args, engine) as render:
            for track_id in track_ids:
                spec = assets.load_spec(track_id, root=manager.root)
                tmp_path, quality = render_checked(
                    render, spec, max_retry=args.max_retry, check=not args.no_qc
                )
                if tmp_path is None:
                    failed += 1
                    print(f"failed qc      {track_id}  ({describe(spec)}) — not added to the library")
                    continue

                audio_path = manager.attach_render(
                    spec,
                    tmp_path,
                    provider=args.provider,
                    model_version=model_version(args, engine),
                    license=license,
                    generated_at=datetime.now(timezone.utc).isoformat(),
                    output_format=record_output_format(args),
                    seed=quality["seed"],
                    qc=quality,
                    rebuild=False,
                )
                rendered += 1
                note = summarize_quality(quality)
                print(f"rendered       {track_id}  -> {audio_path.relative_to(manager.root)}{note}")

    catalog = manager.rebuild()
    print(
        f"\n{rendered} rendered{repair}, {skipped} skipped, {failed} failed qc; "
        f"catalog: {catalog['count']} tracks"
    )
    if failed:
        # Non-zero so a pipeline notices, and the specs stay unrendered so the next run retries.
        raise SystemExit(f"{failed} track(s) never passed qc; inspect with scripts/check_background.py")


def summarize_quality(quality: dict[str, Any]) -> str:
    """The part of a QC record worth putting on the render line."""
    notes = []
    if quality["attempts"] > 1:
        notes.append(f"{quality['attempts']} attempts")
    notes += quality.get("warnings", [])
    return f"  ({'; '.join(notes)})" if notes else ""


if __name__ == "__main__":
    main()
