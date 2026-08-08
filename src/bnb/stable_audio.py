"""Render background specs with self-hosted Stable Audio 3 (docs/background_music.md §1).

The plan names self-hosted Stable Audio 3 Open as the primary library engine, with
ElevenLabs only as the paid stopgap. This module is the local half: it turns a spec
(the provider-agnostic record written by ``scripts/plan_background.py``) into a
Stable Audio 3 invocation and a WAV master.

Stable Audio 3 runs **out of process**. bnb targets Python 3.14 and torch has no
3.14 wheels, so the model cannot live in this venv; instead we shell out to the
sibling ``stable-audio-3`` checkout, which brings its own venv.

Two backends, because on Apple Silicon they cover different models:

``torch``  the upstream ``stable-audio`` CLI. ``small-music`` runs fine on MPS,
           but ``medium`` needs CUDA + Flash Attention (SAME-L uses sliding-window
           attention), so on a Mac this backend is small-models-only.
``mlx``    ``optimized/mlx/sa3`` — a pure-MLX reimplementation, Metal-backed, no
           torch. This is the only way to run ``medium`` on a Mac; it also writes
           16-bit PCM (matching the ElevenLabs masters) rather than float32, and
           honours ``--cfg`` on post-trained checkpoints, which the torch path
           documents as base-model-only. The default, since the default model is
           ``medium``.

Two ways to drive it, same audio out of both:

:func:`render`  one CLI process per track — the default path, and the only option for
                ``mlx``. A load per track, which MLX keeps cheap (~1s of medium's 7.5s
                for a 60s track).
:class:`Worker` a resident process (``sa3_worker.py``, run by the sibling venv's
                python) that loads the checkpoint once and renders on request. ``torch``
                only, so it applies when you trade down to ``--sa3-model small-music
                --sa3-backend torch``, where the load is 8s and dominates everything.

Override executable discovery with ``BNB_SA3_CLI`` / ``BNB_SA3_MLX_CLI``, or point
``BNB_SA3_REPO`` at another checkout.
"""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# medium is the library default: it is the only checkpoint that covers both halves of
# the taxonomy (small-music cannot render sound effects at all), and on the music cells
# it measures visibly livelier — frame-level variation roughly doubled on the same seed
# and prompt (5.3 -> 9.3 dB on singing bowls, 5.6 -> 10.3 dB on the noise bed), which is
# the "less plain" axis. It costs ~3x the sampling time (7.5s vs 2.4s for a 60s track on
# MLX) and ~4 GB peak RAM, both of which are cheap next to re-rendering a library twice.
#
# It implies the MLX backend on Apple Silicon: medium needs Flash Attention 2 under
# torch, whose prebuilt wheels are CUDA/Linux only. Set --sa3-backend torch --sa3-model
# small-music to go back to the small checkpoint (and to the resident-model Worker,
# which is torch-only).
DEFAULT_MODEL = "medium"
DEFAULT_BACKEND = "mlx"
BACKENDS = ("torch", "mlx")
DEFAULT_REPO = Path(__file__).resolve().parents[2].parent / "stable-audio-3"

# Max render length per model family (README "Max length").
MAX_DURATION_S = {"small": 120, "medium": 380}

# Which checkpoints can render sound effects at all (docs/guides/prompting.md, "Model
# Compatibility"). This is not a quality preference: small-music has no SFX capability
# in its training distribution, and asking it for rain yields a bass drone (a 150 Hz
# spectral centroid, measured) or, with an SFX-shaped prompt, near silence.
SFX_MODELS = ("medium", "medium-base", "small-sfx", "small-sfx-base")
DEFAULT_SFX_MODEL = DEFAULT_MODEL  # medium covers both halves, so one checkpoint does the run


def supports_sfx(model: str) -> bool:
    return model in SFX_MODELS

# bnb model name -> (--dit, --decoder) for the MLX CLI, which names them differently
# and takes the codec as a separate flag. The "-base" checkpoints aren't converted.
MLX_MODELS = {
    "small-music": ("sm-music", "same-s"),
    "small-sfx": ("sm-sfx", "same-s"),
    "medium": ("medium", "same-l"),
}

# Post-trained checkpoints are distilled to ~8 steps and ignore cfg_scale /
# negative_prompt; the "-base" checkpoints need many more steps and *do* respond
# to guidance (docs/workflows/inference.md, "Controls").
POST_TRAINED_STEPS = 8
BASE_STEPS = 50


def is_base_model(model: str) -> bool:
    return model.endswith("-base")


def default_steps(model: str) -> int:
    return BASE_STEPS if is_base_model(model) else POST_TRAINED_STEPS


def max_duration_s(model: str) -> int:
    family = "small" if model.startswith("small") else "medium"
    return MAX_DURATION_S[family]


def cli_path(backend: str = DEFAULT_BACKEND) -> Path | None:
    """The executable driving ``backend``, or None if that checkout isn't set up."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}, expected one of {list(BACKENDS)}")

    override = os.environ.get("BNB_SA3_CLI" if backend == "torch" else "BNB_SA3_MLX_CLI")
    if override:
        path = Path(override)
        return path if path.exists() else None

    repo = Path(os.environ.get("BNB_SA3_REPO", DEFAULT_REPO))
    path = repo / ".venv" / "bin" / "stable-audio" if backend == "torch" else repo / "optimized" / "mlx" / "sa3"
    return path if path.exists() else None


def require_cli(backend: str = DEFAULT_BACKEND) -> Path:
    """:func:`cli_path`, or a ``RuntimeError`` with setup instructions if it's missing.

    Split out from :func:`build_command` so a caller (e.g. a preflight check) can
    verify the engine is usable before doing any other work, not just at render time.
    """
    executable = cli_path(backend)
    if executable is None:
        raise RuntimeError(
            f"no {backend} CLI found; set up stable-audio-3 next to bnb "
            f"({'uv sync' if backend == 'torch' else 'optimized/mlx/install.sh'}), "
            "or set BNB_SA3_CLI / BNB_SA3_MLX_CLI / BNB_SA3_REPO"
        )
    return executable


def build_command(
    spec: dict[str, Any],
    out_path: Path | str,
    *,
    model: str = DEFAULT_MODEL,
    duration_s: int | None = None,
    steps: int | None = None,
    cfg: float | None = None,
    backend: str = DEFAULT_BACKEND,
    cli: Path | str | None = None,
) -> list[str]:
    """The argv that renders ``spec`` to ``out_path``.

    Kept separate from :func:`render` so the invocation is assertable without
    loading a model. ``cfg`` turns on classifier-free guidance toward the prompt and
    away from ``negative_prompt``; leave it None for the distilled single-pass default.
    """
    duration = spec["duration_s"] if duration_s is None else duration_s
    limit = max_duration_s(model)
    if duration > limit:
        raise ValueError(f"{model} renders at most {limit}s, got {duration}s")

    executable = cli or require_cli(backend)
    steps = default_steps(model) if steps is None else steps
    if backend == "mlx":
        return _mlx_command(spec, out_path, model=model, duration=duration, steps=steps, cfg=cfg, cli=executable)
    return _torch_command(spec, out_path, model=model, duration=duration, steps=steps, cfg=cfg, cli=executable)


def _torch_command(spec, out_path, *, model, duration, steps, cfg, cli) -> list[str]:
    cmd = [
        str(cli),
        "--model",
        model,
        "-p",
        spec["prompt"],
        "--duration",
        str(duration),
        "--steps",
        str(steps),
        "--seed",
        str(spec["seed"]),  # same seed as the ElevenLabs render, so cells stay comparable
        "-o",
        str(out_path),
    ]
    if cfg is not None:
        cmd += ["--cfg-scale", str(cfg)]
    # Guidance is a no-op on post-trained checkpoints unless cfg is turned up, so the
    # negative prompt rides along only where it can actually steer.
    if is_base_model(model) or cfg is not None:
        cmd += ["--negative-prompt", spec["negative_prompt"]]
    return cmd


def _mlx_command(spec, out_path, *, model, duration, steps, cfg, cli) -> list[str]:
    if model not in MLX_MODELS:
        raise ValueError(f"{model} has no MLX conversion; expected one of {list(MLX_MODELS)}")
    dit, decoder = MLX_MODELS[model]

    cmd = [
        str(cli),
        "--dit",
        dit,
        "--decoder",
        decoder,
        "--prompt",
        spec["prompt"],
        "--seconds",
        str(duration),
        "--steps",
        str(steps),
        # Absolute, so the CLI writes here instead of under optimized/mlx/output/.
        "--out",
        str(Path(out_path).resolve()),
    ]
    if spec["seed"] >= 0:  # this CLI has no -1 sentinel: omitting --seed is the random path
        cmd += ["--seed", str(spec["seed"])]
    if cfg is not None:
        cmd += ["--cfg", str(cfg), "--negative-prompt", spec["negative_prompt"]]
    return cmd


def render(
    spec: dict[str, Any],
    out_path: Path | str,
    *,
    model: str = DEFAULT_MODEL,
    duration_s: int | None = None,
    steps: int | None = None,
    cfg: float | None = None,
    backend: str = DEFAULT_BACKEND,
    timeout: float | None = 900,
) -> Path:
    """Generate the spec's audio into ``out_path``; return it.

    One process per track, so the checkpoint is loaded per render — fine for a single
    track, wasteful for a batch. :class:`Worker` keeps one model resident instead.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_command(
        spec, out_path, model=model, duration_s=duration_s, steps=steps, cfg=cfg, backend=backend
    )
    subprocess.run(cmd, check=True, timeout=timeout)
    return out_path


# --- Persistent worker: one model load per batch, not per track ----------------
#
# Loading a checkpoint dominates a render (tens of seconds against a few seconds of
# sampling for a 60s track), and a QC retry pays it twice. sa3_worker.py holds the
# model open and renders on request; this is the client half.
#
# torch only. The MLX entry point is a script whose sampling loop lives inline in
# main(), with no importable generate step to call twice — driving it as a worker
# would mean reimplementing that loop here, and a divergent copy of someone else's
# sampler is a worse trade than reloading the model.

WORKER_SCRIPT = Path(__file__).resolve().parent / "sa3_worker.py"
WORKER_BACKENDS = ("torch",)


def supports_worker(backend: str) -> bool:
    """Whether ``backend`` can hold a model open across renders."""
    return backend in WORKER_BACKENDS


def venv_python(backend: str = WORKER_BACKENDS[0]) -> Path | None:
    """The interpreter that can import ``stable_audio_3`` — the one beside the CLI.

    Derived from :func:`cli_path` rather than the repo layout so a ``BNB_SA3_CLI``
    override pointing at another venv still resolves to that venv's python.
    """
    cli = cli_path(backend)
    if cli is None:
        return None
    python = cli.parent / "python"
    return python if python.exists() else None


class WorkerError(RuntimeError):
    """The worker process failed to start, died, or rejected a render."""


class Worker:
    """A Stable Audio 3 process that loads the checkpoint once and renders on request.

    Use as a context manager; the model stays resident for the block::

        with Worker(model="small-music") as worker:
            for spec in specs:
                worker.render(spec, out_dir / f"{spec['track_id']}.wav")

    Every render is an independent ``generate`` call seeded from the spec, exactly as
    the one-shot CLI path does it, so reusing the process changes cost and nothing else.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        backend: str = WORKER_BACKENDS[0],  # torch: the only backend a worker can hold open
        steps: int | None = None,
        cfg: float | None = None,
        timeout: float = 900,
        load_timeout: float = 900,
    ):
        if not supports_worker(backend):
            raise WorkerError(f"backend {backend!r} has no worker; expected one of {list(WORKER_BACKENDS)}")
        self.model = model
        self.backend = backend
        self.steps = default_steps(model) if steps is None else steps
        self.cfg = cfg
        self.timeout = timeout
        self.load_timeout = load_timeout
        self.process: subprocess.Popen[str] | None = None

    # --- lifecycle ------------------------------------------------------------

    def start(self) -> Worker:
        """Spawn the worker and block until the model is loaded and it reports ready."""
        python = venv_python(self.backend)
        if python is None:
            raise WorkerError(
                f"no python next to the {self.backend} CLI; set up stable-audio-3 (uv sync) "
                "or set BNB_SA3_CLI / BNB_SA3_REPO"
            )
        self.process = subprocess.Popen(
            [str(python), str(WORKER_SCRIPT), "--model", self.model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherited: model progress stays visible to the user
            text=True,
            bufsize=1,
        )
        reply = self._read(self.load_timeout)
        if reply.get("event") != "ready":
            raise WorkerError(f"worker failed to load {self.model}: {reply.get('message', reply)}")
        return self

    def close(self) -> None:
        """Close stdin (the worker's exit signal) and reap it."""
        process, self.process = self.process, None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def __enter__(self) -> Worker:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- rendering ------------------------------------------------------------

    def render(self, spec: dict[str, Any], out_path: Path | str) -> Path:
        """Render one spec into ``out_path``, reusing the loaded model."""
        if self.process is None:
            raise WorkerError("worker is not running; call start() or use it as a context manager")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        duration = spec["duration_s"]
        limit = max_duration_s(self.model)
        if duration > limit:
            raise ValueError(f"{self.model} renders at most {limit}s, got {duration}s")

        request = {
            "prompt": spec["prompt"],
            # Guidance is a no-op on post-trained checkpoints unless cfg is turned up,
            # so the negative prompt rides along only where it can steer (as in _torch_command).
            "negative_prompt": spec["negative_prompt"] if (is_base_model(self.model) or self.cfg) else None,
            "duration_s": duration,
            "steps": self.steps,
            "cfg": self.cfg,
            "seed": spec["seed"],
            "out": str(out_path.resolve()),
        }
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()

        reply = self._read(self.timeout)
        if reply.get("event") != "rendered":
            raise WorkerError(f"render failed for {spec['track_id']}: {reply.get('message', reply)}")
        return out_path

    def _read(self, timeout: float) -> dict[str, Any]:
        """The worker's next protocol line, or a ``WorkerError`` if it dies or stalls.

        The worker redirects its own output to stderr, but a third-party import can
        still land a line on stdout before that takes effect, so anything unparseable
        is forwarded to stderr rather than treated as a reply. A poll-and-wait keeps a
        crashed or wedged process from hanging the batch.
        """
        assert self.process is not None and self.process.stdout is not None
        deadline = time.monotonic() + timeout
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            while True:
                if not selector.select(max(0.0, deadline - time.monotonic())):
                    self.close()
                    raise WorkerError(f"worker produced no reply within {timeout:g}s")
                line = self.process.stdout.readline()
                if not line:  # EOF: the process is gone
                    self.close()
                    raise WorkerError("worker exited unexpectedly (see its output above)")
                if not line.strip():
                    continue
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    print(line, end="", file=sys.stderr)
        finally:
            selector.close()
