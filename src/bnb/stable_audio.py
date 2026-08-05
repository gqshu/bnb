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
           documents as base-model-only.

Override executable discovery with ``BNB_SA3_CLI`` / ``BNB_SA3_MLX_CLI``, or point
``BNB_SA3_REPO`` at another checkout.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "small-music"
DEFAULT_BACKEND = "torch"
BACKENDS = ("torch", "mlx")
DEFAULT_REPO = Path(__file__).resolve().parents[2].parent / "stable-audio-3"

# Max render length per model family (README "Max length").
MAX_DURATION_S = {"small": 120, "medium": 380}

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

    executable = cli or cli_path(backend)
    if executable is None:
        raise RuntimeError(
            f"no {backend} CLI found; set up stable-audio-3 next to bnb "
            f"({'uv sync' if backend == 'torch' else 'optimized/mlx/install.sh'}), "
            "or set BNB_SA3_CLI / BNB_SA3_MLX_CLI / BNB_SA3_REPO"
        )

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
    """Generate the spec's audio into ``out_path``; return it."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_command(
        spec, out_path, model=model, duration_s=duration_s, steps=steps, cfg=cfg, backend=backend
    )
    subprocess.run(cmd, check=True, timeout=timeout)
    return out_path
