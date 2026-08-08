"""Long-lived Stable Audio 3 render worker — runs *inside* the stable-audio-3 venv.

The upstream CLI loads the checkpoint, renders one prompt, and exits. Across a batch
that is the dominant cost: loading ``small-music`` takes tens of seconds against a
handful of seconds of sampling per 60s track, and a QC retry pays it all over again.
This script keeps one model resident and renders on request, so an N-track run loads
once instead of N times.

It cannot live in the bnb venv (bnb targets Python 3.14; torch has no wheels for it),
so ``bnb.stable_audio.Worker`` starts it with the *sibling checkout's* interpreter.
That makes this file a payload, not a library: stdlib + torch + torchaudio +
stable_audio_3 only, no ``bnb`` imports.

Protocol — one JSON object per line each way, over stdin/stdout::

    <- {"event": "ready", "model": "small-music", "sample_rate": 44100}
    -> {"prompt": "...", "negative_prompt": "..." | null, "duration_s": 60,
        "steps": 8, "cfg": null, "seed": 41654, "out": "/tmp/x.wav"}
    <- {"event": "rendered", "out": "/tmp/x.wav"}          (or "error" + "message")

Anything the model prints goes to stderr, so progress stays visible to the user
without corrupting the protocol stream. EOF on stdin ends the process.

Batching several prompts into one ``generate`` call would save more still, but the
seed is set once per call (``torch.manual_seed``) for the whole batch, so only the
first item would reproduce what a single render of that spec produces. Per-track
reproducibility is worth more than the remaining overlap: renders stay bit-comparable
with the one-at-a-time CLI path.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback

# Claimed *before* the model imports, which chatter on stdout at import time (a missing
# flash_attn, for one). The parent reads this stream as protocol, so nothing else may
# write to it; everything printed from here on goes to stderr instead.
PROTOCOL = sys.stdout
sys.stdout = sys.stderr

import torchaudio  # noqa: E402
from stable_audio_3 import StableAudioModel  # noqa: E402


def send(**payload: object) -> None:
    PROTOCOL.write(json.dumps(payload) + "\n")
    PROTOCOL.flush()


def render(model: StableAudioModel, request: dict) -> str:
    """One request -> one WAV on disk; returns where it landed."""
    audio = model.generate(
        prompt=request["prompt"],
        negative_prompt=request.get("negative_prompt"),
        duration=float(request["duration_s"]),
        steps=int(request["steps"]),
        cfg_scale=float(request.get("cfg") or 1.0),
        seed=int(request["seed"]),
        batch_size=1,
        sample_size=model.model_config["sample_size"],
    )
    out = request["out"]
    torchaudio.save(out, audio[0].cpu(), model.model.sample_rate)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (auto-detected if omitted)")
    parser.add_argument("--no-half", action="store_true")
    args = parser.parse_args()

    try:
        model = StableAudioModel.from_pretrained(
            args.model, device=args.device, model_half=not args.no_half
        )
    except Exception as exc:
        send(event="error", stage="load", message=f"{type(exc).__name__}: {exc}")
        raise SystemExit(1)

    send(event="ready", model=args.model, sample_rate=model.model.sample_rate)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            send(event="rendered", out=render(model, json.loads(line)))
        except Exception as exc:
            # Keep serving: one bad request shouldn't cost the whole model load.
            traceback.print_exc()
            send(event="error", stage="render", message=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
