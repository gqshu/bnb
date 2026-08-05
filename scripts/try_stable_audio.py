"""Render Stable Audio 3 clips from sample prompts — the local-model smoke test.

Each run draws keywords at random from POSITIVE_KEYWORD_MAP, appends the keyword's
description to the base prompt as "Featuring ...", and names the clip after it — so
one run gives you a spread of textures to ear-check rather than one sample.

    uv run scripts/try_stable_audio.py                              # 3 random keywords, 30 s each
    uv run scripts/try_stable_audio.py --count 5
    uv run scripts/try_stable_audio.py --keywords rain,harp         # pick them yourself
    uv run scripts/try_stable_audio.py --body melodic               # solo line, not a static bed
    uv run scripts/try_stable_audio.py --keywords harp --no-tags    # A/B the AudioSparx tags
    uv run scripts/try_stable_audio.py --backend mlx --model medium # the 1.4B model, Metal-backed
    uv run scripts/try_stable_audio.py --cfg 5 --keywords ocean     # steer harder toward the prompt
    uv run scripts/try_stable_audio.py -p "warm tape drone, no rhythm" --duration 60
    uv run scripts/try_stable_audio.py --seed -1                    # -1 = new random seed each run
    uv run scripts/try_stable_audio.py --model small-music-base --steps 50
    uv run scripts/try_stable_audio.py --print-command              # show argv, render nothing

Deliberately independent of the asset repository: no specs, no catalog, no
track_ids, nothing written under assets/. It exercises only the sa3 path — CLI
discovery, subprocess, weights, decode — so a failure here is the model side, not
asset management. Clips land in run/sa3/ (git-ignored).

First run downloads the 2.3 GB small-music checkpoint from a gated HF repo, so
`hf auth login` must have run. See tests/test_stable_audio.py if that download crawls.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import soundfile as sf

from bnb import stable_audio

RUN_DIR = Path(__file__).resolve().parent.parent / "run" / "sa3"

POSITIVE_KEYWORD_MAP = {
    "raindrop":  "soft distant rainfall, gentle raindrops on leaves, no thunder",
    "rain":      "steady soft rain, muffled and distant, soothing",
    "wind":      "faint wind through trees, airy and slow, low whooshing",
    "ocean":     "slow distant ocean waves, long gentle swells, calm surf",
    "stream":    "quiet flowing stream, soft trickling water",
    "forest":    "distant forest ambience, faint birdsong, leaves rustling softly",
    "buddha":    "meditative tibetan singing bowls, deep resonant hum, sparse and slow",
    "singing_bowl": "resonant crystal singing bowls, long sustained overtones",
    "piano":     "sparse soft felt piano, slow single notes, long decay, no melody hook",
    "pad":       "warm analog synth pad, slow attack, sustained chords",
    "strings":   "soft sustained string ensemble, slow legato, low register",
    "harp":      "gentle plucked harp, sparse arpeggios, slow and spacious",
    "flute":     "soft breathy bamboo flute, long sustained tones, sparse",
    "drone":     "deep continuous ambient drone, subtle slow movement",
    "chimes":    "distant wind chimes, sparse and soft, gentle metallic shimmer",
    "night":     "quiet night ambience, faint crickets, distant and low",
    "space":     "vast cosmic ambient texture, deep slow evolving pads",
}

# Keywords the model can hear as a played instrument, so they earn an `Instruments:`
# tag. The rest are ambiences — the "Featuring ..." clause is all they get, since
# AudioSparx has no documented field for them and inventing one is off-distribution.
INSTRUMENT_KEYWORDS = {
    "buddha": "Singing Bowls",
    "singing_bowl": "Singing Bowls",
    "piano": "Piano",
    "pad": "Synthesizer",
    "strings": "Strings",
    "harp": "Harp",
    "flute": "Flute",
    "drone": "Synthesizer",
    "chimes": "Chimes",
}

# Two down-regulation bodies, both shaped like bnb.background.build_prompt output
# (docs/background_music.md §4): "drone" is the static substrate, "melodic" is the
# melodic_instrument substrate — a solo line instead of a bed. The static-harmony and
# no-rhythm clauses in "drone" are what suppress melody, so a keyword like `piano`
# cannot produce a tune under it however the keyword is phrased.
BODY_PROMPTS = {
    "drone": (
        "Instrumental ambient soundscape for deep relaxation. Sustained pad and low drone "
        "tones with no rhythm. Tempo arrhythmic, no pulse, very low energy, very soft "
        "dynamics. Warm and dark timbre, low spectral brightness. Static harmony, very "
        "sparse texture, low register. No vocals, no percussion hits, no sudden "
        "transitions. Seamless, calm, continuous."
    ),
    "melodic": (
        "Instrumental ambient soundscape for deep relaxation. A sparse solo instrument "
        "line, gentle and unhurried, slow legato phrases with long rests between them. "
        "Around a 50 BPM feel, low energy, very soft dynamics. Warm and dark timbre, low "
        "spectral brightness. Simple consonant harmony, sparse texture, mid register. No "
        "vocals, no percussion hits, no sudden transitions. Seamless, calm, continuous."
    ),
}
DEFAULT_BODY = "drone"

# Metadata tags in the AudioSparx form the model was trained on
# (stable-audio-3/docs/guides/prompting.md, "Helpful AudioSparx Tags"). Documented
# fields only — TrackType/VocalType reportedly lift coherence on their own.
BASE_TAGS = "TrackType: Music, VocalType: Instrumental, Genre: Ambient"

SAMPLE_NEGATIVE_PROMPT = "bright, harsh, energetic, fast, distorted, drums, buildup, vocals"
SAMPLE_SEED = 42  # fixed so repeat runs are comparable; --seed -1 for a fresh one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--prompt", help="base prompt, overriding --body; the keyword is appended to it")
    parser.add_argument(
        "--body",
        default=DEFAULT_BODY,
        choices=sorted(BODY_PROMPTS),
        help=f"which substrate to prompt for (default: {DEFAULT_BODY}); melodic asks for a solo line",
    )
    parser.add_argument(
        "--no-tags",
        dest="tags",
        action="store_false",
        help="drop the AudioSparx metadata tags, to A/B whether they help adherence",
    )
    parser.add_argument("--negative-prompt", default=SAMPLE_NEGATIVE_PROMPT)
    parser.add_argument("--count", type=int, default=3, help="how many keywords to render (default: 3)")
    parser.add_argument("--keywords", help=f"comma-separated, instead of a random draw: {','.join(POSITIVE_KEYWORD_MAP)}")
    parser.add_argument("--duration", type=int, default=30, help="seconds (default: 30)")
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED, help=f"default: {SAMPLE_SEED}; -1 = random")
    parser.add_argument("--model", default=stable_audio.DEFAULT_MODEL, help=f"default: {stable_audio.DEFAULT_MODEL}")
    parser.add_argument(
        "--backend",
        default=stable_audio.DEFAULT_BACKEND,
        choices=stable_audio.BACKENDS,
        help="torch (upstream CLI) or mlx (Apple Silicon; the only way to run --model medium here)",
    )
    parser.add_argument("--steps", type=int, help="default: 8 post-trained, 50 for -base checkpoints")
    parser.add_argument("--cfg", type=float, help="guidance scale; turns the negative prompt on (try 3-7)")
    parser.add_argument("-o", "--output", help="output WAV (default: run/sa3/sample_<keyword>_seed<seed>.wav)")
    parser.add_argument("--print-command", action="store_true", help="print the argv and exit")
    return parser.parse_args()


def select_keywords(args: argparse.Namespace) -> list[str]:
    """The keywords to render: an explicit list, or a random draw of ``--count``."""
    if args.keywords:
        chosen = [k.strip() for k in args.keywords.split(",") if k.strip()]
        unknown = [k for k in chosen if k not in POSITIVE_KEYWORD_MAP]
        if unknown:
            raise SystemExit(f"unknown keyword(s): {', '.join(unknown)}")
        return chosen
    if not 1 <= args.count <= len(POSITIVE_KEYWORD_MAP):
        raise SystemExit(f"--count must be between 1 and {len(POSITIVE_KEYWORD_MAP)}")
    # Sample without replacement so one run never renders the same keyword twice.
    return random.sample(list(POSITIVE_KEYWORD_MAP), args.count)


def keyword_prompt(base_prompt: str, keyword: str, tags: bool = True) -> str:
    """Base body + the keyword as prose, then the metadata tags the model was trained on.

    Tags go last and name the keyword a second time: there is no prompt-weighting
    syntax in Stable Audio 3 (the text encoder is T5Gemma, no attention emphasis
    parser), so naming a thing in the model's own metadata vocabulary is the closest
    lever to turning it up.
    """
    prompt = f"{base_prompt} Featuring {POSITIVE_KEYWORD_MAP[keyword]}."
    if not tags:
        return prompt
    instrument = INSTRUMENT_KEYWORDS.get(keyword)
    suffix = f", Instruments: {instrument}" if instrument else ""
    return f"{prompt} {BASE_TAGS}{suffix}"


def output_path(args: argparse.Namespace, keyword: str) -> Path:
    """Where the clip lands.

    Model, body and keyword are all in the name: comparing small-music against medium
    is the whole point of having two backends, and they'd otherwise overwrite each other.
    """
    if args.output:
        given = Path(args.output)
        return given.with_name(f"{given.stem}_{keyword}{given.suffix or '.wav'}")
    return RUN_DIR / f"sample_{args.model}_{args.body}_{keyword}_seed{args.seed}.wav"


def main() -> None:
    args = parse_args()
    keywords = select_keywords(args)
    base_prompt = args.prompt or BODY_PROMPTS[args.body]

    for n, keyword in enumerate(keywords, start=1):
        # The minimal shape bnb.stable_audio reads — the same keys a real spec carries,
        # without going through assets.load_spec.
        spec = {
            "prompt": keyword_prompt(base_prompt, keyword, tags=args.tags),
            "negative_prompt": args.negative_prompt,
            "seed": args.seed,
            "duration_s": args.duration,
        }
        out = output_path(args, keyword)

        try:
            cmd = stable_audio.build_command(
                spec, out, model=args.model, steps=args.steps, cfg=args.cfg, backend=args.backend
            )
        except (ValueError, RuntimeError) as exc:  # over the model's max length, or no CLI found
            raise SystemExit(str(exc))

        if args.print_command:
            print(" ".join(repr(part) if " " in part else part for part in cmd))
            continue

        # flush: the child writes straight to the terminal, so unflushed prints land after it.
        print(
            f"[{n}/{len(keywords)}] {keyword} | {args.body} body | {args.model} on {args.backend} | "
            f"{args.duration}s | seed {args.seed}"
        )
        print(f"prompt: {spec['prompt']}\n", flush=True)

        start = time.perf_counter()
        path = stable_audio.render(
            spec, out, model=args.model, steps=args.steps, cfg=args.cfg, backend=args.backend
        )
        elapsed = time.perf_counter() - start

        info = sf.info(path)
        audio, _ = sf.read(path, dtype="float32")
        print(
            f"\n{path}\n"
            f"  {info.duration:.1f}s  {info.samplerate} Hz  {info.channels}ch  {info.subtype_info}\n"
            f"  peak {abs(audio).max():.3f}  |  rendered in {elapsed:.1f}s (includes model load)\n"
        )


if __name__ == "__main__":
    main()
