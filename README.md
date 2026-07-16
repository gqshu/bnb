# bnb

Backend for an EEG-adaptive binaural beats product. Scope is **down-regulation only**
(Recharge / Wind Down / Sleep Prep) — see [docs/](docs/) for the feasibility assessment
and the product design decisions this code is built against.

## Audio design constraints

These come straight from the product docs and are what the code enforces:

- **Carrier** is a fixed pure sine at ~400–440 Hz, held constant for the whole session.
  Below ~200 Hz and above ~900 Hz the beat percept weakens badly.
- **Beat frequency Δ** is the target state, not a free parameter. All our modes sit at
  ≤ 14 Hz. Δ must stay under 30 Hz or the two tones separate into distinct pitches.
- **Only one ear moves.** Left gets the carrier `f`, right gets `f + Δ`. The user hears a
  steady pitch while the beat slows down underneath it.
- **Loudness is matched between ears.** An interaural imbalance is heard as the sound
  lateralizing to one side and breaks immersion.
- **Pure sine carriers.** Harmonic-rich carriers muddy the beat percept. `render_binaural`
  takes a `waveform` parameter (`sine`, `triangle`, `square`, `sawtooth`) for comparison
  work, but it defaults to `sine` and that is what we ship.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync --extra dev     # create .venv and install deps
uv run pytest           # run the test suite
```

## Listening to the output

`scripts/generate_wavs.py` renders constant-beat samples into `run/` (git-ignored) so you
can put headphones on and check them. These are not the product — the real session steps
the beat down from the user's measured EEG baseline.

```bash
uv run scripts/generate_wavs.py                          # all modes, 60 s each
uv run scripts/generate_wavs.py sleep_prep --duration 300
uv run scripts/generate_wavs.py wind_down --waveform square
```

## Background media library

`src/bnb/background.py` holds the substrate × style taxonomy from
[docs/background_music.md](docs/background_music.md): it turns a `(substrate, style)`
signature into the Eleven Music prompt, composition plan, and per-track metadata
record. `scripts/generate_background.py` is a thin CLI over it that calls ElevenLabs
and writes each track plus a JSON sidecar into `run/background/`.

```bash
export ELEVENLABS_API_KEY=...
uv run --extra media scripts/generate_background.py --list      # print the catalog
uv run --extra media scripts/generate_background.py --dry-run   # write prompts/metadata, no API call
uv run --extra media scripts/generate_background.py             # curated sample set, 60 s each
uv run --extra media scripts/generate_background.py buddhist_meditative:drone --duration 90
```

These samples are for the Stage 1 provider bake-off; the WAV master and measured-MER
extraction pipeline land in Stage 2.

## Layout

- `src/bnb/tone.py` — binaural tone rendering and WAV output
- `src/bnb/background.py` — background-media substrate × style taxonomy and prompts
- `scripts/` — dev utilities, not shipped
- `tests/` — test suite
- `docs/` — product and feasibility docs
- `run/` — generated audio, git-ignored
