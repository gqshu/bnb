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
signature into the Eleven Music prompt, composition plan, seed, and per-track
spec record. Two thin CLIs sit over it and the **asset repository**
(`src/bnb/assets.py`): `scripts/plan_background.py` writes specs offline, and
`scripts/render_background.py` renders them into audio.

The asset repo is a *flat, tagged store* rather than a category tree — the bandit
selects on categorical tags (substrate, style) plus a continuous MER vector, which
a hierarchy can't express:

```
assets/
  specs/<track_id>.json    per-track metadata (§3); the source of truth (git-tracked)
  tracks/<track_id>.wav    rendered audio master (git-ignored: large, costs credits)
  catalog.json             generated index of compact descriptors for selection
```

The selection workflow reads only `catalog.json`; it never opens every spec.

**Planning and rendering are two separate scripts.** Spec management is offline
and free (no API, no key), so you preview the catalog before spending on a paid
model — and the same specs can be rendered by a local model later. A spec is
render-independent; the renderer fills its `render` block (provider, model,
format, provenance, audio file).

`scripts/plan_background.py` — write/inspect specs (free, offline):

```bash
uv run scripts/plan_background.py --list             # print the taxonomy axes + sample set
uv run scripts/plan_background.py --coverage         # print the substrate × style count matrix
uv run scripts/plan_background.py                    # write the curated sample set
uv run scripts/plan_background.py buddhist_meditative:drone --duration 90
uv run scripts/plan_background.py --only-new         # skip pairs whose spec already exists
uv run scripts/plan_background.py --rebuild-catalog  # rebuild catalog.json from specs
```

Grow the library by coverage guide — place the next specs so the substrate × style
grid fills evenly (which also levels the per-substrate and per-style marginals).
Each repeat of a cell advances the seed (`variant`), matching the doc's "3–5 seeds
per cell" (§5):

```bash
uv run scripts/plan_background.py --fill 10          # next 10 specs, evenly distributed
uv run scripts/plan_background.py --per-cell 2       # top every cell up to 2 tracks
uv run scripts/plan_background.py --fill 6 --styles buddhist_meditative,neutral   # restrict the grid
```

`scripts/render_background.py` — turn specs into audio (paid / heavyweight):

```bash
export ELEVENLABS_API_KEY=...
uv run scripts/render_background.py --dry-run        # list what would render, no API call
uv run scripts/render_background.py                  # render every spec still missing audio
uv run scripts/render_background.py buddhist_meditative_drone_seed81657   # render specific track(s)
uv run scripts/render_background.py --force <track_id>    # re-render an existing track
```

Rendering is credit-safe: specs that already have audio are skipped unless you pass
`--force`. Provider is pluggable (`--provider`); only ElevenLabs is wired today. These
samples are for the Stage 1 provider bake-off; the measured-MER extraction pipeline
lands in Stage 2.

## Layout

- `src/bnb/tone.py` — binaural tone rendering and WAV output
- `src/bnb/background.py` — background-media substrate × style taxonomy and prompts
- `src/bnb/assets.py` — the background asset repository (specs, tracks, catalog)
- `scripts/plan_background.py` — write/inspect background specs (offline, free)
- `scripts/render_background.py` — render specs into audio via a provider
- `scripts/` — dev utilities, not shipped
- `tests/` — test suite
- `docs/` — product and feasibility docs
- `assets/` — background-media asset repo (specs + catalog tracked, audio ignored)
- `run/` — binaural ear-check renders, git-ignored
