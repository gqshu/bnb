# bnb

Backend for an EEG-adaptive binaural beats product. Scope is **down-regulation only**
(Recharge / Wind Down / Sleep Prep) — see [docs/](docs/) for the feasibility assessment
and the product design decisions this code is built against.

## Audio design constraints

These come straight from the product docs and are what the code enforces:

- **Carrier** is a fixed pure sine at ~400–440 Hz, held constant for the whole session.
  Below ~200 Hz and above ~900 Hz the beat percept weakens badly.
- **Beat frequency Δ** is the target state, not a free parameter. All our modes sit at
  ≤ 14 Hz. Above ~30 Hz the two tones separate into distinct pitches and the beat
  percept collapses, so nothing we ship goes near it. `MAX_BEAT_HZ` is 40 only so the
  demo portal can explore the gamma range — treat > 30 Hz as a research setting.
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
uv run scripts/generate_wavs.py                          # all modes, 60 s each, binaural
uv run scripts/generate_wavs.py sleep_prep --duration 300
uv run scripts/generate_wavs.py wind_down --waveform square
uv run scripts/generate_wavs.py --stimulus monaural
uv run scripts/generate_wavs.py --stimulus isochronic recharge --duration 30
```

## Background media library

`src/bnb/background.py` holds the taxonomy from
[docs/background_music.md](docs/background_music.md): a `(substrate, style)` *grid
cell* resolves to a prose prompt, an ElevenLabs-shaped composition plan, a seed, and
a per-track spec record. Alongside the grid, `SPECIAL_GROUPS` holds keyword-driven
*special cells* — categories with no substrate/style axis, just a keyword — starting
with `natural_sounds` (rain, ocean, wind, stream, forest, night, chimes). Both kinds
share one spec schema (`kind`, plus `style`/`substrate` or `group`/`keyword`), so
storage, catalog and rendering treat them identically.

Two providers can turn a spec into audio, selected with `render_background.py
--provider`: **Stable Audio 3** (self-hosted, `--provider stable_audio` — the
default; `src/bnb/stable_audio.py` — see that module's docstring for why it shells
out to a sibling `stable-audio-3` checkout instead of running in-process) and
**ElevenLabs** (hosted, paid, `--provider elevenlabs`). A spec's stored
prompt already reads as plain prose that both providers accept; `prompt_for_provider`
adapts it further per provider — ElevenLabs gets its structure from the composition
plan, so the prompt passes through unchanged, while Stable Audio 3 has no structured
composition input, so it gets an AudioSparx metadata-tag suffix instead (the
vocabulary its text encoder was trained on — see `scripts/try_stable_audio.py`).

### Asset repository: one subdirectory per category cell

`src/bnb/assets.py` is the low-level, cell-aware file layer. Every category —
`(style, substrate)` for the grid, `(group, keyword)` for a special group — gets its
own subdirectory holding every seed/provider/variant rendered for it, so a category
can be browsed, bulk-deleted, or `ls`'d as a unit:

```
assets/
  specs/<style>/<substrate>/<track_id>.json     grid spec (§3); the source of truth
  specs/<group>/<keyword>/<track_id>.json       special-cell spec, same shape
  tracks/<style>/<substrate>/<track_id>.wav     rendered audio (git-ignored: large, costs credits/compute)
  tracks/<group>/<keyword>/<track_id>.wav
  catalog.json                                  generated index of compact descriptors
```

`track_id` (e.g. `buddhist_meditative_drone_seed81657`) stays the one flat,
globally-unique identifier everywhere outside `assets.py` — the stream engine, the
API, the CLIs — only its on-disk location is nested by cell.

`src/bnb/catalog.py`'s `CategoryManager` is the interface everything else actually
uses — the stream engine, the demo API, and both CLIs below go through it rather than
poking `assets` functions directly:

- **add** — `add_spec(spec)`, `add_render(spec, data, ...)` (bytes, e.g. ElevenLabs),
  `attach_render(spec, path, ...)` (a file a provider already wrote, e.g. Stable Audio)
- **delete** — `delete(track_id)`, `delete_cell((outer, inner))` (wipe a whole category)
- **search** — `search(style=..., substrate=..., group=..., keyword=..., kind=...,
  cell=..., rendered=..., provider=...)`, any combination, unset filters ignored
- **pick** — `pick(**filters, exclude=track_id)` — one random match, excluding a
  track_id when there's an alternative (what shuffle playback uses)

`CategoryManager(root=...)` points a whole session at a different asset repository
(e.g. a `tmp_path` in tests) instead of the real `assets/`.

### Planning and rendering are two separate scripts

Spec management is offline and free (no API, no key), so you preview the catalog
before spending on a paid model or a slow local render — and the same specs can be
rendered by either provider, or re-rendered by the other one later. A spec is
render-independent; the renderer fills its `render` block (provider, model, format,
provenance, audio file).

`scripts/plan_background.py` — write/inspect specs (free, offline):

```bash
uv run scripts/plan_background.py --list             # print the taxonomy axes, sample set, special groups
uv run scripts/plan_background.py --coverage         # print the substrate × style count matrix
uv run scripts/plan_background.py                    # write the curated sample set
uv run scripts/plan_background.py buddhist_meditative:drone --duration 90
uv run scripts/plan_background.py natural_sounds:rain natural_sounds:ocean   # special cells
uv run scripts/plan_background.py --only-new         # skip pairs whose spec already exists
uv run scripts/plan_background.py --rebuild-catalog  # rebuild catalog.json from specs
```

Grid pairs are `STYLE:SUBSTRATE`; special-cell pairs are `GROUP:KEYWORD` — same
syntax, disambiguated by whether the left side names a style or a special group.

Grow the library by coverage guide — place the next specs so the substrate × style
grid fills evenly (which also levels the per-substrate and per-style marginals; special
cells sit outside this grid by design, so they're unaffected). Each repeat of a cell
advances the seed (`variant`), matching the doc's "3–5 seeds per cell" (§5):

```bash
uv run scripts/plan_background.py --fill 10          # next 10 specs, evenly distributed
uv run scripts/plan_background.py --per-cell 2       # top every cell up to 2 tracks
uv run scripts/plan_background.py --fill 6 --styles buddhist_meditative,neutral   # restrict the grid
```

`scripts/render_background.py` — turn specs into audio; provider defaults to
`stable_audio`:

```bash
uv run scripts/render_background.py                  # render everything missing audio (Stable Audio 3)
uv run scripts/render_background.py --dry-run        # preflight the provider + list what would render
uv run scripts/render_background.py buddhist_meditative_drone_seed81657   # render specific track(s)
uv run scripts/render_background.py --force <track_id>    # re-render an existing track
uv run scripts/render_background.py --sa3-backend mlx --sa3-model medium  # the 1.4B model, Metal-backed
uv run scripts/render_background.py --sa3-cfg 5      # turn on classifier-free guidance

export ELEVENLABS_API_KEY=...
uv run scripts/render_background.py --provider elevenlabs
uv run scripts/render_background.py --provider elevenlabs --output-format mp3_44100_192 --model-id music_v1
```

Every run — `--dry-run` included — starts with a **preflight check** that the
selected provider is actually usable (the SA3 CLI is discoverable / the API key is
set) and exits with setup instructions if not, before touching any spec. That also
covers the case where every target track is already rendered and `--dry-run` would
otherwise never reach the provider at all.

Rendering is credit-safe: specs that already have audio are skipped unless you pass
`--force`. The Stable Audio path needs the sibling `stable-audio-3` checkout set up
(see `src/bnb/stable_audio.py`'s docstring — torch has no Python 3.14 wheels, so it
runs out of process); `scripts/try_stable_audio.py` is a spec-free smoke test of that
path alone (no catalog writes) if you just want to ear-check the model.
`scripts/check_background.py` runs basic listenability checks (silence, clipping,
noise) over rendered audio — point it at `assets/tracks/` (the default) or any
directory, recursively.

## Live stream service

A single-stream backend service that generates the binaural beat live and streams it
over HTTP, with a demo web portal. One stream, one fixed port (`bnb.server.PORT`,
default 8000). The **backend** owns the stream — it renders phase-continuous tones in
small chunks and mixes an optional looped background track — so live spec edits are
heard within a chunk. The browser just plays `/stream.wav` and PATCHes the spec.

```bash
uv run scripts/serve.py        # then open http://127.0.0.1:8000
```

| method | path | purpose |
|---|---|---|
| GET | `/` | the demo portal |
| GET | `/api/stream` | current state (running, beat, background, volumes) |
| POST | `/api/stream/start` | start; `beat` and `background_id` both optional (no beat ⇒ silence/background only) |
| POST | `/api/stream/stop` | stop |
| PATCH | `/api/stream/spec` | live-update `beat` (or `null` to drop it), `background_id`, `background_volume` |
| GET | `/api/backgrounds` | catalog tracks; only `rendered` ones are playable |
| GET | `/stream.wav` | the live audio (open-ended WAV, paced to real time) |

The stream runs at 44.1 kHz to match the background masters (`pcm_44100`), so no
resampling is needed in the common case. Headphones required — the beat only exists
across the two ears.

**Background: shuffle or repeat.** Either way the background plays until the stream is
stopped; the only difference is what comes next.

- **Shuffle** — `background_id: "shuffle"` plays the rendered library on infinite
  random shuffle, handing each finished track off to another random one. State reports
  `shuffle: true` plus the `background_id` audible right now. This is the default for a
  session driver.
- **Repeat** — a pinned `background_id` loops that one track forever (`shuffle: false`).

Both transitions are crossfaded rather than spliced: the hand-off fires while the
current track is inside its final fade-out window, so its real tail overlaps the next
head (for repeat, its own head). The incoming side fades in about twice as fast as the
outgoing fades out, so the next track connects promptly. `background_id: null` is
silence.

**Sham condition.** `beat_hz` may be exactly `0` on the stream endpoints: both ears
then get the same carrier and no beat exists, while the carrier stays audible. That is
the acoustically-matched control arm for the EEG pilot (`docs/control.md` §2.5) —
identical to an active session in everything but Δ. This is deliberately looser than
`tone.render_binaural`, which still rejects 0 because a 0 Hz *binaural* WAV is not a
binaural tone at all; the sham exists only as a live stream state. Passing
`beat: null` is a different thing — it removes the carrier too, leaving background only.

**Presentation mode.** A beat's `mode` selects how the stimulus reaches the ears (see
`docs/Monaural_and_Isochronic_Beats_Implementation.md` for the full design):

- `dichotic` (default) — carrier in the left ear, carrier + Δ in the right: the binaural
  beat, which exists only across the two ears (no physical amplitude modulation).
- `diotic` — both tones summed identically into both ears, so the Δ beat is a real
  acoustic modulation present in each ear. This is the **ASSR/ITPC control**: the sound
  is physically comparable to dichotic but carries no binaural (neural-construct) beat
  to entrain to, so an elevated 40 Hz phase-locking under dichotic but not diotic is
  evidence the response is neural. The channels are bit-identical in diotic.
- `monaural` — the product-facing name for the exact same summed-tone signal `diotic`
  renders. Binaural is the better *listening* experience; monaural is a stronger EEG
  probe (real acoustic beat, not just a neural construct) and a reasonable middle
  ground for product audio. Kept as a separate mode name (not a rename of `diotic`) so
  the ASSR-control framing above stays intact.
- `isochronic` — a single carrier (`carrier_hz`, usable range 100–500 Hz, default 250)
  amplitude-gated on/off `beat_hz` times per second. Mono by nature (identical L/R) and
  the strongest, spectrally cleanest entrainment drive of the three. Three extra
  parameters apply only to this mode:
  - `depth` (0–1, default 1) — how far the "off" phase drops; 1.0 = full silence.
  - `duty` (0.1–0.9, default 0.5) — fraction of each cycle the tone is on.
  - `ramp_ms` (2–10 ms, default 5) — raised-cosine fade on every on/off transition.
    **Mandatory**, not cosmetic: a hard gate clicks audibly and splatters harmonics of
    `beat_hz` into the EEG-relevant band, muddying the spectral readout the probe
    exists to produce.

  **Isochronic can never be combined with a background track** — the pulsing *is* the
  entrainment signal, and any soundscape underneath reduces its effective modulation
  depth. The engine rejects that combination as soon as either side of the spec is set
  (a `400` from `/api/stream/start` or `/api/stream/spec`, before any audio renders),
  not with a per-sample check in the render loop. Binaural and monaural have no such
  restriction — the beat survives a background layered on top either way.

The offline renderer (`bnb.tone`) has one function per stimulus —
`render_binaural`/`render_monaural`/`render_isochronic` — each returning float32
stereo. `render_isochronic` deliberately has no background parameter at all: it's the
no-background "measurement probe" path from the doc, not a live-mixed product track.

### Control client

`bnb.client.StreamClient` wraps the endpoints for driving the stream from Python,
and `scripts/control.py` is a CLI over it (the service must be running):

```bash
uv run scripts/control.py backgrounds                 # list background track meta
uv run scripts/control.py current                     # get the current background
uv run scripts/control.py play --background neutral_noise_seed50801 --beat 10 --volume 0.3
uv run scripts/control.py play --beat 10 --mode monaural
uv run scripts/control.py play --beat 40 --mode isochronic --carrier 250 --depth 1 --duty 0.5
uv run scripts/control.py volume 0.5                   # change the beat volume
uv run scripts/control.py freq 7.83                    # change the beat frequency
```

`play` starts the stream if it's stopped, otherwise changes it live (the background
crossfades). Since the server's PATCH replaces the whole beat, the `volume`/`freq`
helpers read the current beat, change one field, and send it back.

## Layout

- `src/bnb/tone.py` — offline binaural/monaural/isochronic tone rendering and WAV output
- `src/bnb/background.py` — background-media taxonomy (substrate × style grid + special groups) and prompts
- `src/bnb/assets.py` — the cell-aware asset file layer (specs, tracks, catalog)
- `src/bnb/catalog.py` — `CategoryManager`: add/delete/search/pick over the asset repository
- `src/bnb/stable_audio.py` — self-hosted Stable Audio 3 render backend (out-of-process)
- `src/bnb/qc.py` — listenability checks (silence, clipping, noise) for rendered audio
- `src/bnb/stream.py` — the live stream engine (phase-continuous beat + background mix)
- `src/bnb/server.py` — FastAPI service and endpoints
- `src/bnb/client.py` — control client for the stream service
- `src/bnb/web/index.html` — the demo portal
- `scripts/plan_background.py` — write/inspect background specs (offline, free)
- `scripts/render_background.py` — render specs into audio (ElevenLabs or Stable Audio 3)
- `scripts/check_background.py` — QC pass over rendered tracks
- `scripts/try_stable_audio.py` — spec-free Stable Audio 3 smoke test
- `scripts/serve.py` — run the stream service
- `scripts/control.py` — command-line control client
- `scripts/` — dev utilities, not shipped
- `tests/` — test suite
- `docs/` — product and feasibility docs
- `assets/` — background-media asset repo, one subdirectory per category cell (specs + catalog tracked, audio ignored)
- `run/` — binaural ear-check renders, git-ignored
