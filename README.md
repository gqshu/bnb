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
vocabulary its text encoder was trained on — see `scripts/try_stable_audio.py`):
`TrackType: Music, …` for grid cells, `TrackType: SFX` for special ones.

A prompt carries three things beyond the taxonomy axes, all of them because a bed you
sit with for an hour has to stay interesting without ever becoming eventful: a **motion**
clause (slow swells, faint timbral drift, "never builds, resolves, or arrives anywhere"),
the style's **character** — the room, the gear, the grain, e.g. lofi's tape wow and vinyl
noise floor against buddhist_meditative's stone hall and audible air — and a **density**
limit. The axes stay biased low regardless; motion and character move the *texture*, not
the MER coordinate, which is what the down-regulation scope actually constrains.

Density is stated countably ("at most one or two things sounding at any moment") because
"very sparse" reads to these models as timbre, and it is stated in the *positive* prompt
because the negative one only reaches the model under guidance (`--sa3-cfg`), which the
distilled checkpoints run without. Which limit a sound gets depends on how it's made —
`event_driven` on the substrate or keyword. Sounds built from separate events (singing
bowls, a solo line, forest birds, chimes) are told to space them out; continuous beds
(drones, noise, rain, crickets) are told to stay even. Getting that backwards is worse
than saying nothing: asking a cricket wash for "long stretches of near-stillness" doesn't
quiet it, it breaks the smooth bed into discrete chirps with gaps.

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
- **sync** — `rebuild()` re-derives `catalog.json` from the spec tree (normalizing
  stray spec locations and refusing ambiguous ones); `spec_ids()` is disk truth even
  when the catalog is stale, `orphan_tracks()` lists audio whose spec is gone

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
uv run scripts/plan_background.py natural_sounds            # every keyword in the group
uv run scripts/plan_background.py natural_sounds:rain natural_sounds:ocean   # single special cells
uv run scripts/plan_background.py --rebuild-catalog  # rebuild catalog.json from specs
```

A target is a grid cell (`STYLE:SUBSTRATE`), a special cell (`GROUP:KEYWORD` — same
syntax, disambiguated by whether the left side names a style or a special group), or a
bare special group (`GROUP`), which plans every keyword in it. `natural_sounds` is the
only group for now.

**Planning never overwrites.** Seeds are deterministic, so a target whose spec is
already on disk is reported and skipped — runs are idempotent and safe to repeat.
Replanning is a deliberate, manual act: delete the spec file (or the whole cell
directory) and run again. That's also how you change an existing spec's `--duration`
or pick up a prompt-template edit. `catalog.json` is derived state, rebuilt from the
spec tree at the start of every run, so a hand-deleted spec is simply gone; misplaced
specs are moved back into their cell and duplicate `track_id`s are a hard error rather
than a silent winner. Deleting a spec whose audio was already rendered leaves the
`.wav` orphaned — the script warns, because replanning that cell reproduces the same
`track_id` and would adopt the stale audio.

Grow the library by coverage guide — place the next specs so the cells fill evenly.
On the grid that also levels the per-substrate and per-style marginals. Each repeat of
a cell advances the seed (`variant`), matching the doc's "3–5 seeds per cell" (§5):

```bash
uv run scripts/plan_background.py --fill 10          # next 10 specs, evenly distributed
uv run scripts/plan_background.py --per-cell 2       # top every cell up to 2 tracks
uv run scripts/plan_background.py --fill 6 --styles buddhist_meditative,neutral   # restrict the grid

uv run scripts/plan_background.py --per-cell 3 --groups natural_sounds   # 3 seeds per keyword
uv run scripts/plan_background.py --fill 4 --groups natural_sounds
```

`--groups` points the same guides at special cells instead of the grid — the two
taxonomies share no axis, so a run fills one or the other (`--groups` with
`--substrates`/`--styles` is an error). It's also the **only** way to plan more than one
track per keyword: a plain `GROUP:KEYWORD` target always resolves to the cell's first
seed. `--coverage` reports both taxonomies — the substrate × style matrix and a
per-keyword row for each special group — unless an axis filter narrows it to one:

```
                       drone melodic   field   noise   bowls       Σ
neutral                    6       6       6       6       6      30
...
Σ                         30      30      30      30      30     150

Special groups (outside the grid):
                  rain  ocean   wind stream forest  night chimes      Σ
natural_sounds       3      3      3      2      2      2      2     17
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
uv run scripts/render_background.py --max-retry 5    # try harder on a cell that keeps failing QC
uv run scripts/render_background.py --no-qc          # keep whatever comes back, unchecked
uv run scripts/render_background.py --no-worker      # one process per track, reloading each time

export ELEVENLABS_API_KEY=...
uv run scripts/render_background.py --provider elevenlabs
uv run scripts/render_background.py --provider elevenlabs --output-format mp3_44100_192 --model-id music_v1
```

Every run — `--dry-run` included — starts with a **preflight check** that the
selected provider is actually usable (the SA3 CLI is discoverable / the API key is
set) and exits with setup instructions if not, before touching any spec. That also
covers the case where every target track is already rendered and `--dry-run` would
otherwise never reach the provider at all.

**The default engine is `medium` on MLX.** It's the only checkpoint that covers both
halves of the taxonomy — `small-music` cannot render sound effects at all — and on the
music cells it measures visibly livelier: frame-level variation roughly doubled on the
same seed and prompt (5.3 → 9.3 dB on singing bowls, 5.6 → 10.3 dB on the noise bed).
It costs ~3× the sampling time (7.5s vs 2.4s for a 60s track) and ~4 GB peak RAM.

On Apple Silicon that means MLX: `medium` wants Flash Attention 2 under torch and the
prebuilt wheels are CUDA/Linux only, so it runs Metal-backed through `optimized/mlx/sa3`
at ~8× realtime, using the `dit_medium_f16.npz` weights from the `stable-audio-3-optimized`
bundle.

**On a CUDA box, `--sa3-backend torch` is the only flag you need.** The SFX backend
follows the run's backend unless overridden (only the *model* differs per cell), so one
flag moves the whole library. Flash Attention is worth installing but is not a hard
requirement: `medium`'s config sets `sliding_window`, and `transformer.py` degrades
through a documented cascade when `flash_attn` won't import — flex-attention band mask,
then chunked-halo masked SDPA ("math-equivalent, ~30× faster than tier 4"), then a full
N×N masked SDPA that the vendor itself calls high-memory. Same audio, worse throughput,
with the tier you land on depending on whether `torch.compile` works in your image. The
torch path also unlocks the resident-model Worker, which matters more for medium than it
did for small-music — one checkpoint load for the whole batch instead of one per track.

**The model is loaded once per run where the backend allows it.** Loading `small-music`
under torch takes eight seconds against about one second of sampling, and the upstream
CLI pays it on every invocation — so that path runs against a resident model instead.
`bnb.stable_audio.Worker` starts `src/bnb/sa3_worker.py` with the sibling checkout's
interpreter (bnb's own venv can't import torch), keeps it open for the whole run, and
sends one JSON request per track over a pipe. Each request is an independent seeded
`generate` call, so the audio is what the one-shot CLI would have produced. It's `torch`
only — the MLX entry point's sampling loop lives inline in its `main()` with no importable
generate step, and a divergent copy of someone else's sampler is a worse trade than
reloading — so the default MLX path reloads per track and absorbs the ~1s cost. Reach for
it with `--sa3-model small-music --sa3-backend torch` when you want speed over quality.

**Each spec goes to a checkpoint that can render it.** Special cells are field
recordings, not music, and `small-music` has *no* SFX capability — the vendor's own
compatibility table says so, and the first `natural_sounds` renders proved it: "rain"
came back with a **150 Hz spectral centroid** (a bass drone; real rainfall sits nearer
5 kHz). The default `medium` covers both, so one engine renders the whole run; the
routing matters when you trade down. Special cells take `--sa3-sfx-model` /
`--sa3-sfx-backend` (default `medium` / `mlx`) independently of `--sa3-model`, so
`--sa3-model small-music --sa3-backend torch` renders the grid on the small checkpoint
and still sends the field recordings to medium — two engine groups, loaded in turn.
Pointing the SFX model at a music checkpoint is refused up front rather than rendering
plausible nonsense. Special-cell prompts also get the guide's SFX slate —
`TrackType: SFX`, subject first — instead of the music tags that were telling the
encoder to make ambient music.

**Every render is checked before it enters the library.** Generative engines fail
bluntly and at random — silent, clipped, a dead constant buffer — so each result goes
straight through `bnb.qc` while the model is still loaded, and a failure is re-rendered
on the spot with a fresh seed (`--max-retry`, default 3; retrying the *same* seed would
mostly reproduce the same broken audio). A track that never passes is left unrendered
rather than shipped broken, and the run exits non-zero so a pipeline notices. The
render block records what actually happened:

```json
"seed": 41654,
"qc": {"attempts": 1, "seed": 41654, "verdict": "ok", "warnings": [], "metrics": {...}}
```

`render.seed` matters because it can differ from the spec's: a retried render is no
longer reproducible from `spec.seed`, so provenance records the seed that made the file.

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
