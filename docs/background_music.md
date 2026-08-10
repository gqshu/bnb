# Background Track Generator — Implementation Plan

*Standalone build spec for the personalized background-soundscape library that feeds the per-user contextual bandit. Scope: down-regulation only (Recharge, Wind Down, Sleep). This doc fixes the API provider decision and details the music-category (substrate × style) development plan.*

---

## 1. Provider decision (fixed)

| Role | Provider | Why |
|---|---|---|
| **Primary library engine** | **Stable Audio 3.0 Open (self-hosted, Medium weights)** | You own outputs under the Community License (Enterprise only if/when >$1M revenue); fixed infra cost instead of per-track fees at library scale; instrumental/ambient-native (no unwanted vocals); per-second duration + seed control for reproducible grid sweeps. |
| **Hero renders / hosted fallback** | **Stable Audio 2.5 / 3.0 API (hosted)** | Same license family and prompt surface; use for highest-quality "hero" tracks and for spillover when self-host capacity is saturated. ~$0.20/generation. |
| **Structured-control alternative** | **ElevenLabs Eleven Music** | Composition-plan JSON gives the richest parametric surface (per-section BPM/key/styles, `force_instrumental`, seed); use where its structured control beats free-form prompting. Verify current per-minute rate before budgeting (published $0.15/min, possible change to $0.30/min in flight). |

**Rationale recap:** all three clear the three hard constraints (real API + parametric control, pre-gen library fit, royalty-free commercial embedding). Stable Audio uniquely lets us sidestep per-track licensing by owning the compute, which is decisive for a large, periodically-refreshed candidate library. Suno/Udio/BandLab remain disqualified (no official API; litigation shadow). MusicGen is rejected as primary because its pretrained weights are CC-BY-NC (non-commercial).

**Standing decisions to confirm before Stage 2 scale-up:**
- Self-host GPU footprint sized for batch generation (target throughput in §5).
- Legal sign-off on Community License terms for our revenue tier.
- Post-generation objective-feature extraction pipeline (librosa/Essentia) is part of the generator, not an afterthought — the bandit needs *measured* features, not just requested ones.

---

## 2. The taxonomy: substrate × style (two orthogonal axes)

The core modeling decision: **what a sound physically is (substrate)** and **what tradition it evokes (style)** are separate axes. A track is a *pair*. This keeps the feature space orthogonal, lets the bandit learn substrate-preference and style-preference independently, and preserves generalization (a user who rewards Tibetan-bowl tracks can have that partially transferred to a Western bell drone via shared substrate features).

### Axis A — Substrate (what it physically is)
Defined by its MER feature footprint (spectral centroid, onset density, harmonic content, register).

| Substrate | Physical description | Typical MER footprint |
|---|---|---|
| `drone` | sustained tone/pad, no rhythm | arrhythmic, low onset density, steady loudness |
| `melodic_instrument` | piano, guitar, harp, flute lines | sparse onsets, pitched, gentle dynamics |
| `field_recording` | rain, stream, ocean, wind | broadband, no pitch center, no rhythm |
| `noise_texture` | pink/brown-noise-like tonal wash | flat spectrum, no melody, steady |
| `percussive_with_tail` | bowls, bells, gongs, chimes | transient onset + long inharmonic decay |

### Axis B — Cultural style (what tradition it evokes)
Realized *through* a substrate. A conditioning layer on top of the MER space, not a replacement for it.

| Style | Realized via | Signature markers |
|---|---|---|
| `neutral` | any substrate | no strong cultural cue |
| `buddhist_meditative` | bowls, bells, shakuhachi, mantra-hum, drone | inharmonic metallic partials, just-intonation, low register, breathy flute |
| `neoclassical` | felt piano, strings | tonal, sparse, warm close-mic |
| `lofi` | soft keys, tape texture | filtered highs, gentle noise floor |
| `nature_ambient` | field recording + pad | water/wind beds, wide stereo |
| *(extensible)* | | add styles without touching substrate axis |

**Key property:** any (substrate, style) pair is valid and renderable. Buddhist × drone = bowl/mantra drone; Buddhist × melodic_instrument = shakuhachi; Buddhist × percussive_with_tail = singing bowls; Buddhist × field-recording-hybrid = temple bells over a stream. The style axis mostly changes **instrumentation, scale/mode, and specific timbral signatures**; the substrate and the arousal/valence MER features still describe the physical sound.

### The remaining (continuous) MER axes — unchanged, apply to every pair
Tempo/onset-density (T), energy/loudness (E), spectral centroid/brightness (S), mode/harmonic complexity (M), register (R), texture density (X), nature/noise-bed level (N). All biased toward low arousal for down-regulation, but explored within.

---

## 3. Metadata schema (per track)

Every render carries: requested substrate, requested style, requested MER coordinates, **and measured MER features** (extracted post-generation, because models don't reliably hit requested BPM/brightness). The measured vector is what the bandit uses as continuous context; substrate and style are categorical context tags.

```json
{
  "track_id": "buddhist_bowls_T00_Elow_Sdark_seed7",
  "provider": "stable-audio-3.0-medium-selfhost",
  "substrate": "percussive_with_tail",
  "style": "buddhist_meditative",
  "requested_features": {
    "tempo_bpm": null, "energy": "very_low", "spectral_centroid": "warm_metallic",
    "mode": "just_intonation_drone", "register": "low_mid",
    "texture_density": "very_sparse", "nature_bed": "none"
  },
  "measured_features": {
    "tempo_bpm": 0, "rms_lufs": -28.4, "spectral_centroid_hz": 780,
    "onset_density_per_s": 0.08, "dynamic_range_db": 6.2
  },
  "instrumentation": ["tibetan_singing_bowls", "distant_temple_bell"],
  "prompt": "...", "negative_prompt": "...", "seed": 7,
  "duration_s": 150, "loopable": true, "license": "community-owned",
  "provenance": {"generated_at": "...", "model_version": "...", "watermark": null}
}
```

---

## 4. Prompt strategy (substrate template + style modifier)

Style slots in as an instrumentation/tradition modifier layered onto the substrate template. Front-load hard constraints (instrumental, tempo, timbre) because prompt adherence is strongest for leading tokens.

**Base template (Stable Audio / self-host):**
> `Instrumental [style-descriptor] soundscape for deep relaxation. [substrate + instrumentation]. Tempo [T], [E] energy, very soft dynamics. [S] timbre, warm and dark, low spectral brightness. [M] harmony, [X] texture, [R] register. [N] blended softly underneath. No vocals, no percussion hits, no sudden transitions. Seamless, calm, continuous.`

**Negative prompt (all down-regulation tracks):**
> `bright, harsh, energetic, fast, distorted, drums, buildup, EDM, sudden transitions`

### Worked style example: `buddhist_meditative` swept across substrates

The point of the orthogonal taxonomy: hold style fixed, sweep substrate, to give the bandit clean within-style gradients.

1. **Buddhist × percussive_with_tail (singing bowls):**
   > `Instrumental meditative soundscape for deep relaxation. Tibetan singing bowls and a distant temple bell, long resonant decays, inharmonic metallic partials. Arrhythmic, very low energy, warm-dark timbre, sparse, low-mid register. Soft continuous drone underneath. No vocals, no percussion hits, no sudden transitions. Seamless and calm.`

2. **Buddhist × drone (bowl/mantra drone):**
   > `Instrumental meditative drone for deep relaxation. Sustained singing-bowl resonance and a low wordless vocal hum, just-intonation tuning. Arrhythmic, very low energy, warm-dark timbre, very sparse, low register. No lyrics, no percussion, no sudden transitions. Seamless and continuous.`

3. **Buddhist × melodic_instrument (shakuhachi):**
   > `Instrumental meditative soundscape. Solo shakuhachi bamboo flute, breathy long tones, minor-pentatonic, ~50 bpm feel, very soft. Warm-dark timbre, sparse, mid register. Faint singing-bowl drone underneath. No percussion, no sudden transitions. Seamless and calm.`

4. **Buddhist × field-recording hybrid (temple + water):**
   > `Instrumental meditative soundscape. Distant temple bells with long tails over a soft mountain stream, occasional wind chime. Arrhythmic, very low energy, warm-dark timbre, sparse, wide gentle stereo. No vocals, no rhythm, no sudden transitions. Continuous and calming.`

### ElevenLabs rendering of the same
`positive_global_styles`: `["meditative","tibetan singing bowls","just intonation","instrumental","very sparse","low register","ambient drone"]`; `negative_global_styles`: `["bright","aggressive","drums","vocals","fast","buildup"]`; `force_instrumental=true`; one long section = loop length; fixed seed per signature.

---

## 5. Development plan (staged)

### Stage 0 — Taxonomy freeze & tooling (week 1)
- Lock Axis A (5 substrates) and Axis B initial styles (`neutral`, `buddhist_meditative`, `neoclassical`, `lofi`, `nature_ambient`).
- Stand up the objective-feature extraction pipeline (librosa/Essentia → measured MER vector) and the metadata store schema (§3).
- Define loop length (target 120–180 s, seamless-loopable) and export format (44.1 kHz WAV master → app-side compressed delivery).

### Stage 1 — Prototype library & provider bake-off (weeks 2–4)
- Generate the 5 archetype seeds × 5 substrates × the `buddhist_meditative` and `neutral` styles on **hosted Stable Audio** and **ElevenLabs** in parallel.
- Blind internal A/B on (a) ambient quality for sleep/wind-down, (b) prompt adherence (requested vs. measured MER within tolerance), (c) loop seamlessness.
- **Advance gate:** >80% of rendered cells subjectively acceptable as down-regulation beds AND measured features within tolerance of requested coordinates.

### Stage 2 — Scale the library on self-host (weeks 5–10)
- Deploy **self-hosted Stable Audio 3.0 Open (Medium)** on GPUs; batch-generate the full grid.
- **Grid design:** don't enumerate the full Cartesian product (it explodes). Use a **fractional-factorial / Latin-hypercube sample** across {substrate × style × T × E × S × M × X × N}, then render **3–5 seeds per cell** so cell-effect isn't confounded with an unlucky render.
- **Clean gradients for the bandit:** also generate targeted mini-sweeps that hold everything fixed and step one axis (e.g. Buddhist × bowls, step reverb/texture density; or fix substrate, step tempo 45→50→55→60) so reward differences are attributable to a single feature.
- Run every render through objective-feature extraction; store measured vectors.
- Keep hosted Stable Audio for hero renders and overflow.

### Stage 3 — Bandit integration & refresh loop (weeks 9–14, overlapping)
- Expose the tagged library to the contextual bandit. Continuous covariates = measured MER features; categorical context = substrate + style.
- **Per-user sound signature:** anchor each user to a home region (substrate + style + S/M cluster their EEG+ratings favor); let the bandit explore the softer axes (T, E, X, N) locally within that region. Reuse a fixed per-user seed family so signature re-renders stay timbrally consistent.
- **Refresh:** use bandit reward to find high-performing regions, then densify the grid there (new seeds + neighboring cells) on the next self-host batch. Maintain an explicit exploration budget so novel cells keep getting tried (counters habituation).
- Re-benchmark quarterly as models update.

---

## 6. Cultural-style guardrails (specific to Axis B)

Cultural/religious styles carry appropriateness and authenticity considerations that pure acoustic substrates don't. Bake these into the generator, not into review-after-the-fact:

1. **One coherent tradition per track.** AI generation can blend incompatible traditions (Tibetan bowls + Zen shakuhachi + Hindu mantra are *different* lineages). Constrain each `buddhist_meditative` prompt to a single coherent sub-tradition; don't let the model mix them in one render.
2. **Wordless by default.** Mantra chanting often contains real religious text, and generative models can mis-render sacred words. Prompt for **wordless vocal hums / bowl tones**, not mantras, unless you have a specific vetted reason and source.
3. **Authenticity review if it's part of positioning.** If "Buddhist/meditative" is a marketed category (not just an internal tag), have tracks spot-reviewed by someone familiar with the tradition before they enter the user-facing library.
4. **Keep style tags neutral-descriptive internally** (`buddhist_meditative`) but consider softer user-facing labels (e.g. "Temple / Singing Bowls") to avoid implying religious endorsement.

---

## 7. The goal axis: relax vs. focus (up-regulation)

Everything above shipped down-regulation-only on purpose (the doc's original scope note).
This section adds a second
goal, **focus**, as a **third axis, orthogonal to substrate and style** — not a parallel
taxonomy. A track is now a `(substrate, style, goal)` triple; `goal` owns every phrase in
the base template (§4) that used to be hardcoded relax language (the opening intent, the
motion clause, dynamics/brightness descriptors, the closing sentence, the negative prompt).

### 7.1 Not every cell supports both goals

Substrate and style each carry a `goals` allow-list (default: both). Two cells are
relax-only:

- **`percussive_with_tail`** (singing bowls, bells) — long resonant decays between strikes
  are a meditative gesture, not something you want decaying across a task-focused attention
  span.
- **`buddhist_meditative`** — both the sound and the branding read as meditation, not
  productivity; the same guardrail concern §6 raises about mixing traditions applies to
  stretching this style's branding toward "focus."

Every other substrate/style supports both goals; the difference is the MER coordinates
(§7.2), not the identity of the sound.

### 7.2 What actually differs for focus (corrected from the naive assumption)

The intuitive guess — "focus music is higher energy, more power, more melody" — is largely
backwards. High arousal hurts sustained attention (Yerkes-Dodson), and generic "energetic"
music reintroduces exactly the salient, attention-grabbing events a focus bed should avoid.
The evidence-backed shape, and what this library's `focus` goal template encodes:

- **Steady, predictable pulse** instead of arrhythmic — a real tempo the listener doesn't
  have to work to parse, not "no pulse" and not "fast."
- **Lower melodic *complexity/surprise*, not more melody** — no hooks, no key changes, no
  vocals; repetition, not development. The focus negative prompt explicitly bans lyrics,
  key changes, chord progressions and dramatic climaxes that the relax negative prompt
  never needed to mention.
- **Moderate-low energy**, not high — enough to not be sleep-inducing, not enough to be
  arousing.
- **Predictability over stillness** — the relax `MOTION` clause ("never builds, resolves,
  or arrives anywhere") becomes a focus-specific clause: "repeats in a steady, unsurprising
  loop... but it never swells, builds, or arrives anywhere new." Focus tolerates — even
  wants — a beat; it just can't have drama.

### 7.3 The Brain.fm-competing mechanism sits outside the grid

The differentiator that actually competes with Brain.fm isn't a prompt change at all: it's
a **live amplitude-modulation (AM) pulse** applied to a `goal=focus` background track at
serve time, using `tone.render_am_music`'s envelope math run incrementally by the stream
engine (`stream.py`'s `am_music` beat mode). This is mechanistically closer to what Brain.fm
actually does — rhythmic modulation of instrumental music — than binaural beats are; Brain.fm
explicitly positions against binaural beats (see the internal research assessment). The
pulse rate/depth are never surfaced to the user as Hz — only named presets are (`docs/
control.md` §3.3), consistent with the existing relax UX philosophy of hiding raw parameters
behind legible named states.

## 8. Open items / decisions needed
- Confirm loop length and whether the app crossfades or hard-loops (affects render tails, esp. for `percussive_with_tail` bowls with long decays).
- Decide watermark stance: self-hosted Stable Audio outputs are unmarked (Lyria would force SynthID); confirm no provenance-marking requirement for the neurotech product / EU AI Act Article 50 disclosure.
- Confirm objective-feature tolerance thresholds (how far measured MER may drift from requested before a render is rejected).
- Size self-host GPU footprint against target library size and refresh cadence.