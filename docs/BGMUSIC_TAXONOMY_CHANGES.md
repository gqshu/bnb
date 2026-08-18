# Background Music: Taxonomy & Prompt Revision — Implementation Guide

**Audience:** Claude Code (implementing against the existing background-generation codebase).
**Source:** test feedback + design review. Four changes, ordered by dependency.
**Non-negotiable invariant:** credit assignment for the entrainment signal must be
preserved. `band_guard` stays on in every path. Changes below make the *product audio*
richer without letting background amplitude-modulation land on the active entrainment
band.

---

## Context Claude Code needs before editing

The generation stack has two representations of the same space that are **not currently
reconciled**, and this task partially reconciles them:

1. **The taxonomy grid** — `{substrate × style × T(empo) × E(nergy) × S(pectral centroid)
   × M(ode) × X(texture density) × N(ature bed)}`, used for batch library generation.
2. **The runtime prompt builder** — `build_stable_audio_prompt()` with `BASE_STYLE`,
   `TECH_SPECS`, `POS_MAP`, `NEG_BASE`, `NEG_MAP`.

Both feed **Stable Audio 3** (primary; Small-Music on Mac/MPS, Medium self-hosted on
Linux/NVIDIA) and optionally **ElevenLabs Eleven Music** (structured composition-plan
alternative).

Two product-specific mechanisms must be respected throughout:

- **`band_guard`** — negative-prompt / mixing rule that keeps background content from
  imposing amplitude modulation near the active entrainment band. **Never remove.**
- **`BG_POLICY`** — per-modality background rules. This task **splits it by path purpose**
  (probe vs product), see Change 3.

**Measurement distinction that drives this whole task** (do not conflate):
- *Per-Hz ASSR* (clean spectral line at the entrainment frequency) — a **probe** measurement,
  heavily degraded by rich background. Protected by the strict probe path.
- *Relax/focus metric* (broadband relative-alpha band-power ratio: alpha-below-baseline =
  focus, alpha-above-baseline = relax) — the **product/adaptation** reward, far more robust
  to a rich background. This is what lets the product audio get richer safely.

---

## Change 1 — Remove global constraints that forbid melody

**Problem:** melody is suppressed at three levels simultaneously, so every cell collapses
toward the same "warm dark soft drone" wash regardless of substrate. `BASE_STYLE` bakes
drone assumptions into *every* substrate; `NEG_BASE` globally bans melody; individual
`POS_MAP` fragments (`piano`, `harp`) self-suppress.

**Do:**

1. **`NEG_BASE`** — remove the global melody bans: delete `"melodic hook, catchy melody,
   sectional structure"` from the always-on negative floor. Keep the arousal-safety bans
   (`"distortion, harsh high frequencies, dissonance, loud, energetic, aggressive, jarring
   transitions, sudden dynamics, build-up, crescendo"`). Melody is not an arousal hazard;
   abrupt dynamics are. Only the latter belong in the global floor.

2. **`BASE_STYLE` → make it substrate-conditional.** Replace the single fixed string with a
   per-substrate scaffold. The drone assumptions must stop leaking onto melodic and
   percussive substrates:

   ```python
   SUBSTRATE_SCAFFOLD = {
     "drone":       "slow evolving drone, sustained pads, no discernible beat, "
                    "warm timbre, seamless texture",
     "melodic":     "sparse melodic phrases, slow gentle motifs, space between notes, "
                    "soft attack, unhurried",
     "field":       "natural field recording foreground, minimal synthesis, "
                    "organic and continuous",
     "perc_tail":   "sparse resonant strikes with long decay, wide space between events, "
                    "inharmonic tails",
     "noise":       "smooth broadband texture, gentle and continuous, no tonal center",
   }
   ```
   `TECH_SPECS` keeps `"no percussion, 44.1kHz stereo, high fidelity"` but **remove
   `"smooth and consistent throughout"`** — that phrase forbids progression (see Change 2).

3. **`POS_MAP`** — remove the self-suppressing clauses: `piano`'s `"no melody hook"` and
   `harp`'s implicit sparse-only framing should become mode-dependent (a felt-piano *sleep*
   bed still wants restraint; a *focus* piano bed does not). Move the restraint into the
   mode filter (Change 3), not the fragment.

**Acceptance:** rendering the same POS keyword across `substrate=drone` vs
`substrate=melodic` produces audibly different tracks; `audio_features.py` reports
`melodic=True` with `contour_movement_semitones > 1.0` for the melodic-substrate render.

---

## Change 2 — Add a Development / Arc knob (progression)

**Problem:** no axis describes temporal evolution. Users want tracks that "go somewhere"
(tones, slow melody, a sense of progression), but every axis describes steady-state
character and `TECH_SPECS` explicitly forbids evolution.

**Do:**

1. **Add `development` as a new axis** with values `{static, slow_swell, motif_evolving}`.
   It is orthogonal to everything else (you can have a static melodic bed or an evolving
   drone), so it is a real axis, not a fold-in.

2. **Prompt fragments per value:**
   ```python
   DEVELOPMENT_FRAGMENT = {
     "static":         "unchanging and consistent, no development",
     "slow_swell":     "very slowly swelling and receding over minutes, gradual",
     "motif_evolving": "a simple motif that slowly develops and returns, "
                       "gentle harmonic movement over time",
   }
   ```
   The instantiator appends the selected fragment to the positive prompt.

3. **Mode-gate it** (this is where Change 3 connects): `development` is **not free across
   all modes**. Progression aids focus and wind-down engagement but works *against*
   sleep onset (familiarity / low prediction-error aids sleep — a standing project
   finding). The mode filter (Change 3) restricts which `development` values each mode
   admits. Do not let "users want progression" globally overwrite the sleep rationale.

**Acceptance:** a `motif_evolving` render reports `development_arc.arc in
{"slow_swell","motif_evolving"}` and non-trivial `timbral_drift` in `audio_features.py`;
a `sleep`-mode request never emits anything but `static`.

---

## Change 3 — Promote the mode filter to a first-class object over the grid

**Problem:** the relax/sleep/focus distinction is currently *implicit* — it falls out of the
fixed `BASE_STYLE` suppressing melody and progression everywhere. The moment Changes 1–2
make the grid variable, that implicit filter disappears and every mode would inherit the
full richer space. The filter must be made **explicit** so widening the taxonomy doesn't
silently widen every mode.

**Design principle:** "purpose" is a property of the *session*, not of the *sound*. Encode
it as a **mask over taxonomy cells**, keeping the sound taxonomy orthogonal. Same pattern
as `BG_POLICY`, one level up.

**Do:**

1. **Add `MODE_FILTER`:**
   ```python
   MODE_FILTER = {
     "focus": {
        "allow_substrate":   {"melodic", "drone", "perc_tail"},
        "allow_development": {"slow_swell", "motif_evolving"},
        "melody":            "encouraged",
        "energy_ceiling":    "raised",
        "background_group":  "energizer",     # see Change 4
        "require_am_compatible": True,        # HARD GATE — see Change 4
     },
     "relax": {   # wind-down
        "allow_substrate":   {"drone", "melodic", "field", "perc_tail", "noise"},
        "allow_development": {"static", "slow_swell"},
        "melody":            "allowed_sparse",
        "energy_ceiling":    "low",
        "background_group":  "grid",
     },
     "sleep": {   # descent
        "allow_substrate":   {"drone", "field", "noise"},
        "allow_development": {"static"},
        "melody":            "suppressed",
        "energy_ceiling":    "lowest",
        "background_group":  "grid",
     },
   }
   ```

2. **The filter is the single place** that maps a session mode to (a) which taxonomy cells
   are eligible, (b) which `development` values are allowed, (c) how much melody/energy, and
   (d) **which background group** to draw from. When new axis values are added later, they
   are admitted per-mode *here, consciously* — never inherited silently.

3. **`build_stable_audio_prompt()` takes `mode`** and consults `MODE_FILTER` to:
   select the substrate scaffold (Change 1), pick the allowed `development` fragment
   (Change 2), and decide melody restraint. The old `target_band` argument stays for
   `band_guard`.

**Acceptance:** requesting `mode="sleep"` can never produce `melodic=True` or non-`static`
arc; requesting `mode="focus"` routes to the energizer group and refuses any bed failing
the AM-compatibility gate.

---

## Change 4 — "energizer" as a separate AM-compatible focus group (NOT a grid fold-in)

**Decision (confirmed):** the Brain.fm-competing focus beds are **not** folded into the
down-regulation grid. They live in a separate curated group.

**Refinement — file it correctly:** `energizer` is **not** a substrate value parallel to
`natural_sounds`. `natural_sounds` is a *kind of sound* (a nature-bed / substrate value);
`energizer` is a **purpose-defined, pre-vetted pack** whose defining property is a hard
technical gate: every member must be **AM-compatible**. Filing it as a substrate would
reintroduce the substrate/style confusion at a new level. File it as a **named background
pack** that the focus mode-filter points to, defined by:

```python
ENERGIZER_PACK = {
  "purpose": "up-regulation / focus; competes with Brain.fm AM product",
  "members": [ ... curated bed ids ... ],
  "membership_gate": "am_compatible == True",   # checkable, measured
  "character": "more vivid, more melodic, more enjoyable than the down-reg grid",
}
```

**Why the AM-compatibility gate is load-bearing:** focus mode competes with Brain.fm,
which uses **AM**. To hold the competitive frame ("personalized/measured AM vs. their
fixed AM") the focus carrier stays **AM** — we do *not* switch to binaural to get richness.
Therefore every energizer bed must sit **underneath an AM carrier without corrupting it**:

- The bed must not impose amplitude modulation on the **active entrainment band**
  (`band_guard` enforces this). It *may* be rich, wide-band, and melodic otherwise.
- Richness is safe here because the focus **adaptation reward is the relax/focus
  band-power ratio, not the per-Hz ASSR** — the ratio tolerates a rich bed; the ASSR
  would not. (The strict ASSR-protecting path is the *probe*, Change 3.5 below.)

**Membership is measured, not asserted.** A bed enters `ENERGIZER_PACK` only if
`audio_features.py` confirms it is AM-compatible: negligible `am_depth_by_band` in the
target entrainment band while carrying its musical content elsewhere. Wire this as a
gate in the pack-build step:
```
am_compatible := am_depth_by_band[active_band] < THRESHOLD
```
Set `THRESHOLD` empirically (see open item). Reject any candidate bed that fails.

### Change 3.5 — split `BG_POLICY` by path purpose (probe vs product)

This is the mechanism that lets product-AM be rich while keeping ASSR validation possible.

```python
BG_POLICY = {
  ("AM", "probe"):   {"bed": "strip_or_-15dB", "cophase": True,  "band_guard": "strict"},
  ("AM", "product"): {"bed": "rich_allowed",   "cophase": True,  "band_guard": "on"},
  ("binaural","product"): {"bed": "rich_allowed", "band_guard": "on"},
  ("monaural","product"): {"bed": "rich_allowed", "band_guard": "on"},
  ("isochronic","probe"): {"bed": "strip",        "band_guard": "strict"},
}
```
- **probe** paths protect per-Hz ASSR → sparse/stripped, co-phase managed. Not the main
  listening experience; run periodically for validation.
- **product** paths protect only the broadband relax/focus ratio → rich bed allowed,
  `band_guard` still on to keep the bed out of the active band.

**Acceptance:** an energizer bed rendered under an AM product carrier reports rich musical
content (`melodic=True`, healthy energy/centroid) **and** near-zero `am_depth_by_band` in
the active entrainment band. A probe render stays sparse.

---

## Implementation order (dependencies)

1. **Change 1** first (un-ban melody, substrate-conditional scaffold) — unblocks everything.
2. **Change 2** (development axis + fragments) — independent, but its mode-gating needs 3.
3. **Change 3** (`MODE_FILTER`) — depends on 1 & 2 existing to have something to filter.
4. **Change 3.5** (`BG_POLICY` split) — independent; can land with or before 4.
5. **Change 4** (`ENERGIZER_PACK` + AM-compatibility gate) — depends on 3 (filter points to
   it) and 3.5 (product-AM path) and on `audio_features.py` for the membership gate.

## Open item to resolve empirically (do not hardcode a guess)

Run the factorial: **{sparse bed vs. rich melodic bed} × Δ in the relax/focus metric**
(NOT ASSR) using `eeg_report.py`. This yields two numbers this task currently leaves as
parameters:
- how rich the **product-AM** bed can go before the relax/focus ratio degrades
  (`energy_ceiling` / richness cap for the product path);
- the `am_compatible` `THRESHOLD` for energizer membership.

Until measured, expose both as config constants with conservative defaults and a `# TODO:
set from factorial` marker — do not bury a guessed number in logic.

## Guardrails to preserve (do not regress)

- `band_guard` on in **every** path, always.
- Style axis: **one coherent tradition per track**; wordless by default (no rendered
  sacred text); this is unchanged by the above.
- The bandit consumes **measured** feature vectors (`audio_features.py` output), not just
  requested coordinates — the energizer AM-gate and richness cap are both measured.
- Sleep-descent stays static/familiar; progression is a focus/wind-down affordance only.
