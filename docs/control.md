# Session Control & Goal Design

*Build spec for the closed-loop control system and the user-facing goal model. Supersedes the implicit "single dominant frequency + two directions + duration" model. Scope: down-regulation only (Recharge, Wind Down, Sleep). Grounded in the eyes-closed Fp1 resting data showing bursty, intermittent frontal alpha over a strong 1/f background.*

---

## 1. Core principle: separate the *internal control variable* from the *user-facing goal*

Two different things have been conflated under "dominant frequency":

- **The internal control variable** — what the algorithm measures and servos on. Must be robust on bursty, 1/f-heavy, single-channel Fp1 EEG. This is a **state vector**, not a scalar.
- **The user-facing goal** — what the user sets, understands, and sees themselves reaching. Must be legible and motivating. This is a **simple, low-dimensional target** projected from the state vector.

The design keeps a legible goal at the surface while running the robust vector underneath. The scalar "frequency" was attractive only because it accidentally served *both* roles; it does neither well. We split them.

---

## 2. Internal control system

### 2.1 What the resting data established
- Frontal eyes-closed alpha is **bursty**: prominent peaks (prominence 0.7–0.9) appear intermittently, separated by long stretches where no peak clears threshold.
- **Band-ratio signals (relative alpha, θ/α) are defined every frame** and move coherently with the bursts (relA rises to 17–40% during peaks, collapses to 2–9% between; θ/α inversely).
- Instantaneous peak frequency jitters 5–14 Hz and is undefined much of the time → **unusable as a per-second servo target.**

Conclusion: the continuous, robust signal is the **band-power distribution**, not a scalar frequency.

### 2.2 The state vector (computed every ~1 s frame)
From the 1/f-whitened, band-restricted (~4–15 Hz) spectrum:

| Component | Definition | Role | Defined when |
|---|---|---|---|
| `relAlpha` | alpha power / total in-band power | **primary state axis** | every frame |
| `thetaAlpha` | θ power / α power | secondary state axis (drowsiness tips it up) | every frame |
| `burstFreq` | centroid of the alpha peak | personalization only (start Δ) | high-prominence frames only |
| `prominence` | peak height above aperiodic floor | confidence gate | every frame |
| `sigQuality` | POOR_SIGNAL / artifact flag | master gate | every frame |

### 2.3 Smoothing (critical — never act on instantaneous values)
- **`relAlpha`, `thetaAlpha`:** rolling mean/median over ~10–20 s. This is the trend the loop reads.
- **`burstFreq`:** rolling median over high-prominence frames only (prom > 0.65), ~10–20 s window. Ignore all flat frames. Used once at session start to personalize Δ; optionally re-checked, not servoed.
- **All updates suppressed** when `sigQuality` poor.

### 2.4 What drives the loop
- **Open-loop drive:** the binaural beat Δ steps downward toward the mode target (the patent's stepwise descent). This is the *actuator*.
- **Closed-loop confirmation:** read the **smoothed state trend**, not a scalar frequency match. "Beat is at 7 Hz — has the brain followed?" becomes **"is `relAlpha` rising / `thetaAlpha` trending up and holding in the expected range?"**
- **Advance rule:** step Δ down when the smoothed state has moved in the target direction and *stabilized* for a sustained window (e.g. ≥30–60 s), not when a single frame's peak equals Δ.
- **Hold/back-off rule:** if the state reverses or signal quality drops, hold (or back off one step) rather than pushing.

This is a deliberate, documented divergence from the literal patent text: the patent's *narrative* is "predominant frequency follows the beat"; the *implementable, hardware-robust* version is "band-power state shifts toward target as the beat steps down." Same intent, built on a signal the device can actually measure continuously.

### 2.5 Honest limitation (carry forward)
Everything here assumes the beats move these signals *at all* on this hardware — untested until a beat-on vs. sham comparison on the **smoothed band-ratio trajectory** is run. That is the single most valuable next recording.

---

## 3. User-facing goal model

### 3.1 The requirement
Keep the motivational virtue of a single legible target ("set a goal, watch yourself reach it") **without** exposing a scientifically ill-posed scalar frequency, and **without** collapsing the product to "two directions + a duration."

### 3.2 The solution: a **"Calm Score" / journey**, projected from the state vector
Project the multi-dimensional state onto **one legible progress axis** the user can set a target on and watch climb — but define that axis from the robust band-ratio state, not from a raw Hz number.

- Define a **`relaxationDepth`** (0–100), a monotonic mapping from the smoothed state (primarily `relAlpha` rising and `thetaAlpha` shifting) onto a single "how deep are you" number.
- The user sets a **target depth** (or picks a named state — see below), and the session shows a **live journey** from their starting depth to the target: a rising line, a filling ring, a "descending into calm" metaphor.
- This preserves the exact psychological hook of the single frequency (a number to reach, visible progress, a completion moment) while the underlying quantity is the robust, continuously-defined state — not the jittery undefined scalar.

Why this beats exposing frequency: a user watching "your frequency: 7.8 Hz" is watching a number that's undefined 60% of the time and that jerks between 5 and 14 Hz on this hardware — *demotivating and confusing*. A user watching "Calm depth: 62 → 80" is watching a smooth, always-defined, honestly-derived progress signal. It's *more* motivating precisely because it's smooth.

### 3.3 Goal granularity: named states, not raw parameters
Give the user richer goals than direction+duration, but express them as **named target states** mapped to internal (depth target + descent shape + soundscape bias), not as knobs. Examples:

| User picks | Internal target | Descent shape |
|---|---|---|
| "Reset / Recharge" | moderate depth, hold | efficient down then plateau |
| "Wind Down" | high depth (~alpha-theta border) | slow, gentle |
| "Drift to Sleep" | deepest, then fade | slowest, mirror sleep onset, fade audio |
| "Quick Calm" (anxious) | moderate depth, fast | faster descent for quick relief |

Each named state is a *preset* over the real control parameters (target depth, descent rate, soundscape). The user gets meaningful choice; the complexity stays hidden.

### 3.4 Optional second axis for users who want more control
For users who want to define a richer goal without exposing raw EEG, offer **at most two intuitive sliders**, each mapped to internal parameters:

- **"How deep"** (light calm ↔ deep drift) → sets target `relaxationDepth`.
- **"How fast"** (gentle ↔ efficient) → sets descent rate.

That's it. Two semantic sliders + a named state give combinatorially richer goals than "two directions + duration" while remaining a phone-tappable UX. Everything else (carrier, Δ stepping, band-ratio gating, soundscape) is derived, never surfaced.

### 3.5 What the user sees during a session
- A single **progress visual** (ring/line) climbing from start depth to target — the legible "reaching the goal" moment.
- **Not** raw frequency, not band ratios, not the state vector. Those are internal.
- A post-session summary in plain language ("You reached deep calm in 8 minutes and stayed there for 12") — the History/Session-Results surface already in the UX flow.

---

## 4. Mapping summary (internal ↔ surface)

| Layer | Quantity | Exposed to user? |
|---|---|---|
| Sensing | 1/f-whitened band-restricted spectrum | No |
| Control state | `relAlpha`, `thetaAlpha` (smoothed), `burstFreq`, quality | No |
| Actuator | binaural Δ stepwise descent | No (they just hear it) |
| Projection | `relaxationDepth` 0–100 | **Yes — as Calm depth / journey** |
| Goal | named state + optional 2 sliders (how deep / how fast) | **Yes** |
| Feedback | rising progress visual + plain-language summary | **Yes** |

---

## 6. Formulas: relaxationDepth and beat parameters

*Implementation-ready spec. Three separable functions plus a baseline-gate routine. **Design rule: `relaxationDepth` (display/progress) and the beat actuator are separate jobs. Depth may *gate* stepping but must never be a controller output that Δ chases continuously — that would servo the actuator on a score the actuator itself influences.** The initial beat comes from `burstFreq`, not from depth.*

*All numeric constants below are **starting guesses to calibrate on real sessions, not derived values.** They are collected in the config block (§6.5) and must be treated as tunable, not magic numbers.*

### 6.0 Per-user normalization (prerequisite — nothing works without it)
Absolute band power is meaningless across people and TGAT units, so depth must be **baseline-relative**. During the 60 s eyes-closed baseline gate, capture the distribution of the two smoothed inputs, per-user per-session. Use **MAD (median absolute deviation)**, not SD — the signal is burst-driven with outliers.

```
alpha_base_med = median(relAlpha_s)   over baseline window
alpha_base_mad = max(MAD(relAlpha_s),   eps_mad)
ta_base_med    = median(thetaAlpha_s) over baseline window
ta_base_mad    = max(MAD(thetaAlpha_s), eps_mad)
```
`eps_mad` floors the denominator so a very stable baseline can't cause divide-by-~0.

### 6.1 `computeDepth(...) → 0..100` (the display score)
Operates on **smoothed** inputs (10–20 s rolling median), only when `sigQuality` is good. Two robust inputs: relative alpha rising, θ/α tipping up.

```
z_alpha = (relAlpha_s   - alpha_base_med) / alpha_base_mad
z_ta    = (thetaAlpha_s - ta_base_med)    / ta_base_mad

s     = w_a * z_alpha + w_t * z_ta            # w_a=0.7, w_t=0.3
depth = 100 / (1 + exp(-k * (s - s0)))        # k=0.9, s0=0
```
- At baseline `s≈0 → depth≈50`. As alpha rises, `s>0`, depth climbs toward 100 — the "watch yourself descend" journey. Shift `s0` positive (e.g. 1.0) if you want baseline to read lower (~20) so real relaxation is needed to reach 50+. This is cosmetic display-tuning, not physiology.
- `k` = needle sensitivity; `s0` = where "50" sits. Tune so a genuinely relaxed state reads ~75–85. **These affect display only — never control logic.**
- **Anti-gaming:** update only when `sigQuality` good AND the alpha is real (`prominence` present), not an artifact. A jaw clench spikes power broadband; require the *coherent* pattern (relAlpha up **and** θ/α not simultaneously crashing from an EMG high-freq leak). Reject bad frames and **hold last good depth** — never let depth jump on a bad frame.
- **UI rate-limit:** cap displayed change (e.g. ±2 points/s) so the ring glides.

### 6.2 `computeInitialBeat(...) → Δ_0` (does NOT use depth)
The starting beat comes from the user's own alpha, measured during the baseline gate — the personalization the adaptive premise rests on.

```
f_start = rolling_median(burstFreq where prominence > 0.65)  over baseline window

# fallbacks — frontal alpha may not burst during a short baseline:
if too_few_high_prom_frames:
    f_start = clamp(peak_of_whitened_spectrum(4..15Hz), 8, 12)  # default into alpha
if still_undefined:
    f_start = 10   # population alpha prior

Δ_0      = f_start          # start the beat at their measured alpha
Δ_target = mode_target      # 7.8 (Wind Down) / ~5–6 / 4 (Sleep)
```
Starting Δ at *measured* alpha (not a canned number) is the "start where the brain is" principle — why `f_start` comes from `burstFreq`, not depth.

### 6.3 `updateBeat(...) → Δ_next` (depth gates the step; it is not the value)
Advance only when the user has followed the current step. Depth's **trend/stability** (the *change* since the current step began), not an absolute depth number, is the gate.

```
# while Δ > Δ_target:  hold current Δ, beat playing, track depth over dwell window

advanced = (Δdepth_since_step_start > step_threshold
            AND depth stable for ≥ dwell_seconds
            AND sigQuality good throughout)

if advanced:
    Δ = max(Δ - step_size, Δ_target)   # step down (step_size = 1.0 or 0.5 Hz)
    reset dwell timer
elif depth_dropped > back_off_margin:
    Δ = min(Δ + step_size, Δ_0)        # back off one step
    reset dwell timer
else:
    hold
```
- Gate reads the **change in smoothed depth since the step began** + a stability check — not "did they hit 80" (depth is baseline-relative).
- **Δ never chases depth continuously.** Depth = gate (step / hold / back-off), never a linear controller output.
- Δ is **bounded** `[Δ_target, Δ_0]`.
- **Fixed-schedule fallback (ship this until validated):** if depth is too noisy to gate reliably on bursty Fp1, step on a slow fixed cadence (−1 Hz every 60–90 s) and use depth only as a **monitor/abort** (if depth collapses, hold). The beats-on-vs-sham pilot tells you which regime you're in. **Until that pilot confirms the beat moves depth, ship fixed-schedule stepping with depth-as-monitor, not depth-as-gate.**

### 6.4 Function/routine map for Claude Code
- `baselineGate()` → populates `baselineStats` (§6.0) and `f_start` (§6.2)
- `computeDepth(relAlpha_s, thetaAlpha_s, baselineStats, sigQuality, prominence) → 0..100`
- `computeInitialBeat(burstFreqSeries, prominenceSeries, whitenedSpectrum) → Δ_0`
- `updateBeat(Δ_current, Δ_target, depthWindow, sigQuality, dwellTimer) → Δ_next`

### 6.5 Config block (all tunable — calibrate on real sessions)
```
eps_mad        = <small floor>     # normalization denominator floor
w_a            = 0.7               # alpha weight in depth
w_t            = 0.3               # θ/α weight in depth
k              = 0.9               # depth logistic slope (display only)
s0             = 0                 # depth logistic center (display only)
smooth_window  = 10–20 s           # rolling median for relAlpha/thetaAlpha
prom_gate      = 0.65              # min prominence to trust burstFreq
depth_ratelim  = 2 pts/s           # UI glide cap
step_size      = 1.0 Hz            # (or 0.5)
dwell_seconds  = 45–60 s           # hold before allowing a step
step_threshold = <Δdepth to advance>
back_off_margin= <Δdepth to back off>
fixed_cadence  = 60–90 s/step      # fallback stepping when depth too noisy
```

---

## 7. Open items
- **Calibrate the `relaxationDepth` mapping** from real sessions: which combination of smoothed `relAlpha`/`thetaAlpha` (and their personal baselines) maps to a subjectively honest 0–100. Must be normalized per-user (baseline-relative), since absolute band power varies across people and TGAT units.
- **Anti-gaming / honesty:** ensure the depth score can't be trivially inflated by artifacts (jaw clench, movement) — tie it to signal quality and to the coherent relA↑/θα-pattern, not to raw power.
- **Beat-on vs. sham validation** of whether the smoothed state actually responds to stepping — prerequisite to trusting the closed loop and therefore the depth journey.
- **7.83 Hz framing:** where a target near 7.8 Hz is used (Wind Down), justify it as a low-alpha/high-theta relaxation target, *not* as the Schumann resonance. The Schumann provenance is branding trivia, not mechanism, and tying claims to it undercuts the product's scientific credibility.