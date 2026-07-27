
All projects
Binaural Beats Product



How can I help you today?


Recents
Binaural beats EEG measurement research
12 hours ago
EEG channel difference analysis with baseline and music
3 days ago
Binaural beats EEG measurement research
4 days ago
Neurofeedback binaural beats product research plan
5 days ago
Memory
Only you
Purpose & context George is building a consumer neurotechnology product: a sealed over-ear headset with a single Fp1 EEG electrode (NeuroSky TGAT/ThinkGear ASIC chip) paired with a mobile app, delivering adaptive binaural beats for relaxation and sleep. The product is strictly down-regulation only (Recharge, Wind Down, Sleep Preparation modes). Deep Focus/gamma mode was explicitly scoped out as scientifically indefensible on this hardware. The core value proposition is adaptive (not static) binaural beats guided by live EEG feedback. Key scientific grounding George has established: Binaural beat efficacy is real but modest (pooled effect ~medium, Ingendoh et al. 2023 systematic review); down-regulation is far better supported than up-regulation Adaptive vs. static beats: promising feasibility evidence but no definitive head-to-head trial yet Background/soundscape music can interfere with entrainment; pure-sine binaural beats chosen as default for maskability and pleasantness Schumann resonance (7.83 Hz) is a real geophysical phenomenon but "brain-planet harmony" claims are not established; 7.8 Hz is used as a sensible alpha-theta target on scientific, not mystical, grounds --- Current state George is in active technical development, having completed the foundational scientific validation and core signal processing architecture. Key decisions locked in: Tone-pair formula: Convention B symmetric (left = x − Δ/2, right = x + Δ/2), interaural loudness matched to within ~2 dB Dominant frequency estimation: Center of Gravity (CoG) over a 4–15 Hz whitened, band-restricted spectrum (not argmax); FOOOF/specparam or PSD×f whitening to remove 1/f aperiodic slope; two-tier design — CoG for stepping control, band-ratio for state gating Eyes-closed requirement: Baseline must be captured after eye closure to avoid misattributing the Berger drop to the beats; artifact reduction at Fp1 is decisive Control system: Full build spec written (ControlAndGoalDesign.md), covering state vector (relAlpha, thetaAlpha, burstFreq, prominence, sigQuality), per-user MAD-based normalization, logistic relaxationDepth formula, computeInitialBeat from high-prominence burst frequency, and updateBeat gate (depth trend + stability, with fixed-schedule fallback pending beats-on-vs-sham validation) UX design tension resolved: Single scalar frequency is motivating/legible but scientifically ill-posed → projects internal band-ratio state vector onto a 0–100 "relaxationDepth" score (UX-facing), with named target states (Reset, Wind Down, Drift to Sleep, Quick Calm) and two semantic sliders (how deep / how fast) Soundscape personalization: Contextual multi-armed bandit over a MER-defined feature space with blended EEG + subjective reward Live EEG data has been analyzed across two recordings. Key finding: per-second peak frequency is too noisy for closed-loop stepping; bursty intermittent frontal alpha confirmed; 1/f problem resolved via whitening + band restriction. Analysis of a one-minute 10 Hz beat stepping-down log was in progress at the end of the last conversation. --- Key learnings & principles Down-regulation only is the right product scope: Gamma/up-regulation is the most scientifically vulnerable claim on this hardware; eliminating it strengthens credibility Adaptive strategy trust: The adaptive approach is validated from the patent and feasibility literature; the absence of a definitive head-to-head trial is a known gap, not a blocker Signal quality over frequency precision: Per-second peak frequency is too noisy; CoG + band-ratio gating is more robust than argmax for closed-loop control Eyes-closed baseline is non-negotiable: Failing to take baseline post-eye-closure would systematically misattribute the Berger alpha increase to the intervention 1/f slope must be removed before spectral peak estimation: Raw PSD peak-picking on EEG is unreliable; whitening is a prerequisite step UX legibility vs. scientific precision: The tension between a single motivating metric and a multi-dimensional EEG state is real — resolved by keeping the science internal and surfacing a projected scalar to users --- Approach & patterns George conducts deep, first-principles technical reviews before committing to product decisions — scientific literature, hardware constraints, and UX considerations are all evaluated together Decisions are documented as build specs (e.g., ControlAndGoalDesign.md), suggesting a structured engineering workflow Iterates between theory and live data: validates design assumptions against real EEG recordings before finalizing algorithms Prefers to resolve design tensions explicitly (e.g., single scalar vs. multi-dimensional state) rather than leaving ambiguity in the architecture --- Tools & resources Hardware: NeuroSky TGAT (ThinkGear ASIC) chip, single Fp1 electrode, sealed over-ear headset form factor Signal processing: FOOOF/specparam for aperiodic slope removal; CoG-based dominant frequency estimation; MAD-based per-user normalization Key references: Ingendoh et al. 2023 (systematic review), Kahathuduwa/Dhanasekara 2025 (adaptive feasibility, n=25) ML/personalization: Contextual multi-armed bandit for soundscape selection

Last updated 5 days ago

Instructions
Add instructions to tailor Claude’s responses

Files
4% of project capacity used
Search mode

Monaural_and_Isochronic_Beats_Implementation.md
274 lines

md



Binaural Beats and EEG Entrainment: Evidence Synthesis for a Frontal-Pole Adaptive Headset.md
90 lines

md



Control_And_Goal_Design.md
213 lines

md



Neurofeedback-Adaptive Binaural Beat Headset: Scientific and Technical Feasibility Assessment.md
156 lines

md



Addendum_Product_Design_Decisions.md
94 lines

md


binauralbeats_app_flow.jpg

202403202 PCT Filed Drawings.pdf
pdf


202403202 PCT Filed Patent.pdf
pdf



Monaural_and_Isochronic_Beats_Implementation.md


# Implementing Monaural and Isochronic Beats
 
*Design and implementation reference for adding monaural-beat and isochronic-tone
generation alongside the existing binaural-beat + background mixing pipeline.*
 
---
 
## 1. Purpose and scope
 
The existing generator produces **binaural beats**: two pure tones, one per ear,
whose frequency difference is perceived as a beat that is constructed centrally in
the brain. This document specifies two additional stimulus types:
 
- **Monaural beats** — the beat is physically present in the air (the two tones are
  summed before playback), delivered identically to both ears.
- **Isochronic tones** — a single carrier whose amplitude is gated on and off at the
  target rate, producing the strongest and most spectrally direct entrainment drive.
The motivation for adding these is measurement, not listening experience. The
best-controlled literature (Orozco Perez, Dumas & Lehmann, 2020; Ross et al., 2014)
shows that **monaural and isochronic stimuli entrain the cortex more strongly than
binaural beats**, because they carry the rhythm as real acoustic energy. That makes
them better **probes** for demonstrating a measurable EEG response on frontal
hardware, even though binaural beats remain the better **product** audio for
pleasantness and maskability.
 
**Key framing to preserve throughout:** binaural = best listening experience;
isochronic = strongest, cleanest EEG signature; monaural = middle ground. Do not try
to make one stimulus serve both the product-audio role and the measurement-probe
role — their requirements (especially around background audio) pull in opposite
directions (see §5).
 
---
 
## 2. Signal model summary
 
| Stimulus | # carriers | How the beat arises | Beat physically in air? | Modulation depth | Needs envelope ramp? |
|---|---|---|---|---|---|
| Binaural | 2 (one per ear) | Central neural combination of L/R | No | Shallow, fixed by physics | No (interference envelope is already smooth) |
| Monaural | 2 (summed) | Acoustic interference before playback | Yes | Shallow, fixed by physics | No (same smooth interference envelope) |
| Isochronic | 1 | Amplitude gating of a single carrier | Yes | Adjustable, up to 100% | **Yes (mandatory)** |
 
Definitions used below:
 
- `f_carrier` — carrier (tone) frequency in Hz.
- `f_beat` — target entrainment / modulation frequency in Hz.
- `fs` — audio sample rate (e.g. 44100 or 48000 Hz).
- `A` — output amplitude (0..1).
---
 
## 3. Monaural beats
 
### 3.1 Concept
 
Monaural beats reuse the binaural two-tone idea but **sum the two tones into a single
waveform first**, then send that identical waveform to both ears. Removing the
left/right separation is precisely what moves the beat out of the head and into the
air: the summed signal has a real amplitude envelope oscillating at `f_beat`.
 
The two tones are placed symmetrically around the carrier (Convention B, matching the
existing binaural design):
 
```
f_low  = f_carrier - f_beat/2
f_high = f_carrier + f_beat/2
```
 
### 3.2 Generation
 
```
# per sample n, t = n / fs
tone_low  = sin(2*pi * f_low  * t)
tone_high = sin(2*pi * f_high * t)
 
mono = 0.5 * (tone_low + tone_high)   # sum, then halve to avoid clipping
```
 
The `0.5` prevents the summed peak (which reaches 2.0 when both tones align) from
exceeding full scale. The resulting `mono` signal naturally contains an amplitude
envelope at `f_beat` — this is the audible beat.
 
### 3.3 Delivery and background
 
Send the **same** `mono` signal to left and right:
 
```
left_out  = A * mono + background_left
right_out = A * mono + background_right
```
 
Background handling is **identical to binaural** — a soundscape can be layered freely,
because the beat is a low-frequency amplitude envelope that a background does not
erase. (Contrast with isochronic, §5.) If the existing binaural pipeline already
supports background mixing, monaural reuses it unchanged; the only difference upstream
is summing the two tones instead of routing them to separate channels.
 
### 3.4 Notes
 
- Interaural level should be matched (both ears get the identical mix), so no
  per-ear loudness balancing is needed as it is for binaural.
- Monaural beats do **not** require headphones to work (the beat is acoustic), though
  headphones still help isolate from room noise.
---
 
## 4. Isochronic tones
 
### 4.1 Concept
 
An isochronic tone is a **single carrier** whose amplitude is switched on and off
`f_beat` times per second. There is no second frequency and no interference — the
rhythm is imposed directly by an amplitude-gating envelope. Because the gate can go
all the way to silence, the modulation depth can reach 100%, which is why isochronic
tones are the strongest entrainment drive and the cleanest to detect on EEG.
 
### 4.2 Parameters
 
Relative to the beat generators, isochronic **adds three controls** and **drops the
second carrier**:
 
| Parameter | Range | Role | Default |
|---|---|---|---|
| `f_carrier` | ~100–500 Hz | Pitch of the single gated tone. Keep low for strong phase-locking. | 250 Hz |
| `f_beat` | target rate | On/off cycles per second = entrainment target. | task-dependent |
| `depth` | 0.0–1.0 | How far the "off" phase drops. 1.0 = full silence (max entrainment, max audible pulsing); lower = gentler, weaker. | 1.0 for probe |
| `duty` | 0.1–0.9 | Fraction of each cycle the tone is on. 0.5 = equal on/off. Shorter = clickier; longer = blurs toward continuous. | 0.5 |
| `ramp_ms` | 2–10 ms | Fade time on each on/off transition. **Mandatory** — see §4.4. | 5 ms |
| `A` | 0.0–1.0 | Output amplitude. | as existing |
 
`depth` and `duty` do not meaningfully exist for binaural/monaural beats (their
envelope is fixed by the physics of two equal tones). `depth` is the main
comfort-vs-strength dial; `duty` shapes the character of the pulse.
 
### 4.3 Generation
 
Build a gating envelope `g(t)` in [0,1] that cycles at `f_beat`, then multiply the
carrier by it:
 
```
phase_in_cycle = (t * f_beat) mod 1.0        # 0..1 within each beat cycle
 
# base gate: on for the first `duty` fraction of the cycle, off after
gate = 1.0 if phase_in_cycle < duty else 0.0
 
# apply raised-cosine ramps at the on and off edges (see 4.4)
gate = apply_ramp(gate, phase_in_cycle, duty, f_beat, ramp_ms)
 
# depth: scale so the "off" level is (1 - depth) instead of 0
env = (1 - depth) + depth * gate       # env in [1-depth, 1]
 
iso = A * env * sin(2*pi * f_carrier * t)
```
 
### 4.4 Envelope ramping (mandatory)
 
A **hard rectangular** gate creates discontinuities at every on/off transition.
Discontinuities in audio produce two problems:
 
1. **Audible clicks** — the stimulus sounds harsh and buzzy.
2. **Spectral splatter** — a hard square gate spreads sideband energy across *many
   harmonics* of `f_beat`. A 6 Hz square-gated probe would inject 12, 18, 24 Hz …
   lines into the EEG-relevant range, muddying the very spectral readout the probe
   exists to produce.
The fix is to ramp each transition with a short **raised-cosine (Hann) edge** of a few
milliseconds instead of a vertical edge. A raised-cosine gate is a genuine amplitude
modulation whose spectrum is the carrier plus clean sidebands at
`f_carrier ± f_beat` — spectrally tidy, so any detected EEG response is unambiguously
at the target frequency.
 
```
# convert ramp_ms to a fraction of the beat cycle
ramp_frac = (ramp_ms / 1000.0) * f_beat
 
# within the "on" region, fade in over the first ramp_frac and
# fade out over the last ramp_frac using a raised-cosine (0.5 - 0.5*cos) shape
```
 
**Treat the ramp as non-optional.** It is the single detail that separates a clean
pulsed tone from an annoying clicking artifact, and it keeps the stimulus spectrally
clean enough for interpretable analysis.
 
### 4.5 Delivery
 
Isochronic is mono by nature — the same gated signal goes to both ears:
 
```
left_out  = iso + background_left     # see §5 on whether to include background
right_out = iso + background_right
```
 
---
 
## 5. Background audio: the decisive difference
 
This is where isochronic diverges sharply from the beat generators.
 
- **Binaural / monaural:** background is nearly free. The beat is a smooth
  low-frequency envelope (binaural: in the head; monaural: in the air) that a
  soundscape layered on top does not disturb. This is exactly why binaural beats were
  chosen as the product's relaxation audio — maskability and pleasantness.
- **Isochronic:** background *directly competes* with the mechanism. The entrainment
  **is** the audible on/off modulation, so anything filling the silent gaps reduces
  the effective modulation depth the ear and cortex actually see. A rich background
  can partially mask the pulsing that does the work, weakening the response.
### Rule of thumb
 
| Use case | Background? | Rationale |
|---|---|---|
| Isochronic **measurement probe** | **None** — pure gated tone | Maximize modulation depth and produce the cleanest spectral line. A clinical-sounding 30 s is fine; the probe is not for enjoyment. |
| Isochronic as **product audio** | Light at most | Any meaningful background undermines the modulation; the residual audible pulsing is what makes isochronic less pleasant for long wind-downs. |
| Binaural/monaural product audio | Yes | Beat survives layering; pleasant for 30-min sessions. |
 
**Recommended split:** build isochronic as a **no-background probe** for demonstrating
a measurable frontal-EEG response (especially at 40 Hz with a self-referenced F-test),
and keep **binaural + background** as the product audio. Do not force one stimulus to
serve both roles.
 
---
 
## 6. Gating arbitrary music (optional, product-side only)
 
Applying an isochronic-style gate to a full song *does* impose amplitude modulation at
the gate rate, and the cortex will track that envelope (this is rhythmic/musical
entrainment). However, gated music is **not** equivalent to an isochronic tone:
 
- An isochronic tone gates a **single pure carrier** → spectrum is one clean line plus
  tidy sidebands → a clean, unambiguous single-frequency drive.
- A song is **broadband** and already amplitude-varying. Gating it stamps the
  modulation onto every component, but that drive rides on top of the music's own
  irregular envelope, so the entrainment is **muddier and weaker**, and the spectrum
  is not clean.
Therefore:
 
- **Product relaxation audio:** gated music is a legitimate option.
- **Measurement probe:** gated music is the wrong tool — its broadband content is
  exactly what prevents a clean spectral readout. Use a pure isochronic tone instead.
(Same principle as §5, one layer up: broadband content muddies the probe the way a
background does.)
 
---
 
## 7. Implementation checklist
 
- [ ] **Monaural:** sum symmetric tone pair (`f_carrier ± f_beat/2`), halve to avoid
      clipping, route identical mix to both ears. Reuse binaural background pipeline.
- [ ] **Isochronic:** single carrier × gating envelope; expose `depth`, `duty`,
      `ramp_ms` as parameters.
- [ ] **Ramp every isochronic transition** with a raised-cosine edge (2–10 ms).
      Never ship a hard square gate.
- [ ] Verify the isochronic output spectrum shows the carrier plus clean
      `f_carrier ± f_beat` sidebands and **no** strong harmonics of `f_beat`
      (confirms ramping is working).
- [ ] Run the isochronic **measurement probe with no background**.
- [ ] Keep **binaural + background** as the product audio path.
- [ ] Low carrier (~250 Hz) for low-frequency targets; the 40 Hz probe is the most
      likely to yield a detectable frontal-EEG response.
---
 
## 8. Why this matters for the device
 
The strategic tension worth stating plainly: the modality that **sells** (binaural,
for its pleasant maskable audio and better-evidenced relaxation outcomes) is not the
modality that **measures** best (isochronic/AM, for its strong, spectrally clean,
physically-driven EEG signature). These implementations let the product use each
modality for what it is actually good at — binaural for the listening experience,
isochronic for demonstrating that the brain (and the sensor) is responding.
 
