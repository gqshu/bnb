# Addendum — Product Design Decisions

*Companion to the main feasibility assessment. Scope narrowed per direction: relaxation / down-regulation only (no gamma-focus mode for now). Adaptive stepwise strategy from the project patent is treated as the design baseline. This addendum answers three specific product-decision questions.*

## Scope note
Because we are now committing to down-regulation only (Recharge, Wind Down, Sleep), two things simplify: (1) the Fp1/TGAT gamma-detection risk from the main assessment is out of scope — theta/alpha/delta are all reliably measurable at Fp1; (2) all beat targets sit at ≤14 Hz, comfortably inside the well-perceived binaural range. This is the favorable regime for both the science and the hardware.

---

## Q1 — How to choose the carrier frequency x (the "x, x+Δ" pair)?

Separate this into two distinct decisions that are often conflated:

**(A) The beat difference Δ = the target you're training toward.** This is not free — it *is* the goal. If you want to train the brain toward 4 Hz (delta/sleep), Δ = 4 Hz. The patent's stepwise method is exactly this: start Δ near the user's measured baseline dominant frequency f, then step Δ down 1 Hz at a time toward the target while EEG confirms the brain is following. So "train the user for 20" only applies to up-regulation; for our relaxation modes Δ starts at the measured baseline (often 8–18 Hz beta/alpha) and descends toward 4–8 Hz. Δ should always be <30 Hz (above that the two tones separate into distinct pitches and the beat percept collapses) — never an issue for us.

**(B) The carrier x = the base pitch both tones sit near.** This is the actual free parameter, and the literature gives a clear answer: **x should be roughly 400–500 Hz, and can be held constant across the whole session.** Reasons, in order of strength:

- **Perceptual optimum ~400–500 Hz.** Best binaural-beat perception occurs at carrier tones between 400 and 500 Hz (Licklider et al. 1950; Perrott & Nelson 1969), which is why the standard neuroscience protocols (e.g. the brainstem-connectivity study, PMC7082494) center carriers at ~400 Hz. Below ~200 Hz and above ~900 Hz the percept weakens; above ~900 Hz binaural beats are essentially not usable.
- **The Oster Curve is about perception, not entrainment — don't over-fit to it.** The Oster Curve suggests slightly lower "optimal" carriers for low targets (e.g. ~160–210 Hz for theta, ~230 Hz for a 10 Hz alpha beat). But Oster measured *detection thresholds*, not entrainment outcomes, and controlled entrainment studies get strong results at 250 Hz for a 6 Hz target even though Oster would nominate ~200 Hz. Practical implication: precise carrier tuning per target is not worth the engineering complexity. A fixed ~400–440 Hz carrier is well-supported and simpler.
- **A fixed carrier is a feature, not a compromise.** Because Δ is what we step (by shifting only one ear's tone), holding x constant means the user hears a steady, stable pitch while the beat slowly slows down — smoother and less distracting than moving both tones. This matches the patent's "η₀ fixed in both ears, η₁ = η₀ + f modified in one ear" architecture.
- **One caveat for the Sleep (≤4 Hz) mode:** at very low Δ the beat becomes a slow "moving sound" rather than a fluctuation, and cortical following responses attenuate at low frequencies. The 0.25 Hz sleep study (main assessment) found benefits without confirmed entrainment — i.e. below ~4 Hz you are relying more on relaxation than on true following. Keep the Sleep target at 4 Hz (as designed) rather than chasing sub-delta.

**Recommendation:** Fixed carrier x ≈ 400–440 Hz for all modes. Δ = the mode's target, reached via the patent's stepwise descent starting from the individually-measured baseline f. Do not vary x per user or per target in v1; log it as a later optimization variable if desired.

**A subtlety worth flagging:** loudness must be matched between the two ears (the reference protocols sum-and-halve the tones to equalize perceived loudness). An interaural loudness imbalance will be heard as the sound lateralizing to one side and can break immersion. This is a DSP detail the audio engineer must get right.

---

## Q2 — Waveform / sound form: what is most effective?

This is the most important finding in this addendum, and it involves a real tension between *efficacy* and *tolerability*:

**Pure binaural beats are the weakest entrainment stimulus but the most pleasant.** Head-to-head cortical-response studies are consistent: amplitude-modulated / isochronic tones produce the *strongest* cortical steady-state response, monaural beats are intermediate, and pure binaural beats produce the *weakest* (Schwarz & Taylor 2006, *Clin. Neurophysiol.*; Pratt et al. 2010, *BMC Neuroscience*). The physical reason is modulation depth: a binaural beat has an effective modulation depth of only ~3 dB (a ~2:1 loud-to-quiet ratio), because the "beat" is reconstructed internally from phase differences rather than physically present in the sound. An isochronic tone physically switches on and off, giving ~50 dB depth (a ~100,000:1 ratio) and sharp onset transients that the auditory cortex locks onto strongly.

**But the strong stimulus is the annoying one.** Isochronic tones are an overt rhythmic click/buzz that many users find unsettling, and — critically — **they cannot be masked under music or soundscapes without destroying the amplitude transients that make them work.** Binaural beats, being two smooth continuous tones, blend into a warm, barely-perceptible pulsation that can sit under a soundscape for a 20–30 minute session. Since our product is (a) built around long relaxation sessions, (b) explicitly designed to layer soundscapes on top (per the UX flow), and (c) closed-loop — meaning we don't *need* the stimulus to be strong because the EEG tells us in real time whether the brain is following and we adjust — the binaural choice is defensible *specifically because* we have the feedback loop to compensate for the weaker open-loop drive.

**Design options, in rough order of recommendation for a relaxation product:**
1. **Binaural beats (pure sine carriers), closed-loop guided** — most pleasant, maskable, weakest raw drive but the EEG loop compensates. Best fit for our long-session relaxation use case. *Recommended default.*
2. **Binaural + subtle amplitude modulation ("modulated binaural")** — add a shallow AM envelope at the beat frequency to strengthen the cortical response while keeping the beat maskable. A reasonable middle path worth A/B testing.
3. **Monaural beats** — stronger than binaural, still relatively smooth, but require careful loudness handling and lose the "one tone per ear" spatial quality; also thinner research base.
4. **Isochronic tones** — strongest entrainment, but poor tolerability for long sessions and incompatible with soundscape layering. *Not recommended as the default for a relaxation product*, though it could be an optional "intensive" mode for users who tolerate it.

**Carrier waveform itself:** use pure sine tones for the carriers. Complex/harmonic-rich carriers add spectral energy that muddies the beat percept; the entire binaural literature uses sines. Keep carriers in the mid-low register — high-pitched tones are associated with alarm/arousal and work against relaxation.

**Recommendation:** Ship pure-sine binaural beats as the default, leaning on the closed loop to compensate for the weaker raw drive. Prototype a shallow-AM "modulated binaural" variant and A/B it against pure binaural on both entrainment speed (EEG) and self-reported pleasantness. Reserve isochronic as an optional intensive mode, not the default.

---

## Q3 — Soundscape personalization: is "model the search space + self-improve on user data" sound?

**Short answer: yes, this is a sound and well-precedented approach — it is a contextual multi-armed bandit / preference-optimization problem, which is a mature ML paradigm. But the value of it depends entirely on defining the right search space and the right reward signal, and there are two specific traps to design around.**

### The search space is well-characterized in the MIR / music-emotion literature

You do not need to invent the feature space — decades of Music Emotion Recognition (MER) research have mapped which acoustic features drive the two dimensions that matter for relaxation (low arousal, positive valence):

- **Arousal (the primary axis for relaxation)** is driven mostly by *energy and temporal/rhythm features*: overall energy/loudness, tempo, onset density, and dynamics. Energy is repeatedly the single strongest arousal predictor. For relaxation you want low values on these.
- **Valence (keep it positive)** is driven mostly by *spectral and tonal/harmonic features*: mode (major/minor), harmony, spectral characteristics. Sleep-music research (Kirk & Timmers 2025) found sleep/relax playlists need to be *positively valenced* — relaxation is release from tension, not merely low activation.
- **Timbre / spectral centroid**: lower spectral centroid = "softer, darker, calmer" sound; producers deliberately roll off high frequencies for calm. Spectral centroid and loudness were top discriminators between "sleep" and "relaxing" playlists.
- **Concrete relaxation levers from the pentatonic/monaural EEG study (PMC11056517):** slow tempo (they found ~0.2 Hz oscillation — ~12 bpm — maximally relaxing, a "supernormal" slow stimulus), minor pentatonic scales (no semitone tension), mid-low register, simple pure timbres.
- **Sound category is itself a dimension:** musical excerpts vs broadband noise (white/pink/brown) vs nature sounds (rain, water, wind) vs beats. These have systematically different relaxation profiles — nature/water sounds lower heart rate; pink noise deepens slow-wave sleep at low volume.

So a practical **search space** for AI-generated soundscapes could be parameterized along roughly: {sound category} × {tempo/onset-density} × {energy/loudness envelope} × {spectral centroid / brightness} × {mode & harmonic complexity} × {register} × {texture density} × {presence and level of overlaid nature/noise bed}. That's a modest, well-motivated feature space — exactly the kind a bandit can search efficiently.

### The self-improvement mechanism is standard and appropriate

Framing this as "model the space, then use user data to self-improve" is textbook **contextual multi-armed bandit / preference learning**:

- Personalized interactive music recommendation has been formulated as an exploration–exploitation bandit since Wang et al. (2014, *ACM TOMM*), using Bayesian preference models and Thompson sampling to balance exploring a user's unknown taste against exploiting what works.
- More recent **affective recommender systems** treat emotion/state as a dynamic variable and learn policies that *steer listeners toward a desired state* (surveyed in Hasan & Bunescu 2025) — which is precisely our goal (steer toward relaxation), not just "recommend a song they'll like."
- We have an unusually good reward signal that most music recommenders lack: **the EEG itself.** Rather than relying only on thumbs-up/down or dwell time, the closed loop gives an objective outcome — time-to-relaxed-state, depth reached, stability, deviations — per soundscape per session. That is a strong, low-noise reward for the bandit and is the core defensible advantage of the bundle.
- **Cold-start** (new user, no data) is handled the standard way: initialize from population priors keyed on the MER features above (start everyone near "known-relaxing" defaults) and personalize from there. Contextual bandits (e.g. LinUCB-style) are specifically designed for this.

### Two traps to design around

1. **Reward attribution / confounding.** The EEG outcome is influenced by *both* the binaural-beat guidance *and* the soundscape simultaneously. If you optimize soundscape on raw EEG outcome, the bandit may credit the soundscape for what the beat protocol did. Mitigate by holding the beat protocol fixed while exploring soundscape, or by modeling them as separate factors. Also weight subjective feedback (post-session "how do you feel," per the UX) alongside EEG so you're not optimizing a purely physiological proxy that diverges from felt experience.
2. **Feedback-loop / over-exploitation bias.** Continuously-trained recommenders are known to collapse onto a narrow set of already-successful items and stop exploring (the algorithmic-bias / "rotting bandit" problem; Guo et al. 2020). For a wellness product this shows up as *habituation* — the same soundscape every night gets stale and less effective. Two mitigations: (a) keep an explicit exploration budget (Thompson sampling / ε-greedy) so novel soundscapes keep getting tried; (b) note the patent's own observation that varying the stimulus avoids acclimatization — novelty may itself be therapeutic here, which conveniently aligns with maintaining exploration.

### AI-generated music specifically

Generative audio is a good fit because it lets you *sample any point in the parameterized search space on demand* rather than being limited to a fixed track library — the bandit can request "slightly lower tempo, warmer timbre, add light rain bed" and get a fresh render. Caveat: familiarity aids relaxation (familiar music produces fewer prediction errors and lower alertness; Relaxation Music Dataset work), so purely novel AI generation every time may lose the familiarity benefit. A hybrid — evolve within a recognizable per-user "sound signature" rather than jumping around the space — likely beats both a static library and unconstrained generation.

**Recommendation:** Yes, proceed with the model-the-space + self-improve approach. Use the established MER arousal/valence feature space above to define the search dimensions; initialize from population relaxation priors for cold-start; run a contextual bandit (Thompson sampling for healthy exploration) with a **blended reward** = objective EEG relaxation metrics + subjective post-session rating; separate the soundscape factor from the beat protocol to avoid confounded credit; and constrain generation to evolve within a per-user sound signature to preserve familiarity while still exploring.

---

## Summary of decisions

| Question | Recommendation | Confidence |
|---|---|---|
| Carrier x | Fixed ~400–440 Hz sine, held constant; Δ = target, stepped from measured baseline | High (well-established perceptual optimum) |
| Waveform | Pure-sine binaural default (pleasant + maskable + closed-loop compensates); prototype shallow-AM variant; isochronic only as optional intensive mode | Medium-High (efficacy vs tolerability tradeoff is real) |
| Soundscape personalization | Contextual bandit over MER-defined feature space; blended EEG + subjective reward; population cold-start; per-user sound signature | Medium-High (method is mature; success hinges on reward design) |
