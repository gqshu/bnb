# Backend Plan — Streaming Mixed Binaural Audio

*Planning doc for the streaming backend. Distills the architecture discussion into
decisions, rationale, and an incremental build order. Companion to the product docs
([bnb_product1.md](bnb_product1.md), [bnb_product2.md](bnb_product2.md)); this one is
implementation-facing.*

## Goal

A backend that streams a live, personalized relaxation mix to one user's device:

- start from a **background `.wav`** file and a session **length**;
- **loop** the background for the duration of the session;
- **mix** a phase-continuous binaural beat over it, with the beat frequency and the
  relative volumes controllable during the session;
- expose a **control endpoint** to update the mix and a **streaming endpoint** a browser
  `<audio>` tag or a mobile native player can consume without disruption.

Scope is down-regulation only (Recharge / Wind Down / Sleep Prep), so all beat targets sit
at ≤ 14 Hz — the favorable regime for both the perception and the hardware.

## What we validated before designing

Two experiments settled the two open questions (scripts in the session history, not
committed):

1. **Codecs preserve the beat.** MP3, Vorbis, and Opus at default bitrates keep each ear's
   tone in its own channel (cross-ear leakage 60–87 dB down) and preserve the interaural
   *phase* relationship that the beat percept actually is (recovered beat 8.000 Hz, phase
   jitter < 0.005 rad). The beat also survives being buried under a louder pink-noise bed
   at every level tested. Lossy streaming is therefore safe.
2. **Consequence:** we can encode the mix server-side with a lossy codec and stream it; we
   do not need lossless transport to protect the effect.

## Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Stream model | **Per-session pipeline**, not a shared broadcast | The beat is personalized per user; nothing is shareable between users, so broadcast/CDN caching buys nothing. |
| Transport | **Continuous chunked HTTP** (internet-radio style), not HLS/DASH | HLS's segment latency is pointless here and its CDN-caching win is void for personalized streams. Continuous streaming fits a live, per-user mix. |
| Default codec | **MP3** (`audio/mpeg`), Opus offered where supported | Chunked MP3 plays in every browser `<audio>` including Safari/iOS; Ogg/Opus does not play via `<audio>` on Safari. Opus is the efficient upgrade for clients that advertise it. |
| Beat updates | **Gradual glide**, not instant | We relaxed the near-realtime requirement. A frequency glide is click-free by construction, is the physically correct "beat slows down" behavior, and matches the stepwise-descent design. |
| Pacing | **TCP backpressure**, no wall-clock scheduler | Once updates may land a couple of seconds late, the socket's own flow control paces generation a small buffer ahead. Removes an entire component. |
| Server framework | **FastAPI / Starlette** async + numpy mix + **PyAV** streaming encode | Async `StreamingResponse` respects backpressure; PyAV encodes blocks incrementally (soundfile only writes whole files). |

## Architecture

### The core simplification

Relaxing the realtime constraint + assuming **one active listener per session** collapses
the design: the streaming endpoint's async generator *is* the whole pipeline. No background
worker, no real-time clock, no fan-out ring buffer.

```
POST /sessions {background, length}      -> create session, return id
POST /sessions/{id}/mix {setpoints}      -> overwrite setpoints, return immediately
GET  /sessions/{id}/stream               -> async generator:
      loop:
        block = mix(next background block (looped),
                    beat rendered from the phase-continuous glided oscillator,
                    at the current smoothed volumes)
        yield encode(block)              # await suspends here under backpressure
```

The control endpoint only writes **setpoints**. The generator reads them at the top of
each block and eases the live parameters toward them. TCP backpressure keeps generation a
small buffer ahead of playback; that buffer depth is also the update latency (a few
seconds), which is acceptable now.

### Everything is a smoothed parameter

Model every control as a **setpoint the live value eases toward**, per block:

- target **beat_hz** — reached via a per-sample frequency **glide** (see below);
- target **beat volume** and target **bed volume** — ramped to avoid zipper noise;
- master gain — ramped for **loop-seam declick**, **session start fade-in**, and
  **end fade-out**.

Build one "smoothed parameter" primitive and reuse it for all four rather than writing
bespoke click-avoidance each time.

### The glide (the one subtle bit)

Keep a **single running phase accumulator**. Do not recompute `sin(2π·f·t)` when `f`
changes — that jumps the phase (audible click) and corrupts the interaural phase the beat
depends on. Instead advance phase from an interpolated frequency:

- interpolate frequency **per sample** across the block (`f = linspace(f_old, f_new)`);
- `phase = phase0 + cumsum(2π · f / sample_rate)`; carry the last phase into the next block.

Per-sample interpolation (not once-per-block stepping) keeps the glide smooth independent
of block size. Only the right ear glides; the left ear holds the carrier.

### Session lifecycle

- `POST /sessions` decodes the background to a float32 array in memory, initializes
  setpoints and phase, records `length`, starts the clock.
- Background loops via a wrapping read index; crossfade a few ms across the seam (same
  ramp primitive) unless assets are authored loop-ready.
- At `length`, fade out and end the stream.
- The stream is live with no seek; on a dropped connection the client **auto-reconnects**
  and resumes at "now," the buffer hiding the gap.

## The one tradeoff, stated honestly

Client buffer depth is simultaneously the **dropout protection** and the **update
latency** — they pull opposite ways and you cannot maximize both. For this product the
resolution is clear: the beat steps down slowly over minutes, so a ~1–2 s delay on a mix
change is imperceptible, while a mid-session dropout is very perceptible. Bias toward the
buffer; market the updates as "responsive," not "instant."

## Scaling notes (later, not v1)

- **One encoder per active session** → CPU-bound, but libmp3lame at 48 kHz stereo runs far
  faster than realtime, so a single box handles many tens to low hundreds of sessions.
- State lives in memory in a specific worker → the service is **not stateless**; horizontal
  scaling needs **session affinity** (a session pinned to one process).
- **Multiple simultaneous listeners on one session** (phone + laptop) is the case that
  brings back the decoupled worker + fan-out ring buffer and an explicit clock, because
  generation can no longer be driven by a single connection's backpressure. Out of scope
  for v1.

## Open questions

- Confirm **one-listener-per-session** holds; if a user routinely streams to two devices at
  once, plan the worker + fan-out variant up front.
- **iOS background / lock-screen playback**: if this becomes a hard requirement it argues
  for HLS *on iOS specifically* (continuous chunked audio may be suspended when backgrounded).
- Should the glide duration be a session default or per-update in the control payload?
- Where the **stepwise-descent controller** lives: does the backend own the EEG loop and
  drive its own setpoints, or does an upstream service post setpoints to `/mix`?

## Build order (test-first)

Reuses the existing `bnb.tone` module and its render/WAV foundation.

1. **Smoothed-parameter primitive** — ease a value toward a setpoint per block. Unit test:
   monotonic approach, reaches the setpoint, no overshoot.
2. **Phase-continuous glided oscillator** — extend `bnb.tone` with a stateful generator.
   Unit test: phase is continuous across a frequency change (no discontinuity), and the
   recovered beat matches the new setpoint after the glide completes.
3. **Streaming mixer** — loop background + glided beat + smoothed volumes into blocks.
   Unit test: loop seam is click-free; volumes track setpoints; no clipping.
4. **PyAV MP3 encode** of the block stream. Test: decodes back to the expected tones.
5. **Endpoints** — `POST /sessions`, `POST /sessions/{id}/mix`, `GET /sessions/{id}/stream`.
   Integration test with an async client: start a session, stream, POST a new beat, assert
   the decoded tail reflects the new beat.
6. **Manual check** — open `/stream` in a browser `<audio>` tag, POST a mix change, confirm
   the beat glides without a dropout.
