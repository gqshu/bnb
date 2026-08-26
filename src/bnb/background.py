"""Background-soundscape library: the substrate x style taxonomy.

This is the provider-agnostic core of the background media library described in
docs/background_music.md. It turns a (substrate, style) *signature* into the three
things a render needs:

- a prose prompt built from the §4 base template + style modifier (both ElevenLabs
  and Stable Audio 3 read this; :func:`prompt_for_provider` adapts it further),
- the composition plan (positive/negative global styles, one loop section) —
  ElevenLabs-specific structure, translated by :func:`composition_plan_for_model`,
- the per-track metadata record (§3 schema), minus the measured MER vector that
  the objective-feature extraction pipeline fills in after the render.

Alongside the substrate x style grid, :data:`SPECIAL_GROUPS` holds keyword-driven
categories that sit *outside* the grid entirely (e.g. ``natural_sounds``: a plain
keyword like "rain" rather than a substrate/style pair). :func:`build_keyword_signature`
resolves those the same way :func:`build_signature` resolves grid cells.

Scope is down-regulation only, so every continuous MER axis (tempo, energy,
brightness, harmony, texture, register, nature-bed) is authored biased low. The
spec is written by scripts/plan_background.py and rendered by
scripts/render_background.py; nothing here imports the ElevenLabs or Stable Audio
SDKs, so the taxonomy stays importable and testable without an API key.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

# Applies to every down-regulation track (§4). Kept as prose (prompt) and as
# discrete style tags (composition plan) because ElevenLabs consumes both.
NEGATIVE_PROMPT = (
    "bright, harsh, energetic, fast, distorted, drums, buildup, EDM, sudden transitions, "
    "hiss, static, white noise, busy, cluttered, dense, chattering"
)

NEGATIVE_GLOBAL_STYLES: tuple[str, ...] = (
    "bright",
    "aggressive",
    "drums",
    "vocals",
    "fast",
    "buildup",
    "hiss",
    "white noise",
)

# Density is the axis a generative engine overshoots on its own: asked for a forest it
# writes a dawn chorus. The MER axes already say "very sparse", but that is a texture
# word the models read as timbre, so the limit is stated countably instead — and in the
# *positive* prompt, since the negative one only reaches the model under guidance (cfg),
# which the distilled checkpoints run without.
#
# Which limit depends on how the sound is made, and getting this wrong is worse than
# saying nothing: "long stretches of near-stillness" applied to a continuous cricket bed
# doesn't quiet it, it breaks the smooth wash into discrete chirps with gaps — measurably
# *more* eventful. So a bed is told to stay even, and only sounds built from separate
# events are told to space them out.
RESTRAINT = (
    "Restrained and uncrowded: at most one or two things sounding at any moment, "
    "occasional and far apart, with long stretches of near-stillness between them"
)
STEADINESS = (
    "Even and unbroken throughout, one continuous texture with nothing stepping forward "
    "out of it, no separate events and no layering"
)

# The two clauses above are the *stillness* bound, and they are why a melodic request came
# back as a wash: "long stretches of near-stillness between them" and "nothing stepping
# forward out of it, no separate events and no layering" forbid a phrase from continuing,
# whatever `development` asks for. So a melodic development gets its own bound — still
# countable (that is what stops the engine's dawn-chorus overshoot, which was always the
# point), but bounding *clutter* rather than bounding motion, and with the near-stillness
# and no-layering language dropped rather than merely softened.
# ...and it is a *per-goal* bound, because this clause and the development fragment are
# the two most musically salient sentences in the prompt — while both goals shared them
# verbatim, relax and focus renders came back near-indistinguishable however much the
# tempo/timbre adjectives differed around them.
RELAX_FLOWING = (
    "Full but unhurried: three or four gentle layers sounding together, chords and notes "
    "overlapping and sustaining into one another with no gaps or empty stretches, always "
    "consonant and easy to listen to"
)

# Focus is denser *and* gap-free, and the gap-free half is a hard technical requirement,
# not a taste: focus ships as `am_music`, where the bed itself is the carrier the
# entrainment envelope multiplies (tone.render_am_music). Silence in the bed is silence in
# the stimulus — the modulation has nothing to ride on, so the drive drops out for as long
# as the gap lasts and `am_depth_by_band` measures an intermittent, weaker signal. A
# down-regulation bed can breathe; an AM carrier may not.
FOCUS_FLOWING = (
    "Full and continuous: four or five layers always sounding at once — a clear lead line "
    "on top, warm chords beneath it, a soft steady pulse and a sustained bass note holding "
    "underneath — so sound is present at every single moment from beginning to end. It "
    "never thins out, never drops to silence, never leaves a gap or a rest between phrases"
)

# The MER texture words were authored for a library that never moved, so "very sparse
# texture" ends up in the same sentence as FLOWING's "three or four gentle layers" —
# exactly the kind of self-contradiction that made the lo-fi cells unlistenable. Under a
# melodic development the *prose* word relaxes one step. `requested_features
# ["texture_density"]` is deliberately untouched: that is the MER coordinate the bandit
# consumes and the taxonomy's own address for the cell, and this is a prompt fix, not a
# re-coordinate.
TEXTURE_WHEN_MELODIC: dict[str, str] = {
    "very sparse": "uncluttered but full",
    "sparse": "uncluttered but full",
    "steady, unchanging": "steady and full",
}

MELODIC_DEVELOPMENT = frozenset({"slow_swell", "motif_evolving"})
"""The ``development`` values that carry a tune, and so need :data:`FLOWING` room to play
it in. ``static`` keeps the original stillness bound; so do the keyword nature beds
(:func:`build_keyword_signature`), which have no ``development`` axis at all — a rain bed
must never be told to grow layers."""


def density_clause(
    event_driven: bool, development: str | None = None, flowing: str | None = None
) -> str:
    """The density limit that suits how this sound is produced, and how much it develops.

    ``development``/``flowing`` omitted (the special/keyword path, which has no goal and no
    development axis) keeps the original stillness bound — a rain bed must never be told to
    grow layers.
    """
    if development in MELODIC_DEVELOPMENT and flowing is not None:
        return flowing
    return RESTRAINT if event_driven else STEADINESS


# --- Axis A: substrate (what the sound physically is) -----------------------


@dataclass(frozen=True)
class Substrate:
    """One physical sound type (§2, Axis A).

    ``body`` fills the "[substrate + instrumentation]" slot of the base template;
    the seven axis fields fill the continuous MER slots. ``requested`` is the
    categorical MER coordinate recorded in metadata (§3 ``requested_features``);
    styles may override its ``mode`` / ``nature_bed`` without touching the axis.

    These fields are the substrate's **relax** (down-regulation) defaults — the goal this
    library shipped with first. ``goals`` says which goals this substrate is usable for at
    all (a physical sound can be goal-inappropriate outright, e.g. long resonant bowl decays
    don't suit focus); ``focus_overrides`` carries the deltas the **focus** goal applies on
    top of the relax fields (see :func:`_resolve_substrate_for_goal`), so substrates that
    support both goals don't need two full definitions.
    """

    name: str
    short: str  # compact tag for track_id
    body: str
    instrumentation: tuple[str, ...]
    style_tags: tuple[str, ...]  # substrate's contribution to positive_global_styles
    tempo: str
    energy: str
    timbre: str
    harmony: str
    texture: str
    register: str
    requested: dict[str, Any]
    event_driven: bool = False  # built from separate strikes/notes rather than one wash
    goals: frozenset[str] = field(default_factory=lambda: frozenset({"relax", "focus"}))
    focus_overrides: dict[str, Any] = field(default_factory=dict)


SUBSTRATES: dict[str, Substrate] = {
    "drone": Substrate(
        name="drone",
        short="drone",
        body="Sustained pad and drone tones with no rhythm",
        instrumentation=("sustained_pad", "low_drone"),
        style_tags=("ambient drone", "sustained pad"),
        tempo="arrhythmic, no pulse",
        energy="very low",
        timbre="warm-dark",
        # Stillness is `development="static"`'s job to ask for now; hardcoding it here
        # contradicted every other value of the axis (same bug as Change 1's melody bans).
        harmony="warm and consonant",
        texture="very sparse",
        register="low",
        requested={
            "tempo_bpm": None,
            "energy": "very_low",
            "spectral_centroid": "warm_dark",
            "mode": "static_drone",
            "register": "low",
            "texture_density": "very_sparse",
            "nature_bed": "none",
        },
        focus_overrides={
            "tempo": "a slow, steady pulse underneath, unchanging meter",
            "energy": "low-moderate",
            "timbre": "warm-neutral",
            "texture": "sparse, with one subtle repeating rhythmic layer",
            "register": "low-mid",
            "requested": {"energy": "low", "spectral_centroid": "warm_neutral", "texture_density": "sparse"},
        },
    ),
    "melodic_instrument": Substrate(
        name="melodic_instrument",
        short="melodic",
        body="A sparse solo instrument line, gentle and unhurried",
        instrumentation=("solo_instrument",),
        style_tags=("solo instrument", "sparse melody"),
        tempo="around a 50 bpm feel",
        energy="low",
        timbre="warm-dark",
        harmony="simple, consonant",
        texture="sparse",
        register="mid",
        event_driven=True,
        requested={
            "tempo_bpm": 50,
            "energy": "low",
            "spectral_centroid": "warm_dark",
            "mode": "simple_consonant",
            "register": "mid",
            "texture_density": "sparse",
            "nature_bed": "none",
        },
        focus_overrides={
            "tempo": "a steady 80-100 bpm feel, held constant throughout",
            "energy": "low-moderate",
            "timbre": "warm-neutral",
            "harmony": "simple, consonant and warm",
            "texture": "steady, unchanging",
            "requested": {"tempo_bpm": 90, "energy": "moderate", "spectral_centroid": "warm_neutral"},
        },
    ),
    "field_recording": Substrate(
        name="field_recording",
        short="field",
        body="A soft natural field-recording bed, broadband and pitchless",
        instrumentation=("field_recording",),
        style_tags=("field recording", "nature ambience", "wide stereo"),
        tempo="arrhythmic",
        energy="very low",
        timbre="soft broadband",
        harmony="no tonal center",
        texture="continuous wash",
        register="full but gentle",
        requested={
            "tempo_bpm": None,
            "energy": "very_low",
            "spectral_centroid": "soft_broadband",
            "mode": "atonal",
            "register": "full",
            "texture_density": "continuous",
            "nature_bed": "present",
        },
        # A masking bed for focus rather than a lead sound (§4 of the product doc:
        # broadband/nature beds help focus via masking, not by being "more energetic"),
        # so only energy nudges up slightly; it stays arrhythmic and even.
        focus_overrides={
            "energy": "low",
            "requested": {"energy": "low"},
        },
    ),
    "noise_texture": Substrate(
        name="noise_texture",
        short="noise",
        # Framed as deep brown noise / distant surf rather than plain "noise": music
        # models render bare noise as harsh hiss, but a soft, low-passed surf-like
        # wash is the same MER footprint and far more pleasant to sit with.
        body=(
            "A soft, deep brown-noise wash, warm and enveloping, like distant ocean "
            "surf or steady rainfall heard from far away, gently low-pass filtered"
        ),
        instrumentation=("brown_noise", "distant_surf"),
        style_tags=("brown noise", "warm wash", "distant surf", "soft rainfall"),
        tempo="arrhythmic",
        energy="very low",
        timbre="deep, warm, heavily low-pass filtered, no hiss",
        harmony="no tonal center",
        texture="smooth and even",
        register="low, full-bodied",
        requested={
            "tempo_bpm": None,
            "energy": "very_low",
            "spectral_centroid": "deep_warm",
            "mode": "none",
            "register": "low_full",
            "texture_density": "even",
            "nature_bed": "none",
        },
        # Best-evidenced focus masking substrate (broadband noise masks distraction), but
        # the relax framing ("deep brown noise... like distant surf") is bass-heavy on
        # purpose to feel enveloping at bedtime; focus wants a more neutral, less
        # sleep-coded wash, still smooth and steady, never harsh hiss.
        focus_overrides={
            "energy": "low-moderate",
            "timbre": "neutral, gently filtered, no harsh hiss",
            "register": "full-bodied, neutral",
            "requested": {"energy": "low", "spectral_centroid": "neutral_broadband"},
        },
    ),
    "percussive_with_tail": Substrate(
        name="percussive_with_tail",
        short="bowls",
        body="Resonant struck tones with long inharmonic decays and silence between strikes",
        instrumentation=("singing_bowls", "bells"),
        style_tags=("singing bowls", "long resonant tails"),
        tempo="arrhythmic, sparse strikes",
        energy="very low",
        timbre="warm-metallic",
        harmony="inharmonic partials",
        texture="sparse",
        register="low-mid",
        event_driven=True,
        requested={
            "tempo_bpm": None,
            "energy": "very_low",
            "spectral_centroid": "warm_metallic",
            "mode": "inharmonic",
            "register": "low_mid",
            "texture_density": "very_sparse",
            "nature_bed": "none",
        },
        # Relax-only: long resonant decays between strikes are a meditative gesture, not
        # something you want decaying across a task-focused attention span.
        goals=frozenset({"relax"}),
    ),
}


# --- Axis B: cultural style (what tradition it evokes) ----------------------


@dataclass(frozen=True)
class StyleSubstrate:
    """A style's coherent realization for one substrate (§4 worked examples).

    Only authored where a specific instrumentation matters (e.g. Buddhist x
    melodic_instrument = shakuhachi). Pairs without an override fall back to the
    substrate's generic body/instrumentation coloured by the style's tags.
    """

    body: str
    instrumentation: tuple[str, ...]
    extra_style_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Style:
    """One cultural style (§2, Axis B), realized *through* a substrate."""

    name: str
    descriptor: str  # fills the "[style-descriptor]" slot of the base template
    global_styles: tuple[str, ...]
    character: str = ""  # recording space, gear and grain — the §4 "production" slot
    overrides: dict[str, StyleSubstrate] = field(default_factory=dict)
    mode_override: str | None = None  # e.g. buddhist -> just_intonation
    nature_bed_override: str | None = None  # e.g. nature_ambient -> present
    goals: frozenset[str] = field(default_factory=lambda: frozenset({"relax", "focus"}))


STYLES: dict[str, Style] = {
    "neutral": Style(
        name="neutral",
        descriptor="ambient",
        global_styles=("instrumental", "ambient", "calm", "minimal"),
        character=(
            "Recorded close and clean with soft analogue warmth, a wide still stereo "
            "image and a gentle hall reverb behind it"
        ),
    ),
    # One coherent tradition per track, wordless by default (§6 guardrails):
    # the drone override uses a *wordless* hum, never a mantra with real text.
    "buddhist_meditative": Style(
        name="buddhist_meditative",
        descriptor="meditative",
        global_styles=("meditative", "just intonation", "instrumental", "very sparse", "low register"),
        character=(
            "Recorded in a large stone hall, long natural reverb tails, faint room tone "
            "and air audible in the silences between sounds"
        ),
        mode_override="just_intonation",
        overrides={
            "percussive_with_tail": StyleSubstrate(
                body=(
                    "Tibetan singing bowls and a distant temple bell, long resonant decays, "
                    "inharmonic metallic partials, over a soft continuous drone"
                ),
                instrumentation=("tibetan_singing_bowls", "distant_temple_bell"),
                extra_style_tags=("tibetan singing bowls",),
            ),
            "drone": StyleSubstrate(
                body=(
                    "Sustained singing-bowl resonance and a low wordless vocal hum, "
                    "just-intonation tuning"
                ),
                instrumentation=("singing_bowl_drone", "wordless_vocal_hum"),
            ),
            "melodic_instrument": StyleSubstrate(
                body=(
                    "Solo shakuhachi bamboo flute, breathy long tones, minor-pentatonic, "
                    "with a faint singing-bowl drone underneath"
                ),
                instrumentation=("shakuhachi", "singing_bowl_drone"),
                extra_style_tags=("shakuhachi",),
            ),
            "field_recording": StyleSubstrate(
                body=(
                    "Distant temple bells with long tails over a soft mountain stream, "
                    "an occasional wind chime"
                ),
                instrumentation=("temple_bells", "mountain_stream", "wind_chime"),
            ),
        },
        # Relax-only: both the sound and the branding read as meditation, not productivity
        # (§6 of the product doc already flags mixing traditions as a guardrail concern;
        # stretching the same style toward "focus" branding is a similar overreach).
        goals=frozenset({"relax"}),
    ),
    "neoclassical": Style(
        name="neoclassical",
        descriptor="neoclassical",
        global_styles=("neoclassical", "felt piano", "strings", "warm", "intimate"),
        character=(
            "Close-mic'd in a small wooden room, felt and key mechanics faintly audible, "
            "warm tape saturation and a soft intimate sustain"
        ),
        overrides={
            "melodic_instrument": StyleSubstrate(
                body="Solo felt piano, soft close-mic, sparse tonal phrases with gentle sustain",
                instrumentation=("felt_piano",),
            ),
            "drone": StyleSubstrate(
                body="A warm sustained string pad, soft and consonant",
                instrumentation=("string_pad",),
            ),
        },
    ),
    # This style read as unpleasant noise for a reason worth writing down: its defining
    # character *was* the noise ("a dusty vinyl noise floor", "tape hiss"), while the density
    # and harmony clauses removed the music that floor is supposed to sit under — leaving
    # filtered hiss and nothing else. Worse, it asked for hiss while NEGATIVE_PROMPT bans
    # "hiss, static, white noise", so the style fought its own negative prompt. The medium
    # (tape warmth, wow and flutter) is what makes lo-fi pleasant; the noise floor is not,
    # and generative models render "noise floor" literally rather than as a subtle artefact.
    # So: keep the tape, drop the hiss, and give every substrate real chords to play.
    "lofi": Style(
        name="lofi",
        descriptor="lo-fi",
        global_styles=("lofi", "tape texture", "soft keys", "warm", "mellow"),
        character=(
            "Warm analogue tape saturation with gentle wow and flutter, soft rounded highs "
            "and a mellow full low end, like a well-loved record played late at night"
        ),
        overrides={
            "melodic_instrument": StyleSubstrate(
                body=(
                    "Soft electric piano playing warm mellow chords, rounded and unhurried, "
                    "with a little tape wobble"
                ),
                instrumentation=("soft_keys", "tape_texture"),
            ),
            # Without this, lofi x drone fell back to the generic body and the style
            # contributed nothing but grain — a pad made of noise.
            "drone": StyleSubstrate(
                body=(
                    "Sustained tape-saturated chords on soft electric piano and warm pad, "
                    "blurred at the edges, no rhythm"
                ),
                instrumentation=("soft_keys", "sustained_pad", "tape_texture"),
            ),
        },
    ),
    "nature_ambient": Style(
        name="nature_ambient",
        descriptor="nature-ambient",
        global_styles=("nature ambient", "field recording", "wide stereo", "water", "wind"),
        character=(
            "Wide open natural stereo, recorded outdoors at dawn with real depth and "
            "distance, the space itself audible around the sound"
        ),
        nature_bed_override="present",
        overrides={
            "field_recording": StyleSubstrate(
                body="A gentle mountain stream and soft wind through trees, wide natural stereo",
                instrumentation=("mountain_stream", "wind"),
            ),
            "drone": StyleSubstrate(
                body="A soft warm pad under a bed of gentle rain and distant wind",
                instrumentation=("warm_pad", "rain", "distant_wind"),
            ),
        },
    ),
}


# --- Signature: a resolved (substrate, style) render spec -------------------


@dataclass(frozen=True)
class Signature:
    """A fully resolved render spec for one (substrate, style, goal, duration) cell."""

    substrate: Substrate
    style: Style
    goal: "Goal"
    duration_s: int
    seed: int
    prompt: str
    negative_prompt: str
    instrumentation: tuple[str, ...]
    composition_plan: dict[str, Any]
    requested_features: dict[str, Any]
    development: str

    @property
    def track_id(self) -> str:
        base = f"{self.style.name}_{self.substrate.short}_{self.goal.name}_seed{self.seed}"
        # The goal's default development keeps the pre-axis track_id/seed identity exactly
        # (every already-planned/rendered track stays reproducible); only an explicit,
        # non-default request earns a suffix, so it can never collide with the default cell.
        if self.development == self.goal.default_development:
            return base
        return f"{base}_dev{self.development}"

    def spec(self) -> dict[str, Any]:
        """The render-independent request record (§3), before any audio exists.

        Everything here is provider-agnostic on purpose: spec management is free
        and offline, so the same spec can be rendered by ElevenLabs now or a local
        model later. The renderer fills the ``render`` block (provider, model,
        output format, provenance, audio file). ``measured_features`` stays
        ``None`` until the objective-feature extraction pass runs over the audio —
        the bandit needs *measured* MER, and models don't reliably hit requested
        coordinates.
        """
        return {
            "track_id": self.track_id,
            "kind": "grid",
            "substrate": self.substrate.name,
            "style": self.style.name,
            "goal": self.goal.name,
            "group": None,
            "keyword": None,
            "requested_features": self.requested_features,
            "measured_features": None,
            "instrumentation": list(self.instrumentation),
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "composition_plan": self.composition_plan,
            "seed": self.seed,
            "duration_s": self.duration_s,
            "loopable": True,
            "render": None,
        }


def _seed(key_parts: tuple[str, str], variant: int = 0) -> int:
    """A deterministic per-signature seed so re-renders stay reproducible (§1).

    ``key_parts`` is whatever identifies the cell — ``(style, substrate)`` for a
    grid signature, ``(group, keyword)`` for a special one. ``variant`` gives a
    cell more than one render (the doc wants 3-5 seeds per cell, §5); ``variant
    == 0`` keeps the original one-seed-per-cell identity so existing track_ids
    stay stable.
    """
    key = f"{key_parts[0]}:{key_parts[1]}"
    if variant:
        key = f"{key}:{variant}"
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:4], "big") % 100_000


def retry_seed(track_id: str, attempt: int) -> int:
    """A fresh seed for re-rendering a track whose first render failed QC.

    Retrying with the spec's own seed would mostly reproduce the same broken audio —
    a dead or silent render is a property of that point in the noise space, not of
    luck. Derived from the track_id so the Nth retry is still reproducible, and kept
    out of the cell-variant space (:func:`_seed` over ``(style, substrate)``) so it
    can't collide with another track's planned seed.
    """
    return _seed((track_id, "qc-retry"), attempt)


def _dedupe(tags: tuple[str, ...]) -> list[str]:
    """Order-preserving dedupe for style-tag lists."""
    return list(dict.fromkeys(tags))


# Every track is a bed you sit with for tens of minutes, and the first library came back
# *correct but plain*: the axes were on target and nothing moved. Motion at the timescale of
# a breath is what makes a drone worth staying with, and it costs nothing on the MER axes
# because the mean level, brightness and density don't move. This used to be a single fixed
# clause per goal; it's now `development`, a real axis (test feedback wanted tracks that "go
# somewhere") gated per goal below via `Goal.allowed_development` — sleep-onset wants
# familiarity/low prediction-error, so the strictest cases stay `static`, while relax and
# focus both default to `slow_swell`.
# These were amplitude instructions ("swelling and receding") and nothing else, which is
# why un-banning melody in Change 1 produced no melody: nothing in the prompt ever *asked*
# for notes. Un-banning is not requesting. They now name the harmonic content explicitly —
# chords, resolution, a phrase — because that is the thing the listener notices as
# "progression", and they say "pleasant"/"consonant" out loud, since a model given only
# "gentle harmonic movement" is as likely to drift somewhere sour as somewhere warm.
DEVELOPMENT_FRAGMENT: dict[str, str] = {
    "static": "It holds to one harmony throughout, unchanging, with no melodic movement",
    "slow_swell": (
        "It moves through a slow, pleasant chord progression: four or five different warm "
        "consonant chords, each held about fifteen seconds and then changing clearly and "
        "audibly to the next, so you can always hear the harmony going somewhere. The bass "
        "note moves with it. The whole texture swells and recedes over minutes"
    ),
    "motif_evolving": (
        "A clear, pleasant melody plays over a slow chord progression underneath: a short "
        "singable phrase of a few notes, played, then answered, then varied and returned "
        "to, with the chords beneath it changing audibly every few bars and the bass note "
        "moving with them, always resolving warmly. The tune is the point — it should be "
        "easy to follow and easy to hum, never a single held note or a static wash"
    ),
}

# Change 1 removed "melodic hook, catchy melody" here but left "key change, chord
# progression" — which bans, in as many words, the slow tone progression `development` is
# for. Dropping both: what actually hurts sustained attention is *drama* (climax, buildup,
# drop, sudden transitions), all of which stay banned. A warm chord progression is not drama.
FOCUS_NEGATIVE_PROMPT = (
    "lyrics, vocals, spoken word, dramatic climax, sudden transitions, EDM, buildup, drop, "
    "harsh, distorted, chaotic"
)

FOCUS_NEGATIVE_GLOBAL_STYLES: tuple[str, ...] = (
    "vocals",
    "lyrics",
    "buildup",
    "drop",
    "chaotic",
    "harsh",
)


@dataclass(frozen=True)
class Goal:
    """One arousal target — a third axis, orthogonal to substrate and style.

    Substrate says *what* the sound physically is; style says *what tradition* it evokes;
    goal says *how aroused* it should feel, and owns every phrase in the base template that
    used to be hardcoded relax language (the opening intent, the dynamics/brightness
    descriptors, the closing sentence, and the negative prompt) plus which points on the
    `development` axis (:data:`DEVELOPMENT_FRAGMENT`) it admits at all.
    """

    name: str
    intent: str  # "...soundscape for {intent}." — the template's opening
    dynamics: str  # e.g. "very soft dynamics" vs "smooth, controlled dynamics"
    brightness: str  # e.g. "warm and dark, low spectral brightness"
    closing: str
    flowing: str  # the density bound once `development` is melodic (§ :func:`density_clause`)
    negative_prompt: str
    negative_global_styles: tuple[str, ...]
    allowed_development: frozenset[str]
    default_development: str


GOALS: dict[str, Goal] = {
    "relax": Goal(
        name="relax",
        intent="deep relaxation",
        # Both of these used to sit at the far end of their axis ("very soft", "warm and
        # dark, low spectral brightness"), which stacks with very-low energy and a sparse
        # texture into something muffled rather than restful. Still unambiguously
        # down-regulating, just no longer authored at the extreme.
        dynamics="soft, even dynamics",
        brightness="warm, with soft natural clarity in the upper mids, never bright or harsh",
        closing="No vocals, no percussion hits, no sudden transitions. Seamless, calm, continuous.",
        flowing=RELAX_FLOWING,
        negative_prompt=NEGATIVE_PROMPT,
        negative_global_styles=NEGATIVE_GLOBAL_STYLES,
        # static is the strictest, sleep-safe end; motif_evolving is excluded here — real
        # progression is a focus/wind-down affordance, not a down-regulation one (see
        # docs/BGMUSIC_TAXONOMY_CHANGES.md Change 2).
        allowed_development=frozenset({"static", "slow_swell"}),
        default_development="slow_swell",
    ),
    "focus": Goal(
        name="focus",
        intent="sustained focus",
        dynamics="smooth, controlled dynamics, no sudden loud or soft jumps",
        brightness="clear and present, natural spectral brightness, never harsh",
        closing="No vocals, no lyrics, no sudden transitions. Seamless and steady, never dramatic.",
        flowing=FOCUS_FLOWING,
        negative_prompt=FOCUS_NEGATIVE_PROMPT,
        negative_global_styles=FOCUS_NEGATIVE_GLOBAL_STYLES,
        # static is excluded: predictability, not stillness, is the goal — focus tolerates,
        # even wants, a beat, it just can't have drama (docs/background_music.md §7.2).
        # The default is the *melodic* end, where relax's is slow_swell: sharing a default
        # made the two goals' most salient sentence identical, and up-regulation is the
        # side that should carry an actual tune.
        allowed_development=frozenset({"slow_swell", "motif_evolving"}),
        default_development="motif_evolving",
    ),
}


def mode_filter_summary(goal_name: str) -> dict[str, Any]:
    """The effective per-goal filter — which cells and ``development`` values a goal admits.

    docs/BGMUSIC_TAXONOMY_CHANGES.md (Change 3) asks for one ``MODE_FILTER`` object as "the
    single place" this lives. Here that data is already distributed and authoritative on the
    real per-axis objects — ``Substrate.goals``/``Style.goals`` (checked in
    :func:`build_signature`) and ``Goal.allowed_development``/``default_development`` — so this
    is a read-only assembler, not a second copy: it can't drift out of sync with the checks
    that actually gate a render.
    """
    if goal_name not in GOALS:
        raise ValueError(f"unknown goal {goal_name!r}, expected one of {list(GOALS)}")
    goal = GOALS[goal_name]
    return {
        "allow_substrate": {name for name, sub in SUBSTRATES.items() if goal_name in sub.goals},
        "allow_style": {name for name, sty in STYLES.items() if goal_name in sty.goals},
        "allow_development": set(goal.allowed_development),
        "default_development": goal.default_development,
    }


def _resolve_substrate_for_goal(substrate: Substrate, goal_name: str) -> Substrate:
    """The substrate's MER fields as they apply under ``goal_name``.

    ``relax`` is the substrate's own definition; ``focus`` shallow-merges
    ``substrate.focus_overrides`` on top (``requested`` merges into the existing dict rather
    than replacing it). Identity fields (``body``, ``instrumentation``, ``short``, ...) are
    never in ``focus_overrides`` and so never move — only arousal changes with goal.
    """
    if goal_name == "relax" or not substrate.focus_overrides:
        return substrate
    overrides = dict(substrate.focus_overrides)
    requested_override = overrides.pop("requested", None)
    if requested_override:
        overrides["requested"] = {**substrate.requested, **requested_override}
    return replace(substrate, **overrides)


def build_prompt(
    substrate: Substrate, style: Style, body: str, nature_bed: str, goal: Goal, development: str
) -> str:
    """Fill the §4 base template: substrate body, style descriptor and character, the
    MER axes, the `development`-selected motion clause, and the goal-conditioned
    dynamics/brightness/closing clauses.

    A melodic ``development`` also relaxes the density bound (:func:`density_clause`) and
    the prose texture word (:data:`TEXTURE_WHEN_MELODIC`), so the prompt doesn't ask for a
    tune and forbid the room to play it in the same breath.
    """
    nature = "" if nature_bed == "none" else "A soft natural bed blended gently underneath. "
    character = f"{style.character}. " if style.character else ""
    texture = substrate.texture
    if development in MELODIC_DEVELOPMENT:
        texture = TEXTURE_WHEN_MELODIC.get(texture, texture)
    return (
        f"Instrumental {style.descriptor} soundscape for {goal.intent}. "
        f"{body}. "
        f"{DEVELOPMENT_FRAGMENT[development]}. "
        f"{density_clause(substrate.event_driven, development, goal.flowing)}. "
        f"{character}"
        f"Tempo {substrate.tempo}, {substrate.energy} energy, {goal.dynamics}. "
        f"{substrate.timbre} timbre, {goal.brightness}. "
        f"{substrate.harmony} harmony, {texture} texture, {substrate.register} register. "
        f"{nature}"
        f"{goal.closing}"
    )


def build_signature(
    substrate_name: str,
    style_name: str,
    goal_name: str,
    duration_s: int,
    variant: int = 0,
    development: str | None = None,
) -> Signature:
    """Resolve a (substrate, style, goal) triple into a full render spec.

    A substrate/style pair is valid (§2) only for the goals both support — some cells are
    goal-restricted (e.g. ``percussive_with_tail`` and ``buddhist_meditative`` are relax-only;
    see their definitions for why) — where the style has a coherent realization for the
    substrate we use it, otherwise the substrate's generic body is coloured by the style's
    global tags. ``variant`` selects a distinct seed within the cell.

    ``development`` (:data:`DEVELOPMENT_FRAGMENT`) is the progression axis; omitted, it
    resolves to the goal's ``default_development`` (keeping every pre-axis track_id/seed
    unchanged, see :attr:`Signature.track_id`). An explicit value must be one the goal's
    ``allowed_development`` admits — a mode-gate, same shape as the substrate/style ``goals``
    checks below (docs/BGMUSIC_TAXONOMY_CHANGES.md Change 3).
    """
    if substrate_name not in SUBSTRATES:
        raise ValueError(f"unknown substrate {substrate_name!r}, expected one of {list(SUBSTRATES)}")
    if style_name not in STYLES:
        raise ValueError(f"unknown style {style_name!r}, expected one of {list(STYLES)}")
    if goal_name not in GOALS:
        raise ValueError(f"unknown goal {goal_name!r}, expected one of {list(GOALS)}")

    substrate = SUBSTRATES[substrate_name]
    style = STYLES[style_name]
    if goal_name not in substrate.goals:
        raise ValueError(f"substrate {substrate_name!r} does not support goal {goal_name!r}")
    if goal_name not in style.goals:
        raise ValueError(f"style {style_name!r} does not support goal {goal_name!r}")

    goal = GOALS[goal_name]
    if development is None:
        development = goal.default_development
    elif development not in DEVELOPMENT_FRAGMENT:
        raise ValueError(f"unknown development {development!r}, expected one of {list(DEVELOPMENT_FRAGMENT)}")
    elif development not in goal.allowed_development:
        raise ValueError(f"goal {goal_name!r} does not allow development {development!r}")

    substrate = _resolve_substrate_for_goal(substrate, goal_name)
    override = style.overrides.get(substrate_name)

    body = override.body if override else substrate.body
    instrumentation = override.instrumentation if override else substrate.instrumentation
    extra_tags = override.extra_style_tags if override else ()

    # Requested MER coordinate: substrate default (goal-resolved above), with the style
    # allowed to move only mode/nature_bed (§2: style changes instrumentation/mode, not
    # the axis).
    requested = dict(substrate.requested)
    if style.mode_override is not None:
        requested["mode"] = style.mode_override
    if style.nature_bed_override is not None:
        requested["nature_bed"] = style.nature_bed_override
    requested["development"] = development

    positive_global_styles = _dedupe(substrate.style_tags + style.global_styles + extra_tags)

    # One long section == the loop length; empty lines force an instrumental,
    # wordless render (guardrail §6.2). Local styles carry the instrumentation.
    composition_plan: dict[str, Any] = {
        "positive_global_styles": positive_global_styles,
        "negative_global_styles": list(goal.negative_global_styles),
        "sections": [
            {
                "section_name": "loop",
                "positive_local_styles": list(instrumentation),
                "negative_local_styles": list(goal.negative_global_styles),
                "duration_ms": duration_s * 1000,
                "lines": [],
            }
        ],
    }

    # The seed key only grows a development suffix off the default (see track_id): the
    # common, no-argument call reproduces every seed already on disk.
    key_style_goal = f"{style_name}:{goal_name}"
    if development != goal.default_development:
        key_style_goal = f"{key_style_goal}:{development}"

    return Signature(
        substrate=substrate,
        style=style,
        goal=goal,
        duration_s=duration_s,
        seed=_seed((key_style_goal, substrate_name), variant),
        prompt=build_prompt(substrate, style, body, requested["nature_bed"], goal, development),
        negative_prompt=goal.negative_prompt,
        instrumentation=instrumentation,
        composition_plan=composition_plan,
        requested_features=requested,
        development=development,
    )


def composition_plan_for_model(spec: dict[str, Any], model_id: str) -> dict[str, Any]:
    """Translate a spec's stored plan into the shape the target model expects.

    The spec stores the ``music_v1`` plan (a ``MusicPrompt``: global styles + one
    loop ``section``). ``music_v2`` (and newer) instead take a ``CompositionPlan``
    of ``chunks`` where ``text`` is section/lyrics and ``positive_styles`` carries
    the musical direction. We derive the v2 chunk from the same stored fields, so
    existing specs render on either model without re-planning.

    Instrumental is enforced the same way in both: no lyric lines (the v2 ``text``
    is a bare section tag) plus ``vocals`` in the negative styles.
    """
    plan = spec["composition_plan"]
    if model_id == "music_v1":
        return plan

    section = plan["sections"][0]
    styles = _dedupe(
        tuple(plan["positive_global_styles"])
        + tuple(section["positive_local_styles"])
        + ("instrumental", "high production quality")
    )
    return {
        "chunks": [
            {
                "text": "[Ambient instrumental soundscape]",
                "duration_ms": int(spec["duration_s"] * 1000),
                "positive_styles": styles,
                "negative_styles": list(plan["negative_global_styles"]),
            }
        ]
    }


# --- Special cells: keyword-driven categories outside the grid --------------
#
# Not every useful bed fits a (substrate, style) pair — a plain "rain" or "ocean"
# ambience isn't really a *style* of anything. Special groups are a second,
# independent taxonomy: one keyword, one prompt clause, no substrate/style axis.
# They still produce the same spec shape (§3) so storage, catalog and rendering
# don't need to special-case them; only ``kind`` (and ``group``/``keyword`` in
# place of ``style``/``substrate``) tells them apart.


@dataclass(frozen=True)
class KeywordEntry:
    """One keyword within a special group.

    ``description`` *opens* the prompt and carries the whole subject: the SA3 prompt
    guide's SFX section asks for the core source, the action, and the production
    (mic placement, room), so these read as a recordist's slate rather than a mood.

    ``event_driven`` says whether the sound is made of separate events (a bird call, a
    chime strike) or is one continuous bed (rain, wind) — which decides the density
    limit it gets, since telling a bed to leave gaps makes it *more* eventful, not less
    (:func:`density_clause`).

    ``goals`` is metadata only, unlike the grid's ``goals`` allow-lists: a field recording
    doesn't need a different *render* per goal (the same rainfall suits both relaxing and
    masking-for-focus), so this doesn't gate :func:`build_keyword_signature` or change
    ``track_id`` — it just documents which goals a keyword is expected to suit, for a future
    caller (e.g. background selection) that wants to filter by it.

    ``flowing`` and ``body`` override the group's corresponding clause for this one
    keyword. Both exist because a group is not always uniform: ``natural_sounds`` is a
    still, no-music shell that suits rain and fire, and ``universe`` sits inside it while
    needing the opposite — movement, and tonal content the shell's "no music, no
    instruments" tail would otherwise argue away. Overriding one keyword beats either
    loosening the shell for all of them or inventing a group for one member.
    """

    description: str
    event_driven: bool = False
    goals: frozenset[str] = field(default_factory=lambda: frozenset({"relax", "focus"}))
    flowing: str | None = None
    body: str | None = None


@dataclass(frozen=True)
class SpecialGroup:
    """A keyword-driven category, independent of the substrate x style grid.

    The three defaulted fields exist because a special group is *not* necessarily a
    down-regulation one. ``natural_sounds`` shares the library's relax negatives, but a
    group like ``energizer`` is up-regulating: the shared :data:`NEGATIVE_PROMPT` bans
    "energetic, fast, drums, EDM, bright", which would forbid the group's whole point. So
    the negatives are per-group, defaulting to the relax pair that ``natural_sounds`` was
    already using.

    ``flowing`` overrides the density bound the same way :attr:`Goal.flowing` does for the
    grid: left ``None`` a keyword gets the stillness bound suited to how it is produced
    (:func:`density_clause`), which is right for a field recording and wrong for anything
    that has to sustain a continuous musical body.
    """

    name: str
    body: str  # the shared prompt shell every keyword in the group layers onto
    keywords: dict[str, KeywordEntry]
    global_styles: tuple[str, ...] = ()  # composition-plan styles common to the group
    negative_prompt: str = NEGATIVE_PROMPT
    negative_global_styles: tuple[str, ...] = NEGATIVE_GLOBAL_STYLES
    flowing: str | None = None
    development: str | None = None  # a key of DEVELOPMENT_FRAGMENT; None = no progression clause


SPECIAL_GROUPS: dict[str, SpecialGroup] = {
    "natural_sounds": SpecialGroup(
        name="natural_sounds",
        # The keyword's own description comes *first* and this is the tail, because a
        # field recording is a sound effect, not music: the earlier phrasing opened
        # with "Instrumental natural soundscape ... very low energy, warm-dark timbre"
        # and buried the actual subject after 50 words of MER vocabulary, which read to
        # the text encoder as an instruction to make an ambient music bed. It duly did
        # — the first rain renders measured a 150 Hz spectral centroid (a bass drone;
        # real rainfall lands nearer 5 kHz). Everything the grid says about tempo,
        # harmony and register is meaningless here and is gone.
        # "Calm and unhurried" alone left the renders too busy: it constrains *character*
        # but says nothing about *rate*, so a bed could be calm in timbre and still patter,
        # chirp or lap away continuously. The pacing is now stated as a rate, and stated
        # for both kinds of keyword at once — event-driven ones get "infrequent and widely
        # spaced", continuous ones "the overall rate never picks up" (a slow drizzle rather
        # than heavy rainfall), since the density clause only reaches the former.
        body=(
            "Slow-paced throughout: the overall rate stays low and never picks up, and "
            "anything that stands out from the bed is infrequent and widely spaced. Calm "
            "and unhurried, with no sudden events, no busy passages and no build. Natural "
            "stereo field recording, unprocessed, no music, no instruments, no voices."
        ),
        global_styles=("field recording", "nature ambience", "instrumental", "wide stereo"),
        keywords={
            "rain": KeywordEntry(
                "Steady soft rainfall on leaves and wet ground, fine even patter, "
                "no thunder and no wind gusts, recorded from under a porch a few feet away"
            ),
            "ocean": KeywordEntry(
                "Slow ocean waves washing onto a sandy shore, long gentle swells and "
                "receding foam, distant calm surf recorded from up the beach"
            ),
            "wind": KeywordEntry(
                "Soft wind moving through pine trees, airy and slow, needles and leaves "
                "rustling, no howling, recorded outdoors at a distance"
            ),
            "stream": KeywordEntry(
                "A small stream trickling over rocks, steady flowing water with soft "
                "gurgles, recorded close to the bank"
            ),
            # The event-driven keywords are where an engine reaches for a dawn chorus,
            # so each one names how *few* events it wants, not just which ones.
            "forest": KeywordEntry(
                "A still forest clearing at dawn, mostly quiet air and the faint rustle "
                "of leaves, with a single distant bird calling once in a while and long "
                "silences in between, no dawn chorus and no flocks",
                event_driven=True,
            ),
            # Kept close to the original wording: every rewrite measured busier, and a
            # cricket bed is a continuous wash, so it takes the steadiness clause.
            "night": KeywordEntry(
                "A quiet summer night in open country, one even layer of faint distant "
                "crickets and still air, nothing close and nothing in the foreground"
            ),
            "chimes": KeywordEntry(
                "Wind chimes turning in a light breeze on a porch, two or three soft "
                "metallic notes at a time with long shimmering decays, then quiet garden "
                "air until the next gust",
                event_driven=True,
            ),
            # Space is silent, so there is no field recording to imitate — but there is a
            # well-known referent that *is* one: the plasma-wave recordings spacecraft
            # return, which is a far more concrete instruction than "space ambience" and
            # keeps the group's recordist framing intact. Continuous, so it takes the
            # steadiness bound rather than being broken into discrete events.
            "universe": KeywordEntry(
                "Deep space ambience, like a spacecraft's plasma-wave recording — a vast "
                "low resonant hum with slow sweeping tones rising and falling through it, "
                "distant shimmer passing by, and deep sub-bass swells arriving and receding",
                # The one keyword in this group that has to *move*. The group's stillness
                # bound ("nothing stepping forward out of it, no separate events and no
                # layering") and its "changing almost imperceptibly" original were the
                # whole reason it came back as an undifferentiated hum.
                flowing=(
                    "It travels: always moving somewhere across minutes — tones sweeping "
                    "slowly up and down in pitch, layers drifting in and then away, one "
                    "distinct passage giving way to the next so it is never the same for "
                    "long. Vast and unhurried, but never still and never uniform"
                ),
                # The shared shell's "no music, no instruments" tail would argue away the
                # tonal content that makes those sweeps audible. This keeps it a *sound*
                # rather than music by banning the musical machinery instead — beat, drums,
                # chord progression — which leaves pitch and timbre free to move.
                body=(
                    "Immense, dark and open, with a real sense of enormous empty space and "
                    "depth. Wide stereo, deep sub-bass, weightless and unhurried. A sound, "
                    "not music: no beat, no drums, no chord progression, no voices."
                ),
            ),
            # A fire is both things at once: a continuous soft roar of flame *and* separate
            # crackles. Marked continuous on purpose — the event-driven clause would demand
            # "long stretches of near-stillness", which breaks the flame bed into detached
            # pops (the same failure the cricket wash has). The crackle spacing is stated in
            # the description instead, which is how "night" handles it.
            "fireplace": KeywordEntry(
                "A log fire burning low in a stone fireplace, a steady soft roar of flame "
                "close by, with only an occasional gentle crackle or pop well spaced apart, "
                "recorded a few feet from the hearth in a quiet room"
            ),
        },
    ),
    # docs/BGMUSIC_TAXONOMY_CHANGES.md Change 4 asks for the focus/Brain.fm-competing beds
    # to live in their own curated pack rather than folded into the down-regulation grid,
    # defined by purpose and by a hard technical gate: every member must be AM-compatible.
    # This is the prompt half of that — it asks for the continuous musical body an AM
    # carrier needs. The *measured* half (`am_depth_by_band[active_band] < THRESHOLD`, via
    # scripts/audio_features.py) is still unimplemented, so membership here is asserted,
    # not yet verified; see the note in that doc.
    #
    # Deliberately focus-only, uniformly (test_energizer_keywords_are_focus_only,
    # test_energizer_prompts_meet_every_stated_requirement — every member shares this
    # group's body/flowing verbatim, no per-keyword mood). The AM-compatibility mechanism
    # (a continuous, gap-free bed) isn't inherently focus-specific, but this pack's taste —
    # "awake and warm", a steady beat — is; relax-goal AM content with a different taste
    # lives in the parallel `unwind` group below instead of diluting this one.
    "energizer": SpecialGroup(
        name="energizer",
        # Continuity is not restated here: `flowing` (FOCUS_FLOWING) already carries it,
        # and saying it three times in one prompt only dilutes everything else. This
        # carries what that clause does not — energy level, volume ceiling, and taste.
        body=(
            "Moderate, steady energy — awake and warm rather than sleepy — but never loud, "
            "never intense and never dramatic. Easy and familiar to an ordinary listener, "
            "the kind of thing that is pleasant to have playing for an hour without ever "
            "asking for attention."
        ),
        global_styles=(
            "instrumental",
            "melodic",
            "warm",
            "steady groove",
            "background music",
            "high production quality",
        ),
        # The relax negatives would ban this group's entire purpose ("energetic", "fast",
        # "drums", "EDM", "bright"). What actually has to stay out is *drama* and *volume* —
        # plus silence and gaps, which are a technical fault here rather than a taste one:
        # a gap in the bed is a gap in the AM carrier (§ FOCUS_FLOWING).
        negative_prompt=(
            "lyrics, vocals, spoken word, rap, dramatic climax, buildup, drop, sudden "
            "transitions, aggressive, distorted, harsh, loud, frantic, chaotic, sparse, "
            "silence, gaps, fade out, ambient drone"
        ),
        negative_global_styles=(
            "vocals",
            "lyrics",
            "aggressive",
            "distorted",
            "harsh",
            "buildup",
            "drop",
            "silence",
        ),
        # Reused verbatim from the focus goal rather than restated: this group *is* the
        # focus pack, and the requirement is identical — one wording, one thing to keep true.
        flowing=FOCUS_FLOWING,
        # Continuous is not the same as developing: the clauses above ask the bed to keep
        # *playing*, and nothing asked it to go anywhere, which is the same gap that made
        # un-banning melody produce no melody on the grid. This borrows the grid's axis
        # wholesale — motif_evolving is what the focus goal already defaults to.
        development="motif_evolving",
        keywords={
            "uplift": KeywordEntry(
                "Warm, gently uplifting instrumental music in a major key — a simple "
                "hopeful melody on soft synth and electric piano over a light steady pulse, "
                "open and unhurried",
                goals=frozenset({"focus"}),
            ),
            "chillhop": KeywordEntry(
                "Relaxed instrumental chillhop — a mellow melodic hook on electric piano "
                "over a soft unhurried head-nodding groove and a round warm bass, easy and "
                "familiar",
                goals=frozenset({"focus"}),
            ),
            "daydream": KeywordEntry(
                "Airy melodic instrumental — a clear simple synth melody floating over warm "
                "sustained chords and a soft steady pulse, spacious and pleasant",
                goals=frozenset({"focus"}),
            ),
            "momentum": KeywordEntry(
                "Steady, gently propulsive instrumental — a repeating melodic figure over an "
                "even soft beat and a moving bass line, forward-moving but calm and never "
                "urgent",
                goals=frozenset({"focus"}),
            ),
            "warmth": KeywordEntry(
                "Soft neo-soul-flavoured instrumental — warm electric piano chords and a "
                "simple singing melody over a light brushed groove and a mellow bass, "
                "comfortable and unhurried",
                goals=frozenset({"focus"}),
            ),
            # No overrides: this is squarely what the group's own defaults (FOCUS_FLOWING,
            # the "awake and warm" body) were written for.
            # v1 leaned on "crisp", "precise", "clinical" to keep it functional rather than
            # danceable — instead it rendered dry, cold, occasionally unsettling. v2 warmed
            # the language up but still came back dry and thin: naming the arpeggio and the
            # beat but only gesturing at "lush evolving pads" as a modifier left nothing to
            # actually fill out the body. This gives the pad its own sentence as a named,
            # continuously-present layer — which is also the AM half of the fix: FOCUS_FLOWING
            # already asks for a sustained layer under everything so the carrier is never
            # thin/silent for the modulation to ride on, but a layer only mentioned in passing
            # is easy for the model to shortchange, and a thin bed is a weak AM carrier.
            # Calling it out explicitly is what makes both problems the same fix.
            "clarity": KeywordEntry(
                "Warm ambient electronica for sustained focus — a soft, clearly melodic "
                "synth arpeggio over a slow, steady beat, and underneath it a slow-moving "
                "ambient pad layer that continuously swells and shifts, filling out the "
                "sound with real low-end warmth and depth; full-bodied throughout, never "
                "dry, thin, cold, clinical or unsettling",
                goals=frozenset({"focus"}),
            ),
        },
    ),
    # energizer's relax-goal counterpart: same AM-compatibility requirement (a continuous,
    # gap-free bed — see energizer's comment above), different taste. "Moderate, steady
    # energy — awake and warm" is wrong for something meant to destress or meditate to, so
    # this is its own group with its own body/flowing rather than per-keyword overrides
    # inside energizer (which test_energizer_prompts_meet_every_stated_requirement locks to
    # one shared mood on purpose).
    "unwind": SpecialGroup(
        name="unwind",
        body=(
            "Calm, settled and easy rather than sleepy — spacious and unhurried, gently "
            "absorbing without ever asking for attention, the kind of thing that can sit "
            "underneath an hour of rest, meditation or quiet unwinding without ever feeling "
            "urgent."
        ),
        global_styles=(
            "instrumental",
            "melodic",
            "warm",
            "background music",
            "high production quality",
        ),
        # Same AM-continuity concern as energizer (a gap in the bed is a gap in the
        # carrier), so the same negative list — the shared relax negatives ban "energetic,
        # fast" language this group doesn't need, but say nothing about silence/gaps.
        negative_prompt=(
            "lyrics, vocals, spoken word, rap, dramatic climax, buildup, drop, sudden "
            "transitions, aggressive, distorted, harsh, loud, frantic, chaotic, sparse, "
            "silence, gaps, fade out, ambient drone"
        ),
        negative_global_styles=(
            "vocals",
            "lyrics",
            "aggressive",
            "distorted",
            "harsh",
            "buildup",
            "drop",
            "silence",
        ),
        flowing=RELAX_FLOWING,
        development="motif_evolving",
        keywords={
            # v1 opened on a scene ("temple courtyard at dusk") before naming an instrument,
            # and called the bowl a "drone" — it rendered as ambience/noise, no music. Named
            # instruments come first now, two of them carrying the tune together (so there
            # is always a melodic voice playing, not one flute alone against silence), and
            # the bowl is demoted to "very quiet ... underneath" so it reads as backing
            # texture rather than the competing instruction "drone" was. Same instrumentation
            # family the grid already established (shakuhachi, singing bowl) plus a plucked
            # guzheng for a second clear melodic line; wordless throughout (§6 guardrail — no
            # rendered sacred text).
            "temple": KeywordEntry(
                "A Buddhist temple ensemble — a solo shakuhachi playing a clear, breathy "
                "melody, answered and layered by a soft plucked guzheng carrying the tune "
                "together, with only a very quiet singing-bowl resonance underneath, warm "
                "and meditative, wordless throughout",
                goals=frozenset({"relax"}),
            ),
            "lounge": KeywordEntry(
                "A late-night slow jazz quartet — soft brushed-drum swing, a warm upright "
                "bass walking gently underneath, a Rhodes electric piano comping soft "
                "chords, and a muted trumpet playing an unhurried, tender melody, close and "
                "intimate like a dim, half-empty room",
                goals=frozenset({"relax"}),
            ),
            # Solo, not the ensemble lounge already carries — a distinct instrumentation
            # rather than a mood variant of it. v1 asked for "rubato" (no steady tempo) and
            # nothing but a held melody over pedal resonance, which rendered pleasant but
            # inert — no pace, nowhere to go. This gives the left hand a steady, flowing
            # part to carry pace, and states the chords have to move and resolve, so there
            # is somewhere for the piece to go.
            "piano": KeywordEntry(
                "A solo grand piano playing a warm, clearly melodic tune at a gentle, "
                "flowing pace — a singable right-hand line moving over a steady left-hand "
                "arpeggio, with the chords shifting and resolving warmly every few bars, "
                "tender and full of quiet forward motion",
                goals=frozenset({"relax"}),
            ),
        },
    ),
}


# ── Soundscape keywords ──────────────────────────────────────────────────────────
#
# What a client filters the library by. A profile card names a *list* of these
# (``spec.soundscape``), and the pool is every track matching any of them.
#
# The vocabulary is the taxonomy's own axis values, because a track sits on exactly two
# axes and both are worth filtering by: a grid track is a (substrate, style) pair, a
# special one is a (group, keyword) pair. So "lofi" and "melodic_instrument" each name a
# slice of the grid, "unwind" and "temple" each name a slice of a group, and a dotted
# keyword intersects two — "lofi.melodic_instrument" is the one cell where they overlap.
#
# Matching is order-free: a dotted keyword is a *set* of tags a track must carry, not a
# path through a hierarchy. There is no hierarchy to walk — substrate and style are
# orthogonal, neither contains the other — and inventing a canonical order would just be
# one more thing to get wrong in hand-authored JSON, with a silently empty pool as the
# only feedback. :func:`soundscape_problems` rejects the pairs that *can't* co-occur
# (two substrates, a group with another group's keyword), which is the real error.
SOUNDSCAPE_SEPARATOR = "."


def soundscape_tags(entry: dict[str, Any]) -> list[str]:
    """The taxonomy tags a catalog entry is filterable by: its two axis values.

    Grid tracks give ``[substrate, style]``, special ones ``[group, keyword]``. This is
    what the published manifest carries per track (``tracks[].tags``) and what the client
    matches a profile's ``spec.soundscape`` against, so both ends filter on exactly the
    same strings.
    """
    if entry.get("kind") == "special":
        pair = (entry.get("group"), entry.get("keyword"))
    else:
        pair = (entry.get("substrate"), entry.get("style"))
    return [tag for tag in pair if tag]


def soundscape_matches(tags: Iterable[str], keyword: str) -> bool:
    """Whether a track carrying ``tags`` is selected by one soundscape ``keyword``.

    A bare keyword matches when the track carries it; a dotted one when the track carries
    every part, in any order. An empty keyword matches nothing — a blank string in an
    authored list is a mistake, and treating it as "everything" would silently widen the
    pool to the whole library.
    """
    parts = keyword.split(SOUNDSCAPE_SEPARATOR)
    if not keyword or not all(parts):
        return False
    tag_set = set(tags)
    return all(part in tag_set for part in parts)


def soundscape_selects(tags: Iterable[str], soundscape: Sequence[str] | None) -> bool:
    """Whether a track is in the pool a whole ``soundscape`` list describes (any match).

    An empty or absent list is "no filter" — every track qualifies, which is what a card
    that names no soundscape means.
    """
    if not soundscape:
        return True
    tag_set = set(tags)
    return any(soundscape_matches(tag_set, keyword) for keyword in soundscape)


def soundscape_vocabulary() -> list[str]:
    """Every term a soundscape keyword may be built from, for error messages."""
    terms = {*SUBSTRATES, *STYLES, *SPECIAL_GROUPS}
    for group in SPECIAL_GROUPS.values():
        terms.update(group.keywords)
    return sorted(terms)


def _keyword_groups(term: str) -> set[str]:
    """The special groups ``term`` is a keyword of (usually one, possibly none)."""
    return {name for name, group in SPECIAL_GROUPS.items() if term in group.keywords}


def _can_co_occur(first: str, second: str) -> bool:
    """Whether any single track could carry both terms — i.e. they name the two axes of
    one grid cell, or a group and one of *its own* keywords."""
    one_grid_cell = (first in SUBSTRATES and second in STYLES) or (
        first in STYLES and second in SUBSTRATES
    )
    own_keyword = (first in SPECIAL_GROUPS and first in _keyword_groups(second)) or (
        second in SPECIAL_GROUPS and second in _keyword_groups(first)
    )
    return one_grid_cell or own_keyword


def soundscape_problems(keyword: str) -> list[str]:
    """What is wrong with one authored soundscape keyword, as human-readable strings.

    Catches the three ways a keyword ends up selecting nothing — which is invisible until
    someone taps the card and gets whatever the fallback hands them: a term the taxonomy
    doesn't know, more than the two levels a track has, and a pair no track can carry at
    once (``drone.noise_texture``, ``unwind.rain``).
    """
    if not keyword or not isinstance(keyword, str):
        return [f"soundscape keyword {keyword!r} is not a non-empty string"]
    parts = keyword.split(SOUNDSCAPE_SEPARATOR)
    known = set(soundscape_vocabulary())
    problems = [f"soundscape {keyword!r}: unknown term {part!r}" for part in parts if part not in known]
    if problems:
        return problems
    if len(parts) > 2:
        return [
            f"soundscape {keyword!r}: {len(parts)} levels, but a track carries only two "
            f"tags (substrate.style, or group.keyword)"
        ]
    if len(parts) == 2 and not _can_co_occur(*parts):
        return [f"soundscape {keyword!r}: no track can carry both {parts[0]!r} and {parts[1]!r}"]
    return []



@dataclass(frozen=True)
class KeywordSignature:
    """A fully resolved render spec for one (group, keyword, duration) special cell.

    Mirrors :class:`Signature`'s ``track_id``/``spec()`` surface so storage and
    rendering can treat grid and special signatures identically.
    """

    group: SpecialGroup
    keyword: str
    duration_s: int
    seed: int
    prompt: str
    negative_prompt: str
    instrumentation: tuple[str, ...]
    composition_plan: dict[str, Any]

    @property
    def track_id(self) -> str:
        return f"{self.group.name}_{self.keyword}_seed{self.seed}"

    def spec(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "kind": "special",
            "substrate": None,
            "style": None,
            "group": self.group.name,
            "keyword": self.keyword,
            "requested_features": None,
            "measured_features": None,
            "instrumentation": list(self.instrumentation),
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "composition_plan": self.composition_plan,
            "seed": self.seed,
            "duration_s": self.duration_s,
            "loopable": True,
            "render": None,
        }


def _keyword_prompt(group: SpecialGroup, entry: KeywordEntry) -> str:
    """Assemble one special cell's prompt, most specific override winning.

    Same clause order as the grid (:func:`build_prompt`): subject, then how it develops,
    then how dense it may get, then the shared shell.
    """
    if group.development is not None and group.development not in DEVELOPMENT_FRAGMENT:
        raise ValueError(
            f"group {group.name!r} has unknown development {group.development!r}, "
            f"expected one of {list(DEVELOPMENT_FRAGMENT)}"
        )
    density = entry.flowing or group.flowing or density_clause(entry.event_driven)
    lead = entry.description
    if group.development is not None:
        lead = f"{lead}. {DEVELOPMENT_FRAGMENT[group.development]}"
    return f"{lead}. {density}. {entry.body or group.body}"


def build_keyword_signature(
    group_name: str, keyword: str, duration_s: int, variant: int = 0
) -> KeywordSignature:
    """Resolve a (group, keyword) pair into a full render spec, grid-signature style."""
    if group_name not in SPECIAL_GROUPS:
        raise ValueError(f"unknown special group {group_name!r}, expected one of {list(SPECIAL_GROUPS)}")
    group = SPECIAL_GROUPS[group_name]
    if keyword not in group.keywords:
        raise ValueError(
            f"unknown keyword {keyword!r} for group {group_name!r}, expected one of {list(group.keywords)}"
        )
    entry = group.keywords[keyword]
    instrumentation = (keyword,)

    composition_plan: dict[str, Any] = {
        "positive_global_styles": _dedupe(group.global_styles + (keyword,)),
        "negative_global_styles": list(group.negative_global_styles),
        "sections": [
            {
                "section_name": "loop",
                "positive_local_styles": list(instrumentation),
                "negative_local_styles": list(group.negative_global_styles),
                "duration_ms": duration_s * 1000,
                "lines": [],
            }
        ],
    }

    return KeywordSignature(
        group=group,
        keyword=keyword,
        duration_s=duration_s,
        seed=_seed((group_name, keyword), variant),
        prompt=_keyword_prompt(group, entry),
        negative_prompt=group.negative_prompt,
        instrumentation=instrumentation,
        composition_plan=composition_plan,
    )


# --- Provider prompt adaptation ----------------------------------------------
#
# ElevenLabs gets its structure from composition_plan (§4), so the stored prose
# prompt goes through unchanged. Stable Audio 3 has no structured composition
# input — the closest lever to prompt adherence is naming things in the AudioSparx
# metadata vocabulary its text encoder was trained on
# (stable-audio-3/docs/guides/prompting.md, "Helpful AudioSparx Tags"), the same
# technique scripts/try_stable_audio.py uses for its keyword smoke tests.

AUDIOSPARX_BASE_TAGS = "TrackType: Music, VocalType: Instrumental, Genre: Ambient"

# Special cells are field recordings, so they take the guide's SFX slate instead —
# "TrackType: SFX ... tends to produce more semantically reasonable sound effects".
# The music tags were actively harmful here: labelling rain as Genre: Ambient music is
# how the first renders came back as drones.
AUDIOSPARX_SFX_TAGS = "TrackType: SFX"

# Internal instrumentation tags -> the AudioSparx Instruments vocabulary. Special-group
# keywords carry their own KeywordEntry.instrument and bypass this map entirely.
AUDIOSPARX_INSTRUMENTS: dict[str, str] = {
    "felt_piano": "Piano",
    "soft_keys": "Piano",
    "shakuhachi": "Flute",
    "solo_instrument": "Flute",
    "string_pad": "Strings",
    "tibetan_singing_bowls": "Singing Bowls",
    "singing_bowl_drone": "Singing Bowls",
    "singing_bowls": "Singing Bowls",
    "bells": "Chimes",
    "distant_temple_bell": "Chimes",
    "temple_bells": "Chimes",
    "wind_chime": "Chimes",
    "warm_pad": "Synthesizer",
    "sustained_pad": "Synthesizer",
    "low_drone": "Synthesizer",
}


def _audiosparx_instrument(spec: dict[str, Any]) -> str | None:
    """The AudioSparx ``Instruments:`` tag for a spec, if its instrumentation names one."""
    for tag in spec.get("instrumentation", ()):
        if tag in AUDIOSPARX_INSTRUMENTS:
            return AUDIOSPARX_INSTRUMENTS[tag]
    return None


def prompt_for_provider(spec: dict[str, Any], provider: str) -> str:
    """Adapt a spec's stored prompt to a provider's prompt surface.

    ``elevenlabs`` (and any other provider that reads the prompt as-is) gets the
    stored prose unchanged. ``stable_audio`` gets the AudioSparx tag suffix appended —
    the music slate for grid cells, the SFX slate for special ones.
    """
    if provider != "stable_audio":
        return spec["prompt"]
    if spec.get("kind") == "special":
        return f"{spec['prompt']} {AUDIOSPARX_SFX_TAGS}"
    instrument = _audiosparx_instrument(spec)
    suffix = f", Instruments: {instrument}" if instrument else ""
    return f"{spec['prompt']} {AUDIOSPARX_BASE_TAGS}{suffix}"


# A curated, representative slice of the catalog for sample generation. Mirrors
# Stage 1 (the buddhist_meditative + neutral sweep) plus one cell each of the
# other styles so all five styles and all five substrates appear at least once.
SAMPLE_PAIRS: tuple[tuple[str, str], ...] = (
    ("buddhist_meditative", "percussive_with_tail"),  # the flagship: singing bowls
    ("buddhist_meditative", "drone"),  # bowl/mantra drone (wordless hum)
    ("buddhist_meditative", "melodic_instrument"),  # shakuhachi
    ("buddhist_meditative", "field_recording"),  # temple bells over a stream
    ("neutral", "noise_texture"),  # plain brown-noise bed
    ("neoclassical", "melodic_instrument"),  # felt piano
    ("lofi", "melodic_instrument"),  # lo-fi keys
    ("nature_ambient", "field_recording"),  # stream + wind
)


def sample_signatures(duration_s: int) -> list[Signature]:
    """The default sample set, one signature per SAMPLE_PAIRS entry (relax — the goal the
    curated sample set was built for)."""
    return [build_signature(sub, style, "relax", duration_s) for style, sub in SAMPLE_PAIRS]


# --- Coverage-driven enrichment ---------------------------------------------
#
# The library grows one (substrate, style) cell at a time. Rather than hand-pick
# what to add next, we let a coverage guide place the next N specs. "Evenly
# distribute across substrates and styles" == keep the per-cell counts as level
# as possible, which also levels the substrate and style marginals.

Cell = tuple[str, str]  # (substrate, style)


def _resolve_axes(
    substrates: Sequence[str] | None, styles: Sequence[str] | None, goal: str = "relax"
) -> tuple[list[str], list[str]]:
    if goal not in GOALS:
        raise ValueError(f"unknown goal {goal!r}, expected one of {list(GOALS)}")
    subs = list(substrates) if substrates is not None else list(SUBSTRATES)
    stys = list(styles) if styles is not None else list(STYLES)
    for name in subs:
        if name not in SUBSTRATES:
            raise ValueError(f"unknown substrate {name!r}, expected one of {list(SUBSTRATES)}")
    for name in stys:
        if name not in STYLES:
            raise ValueError(f"unknown style {name!r}, expected one of {list(STYLES)}")
    # Goal-incompatible axis members are dropped rather than rejected: an explicit
    # --substrates/--styles restriction combined with --goal focus is meant to intersect,
    # not error (only an unknown *name* above is a mistake worth raising on).
    subs = [name for name in subs if goal in SUBSTRATES[name].goals]
    stys = [name for name in stys if goal in STYLES[name].goals]
    return subs, stys


def coverage_report(
    existing_cells: Iterable[Cell],
    substrates: Sequence[str] | None = None,
    styles: Sequence[str] | None = None,
    goal: str = "relax",
) -> dict[str, Any]:
    """Count how many tracks each cell / substrate / style currently has, for one goal."""
    subs, stys = _resolve_axes(substrates, styles, goal)
    grid = {(sub, sty) for sub in subs for sty in stys}
    counts = Counter(cell for cell in existing_cells if cell in grid)
    return {
        "total": sum(counts.values()),
        "per_cell": {f"{sty}:{sub}": counts[(sub, sty)] for sub in subs for sty in stys},
        "per_substrate": {sub: sum(counts[(sub, sty)] for sty in stys) for sub in subs},
        "per_style": {sty: sum(counts[(sub, sty)] for sub in subs) for sty in stys},
    }


def _fill_cells(
    n: int,
    cells: Sequence[Cell],
    counts: Counter[Cell],
    used: set[str],
    build: Callable[[Cell, int], Any],
    priority: Callable[[Cell], tuple[Any, ...]],
) -> list[Any]:
    """Greedily place ``n`` signatures, each into the currently least-covered cell.

    The shared core of both coverage guides: they differ only in what a cell is and
    how ties break (``priority``, re-evaluated every step because ``counts`` moves),
    and in how a cell becomes a signature (``build(cell, variant)``). The variant
    advances until the track_id is free, so repeated picks of one cell are distinct
    renders rather than collisions with what's already planned.
    """
    picked: list[Any] = []
    if not cells:
        return picked
    for _ in range(max(0, n)):
        cell = min(cells, key=priority)
        variant = 0
        while True:
            sig = build(cell, variant)
            if sig.track_id not in used:
                break
            variant += 1
        picked.append(sig)
        used.add(sig.track_id)
        counts[cell] += 1
    return picked


def plan_coverage(
    n: int,
    duration_s: int,
    *,
    existing_cells: Iterable[Cell] = (),
    used_track_ids: Iterable[str] = (),
    substrates: Sequence[str] | None = None,
    styles: Sequence[str] | None = None,
    goal: str = "relax",
) -> list[Signature]:
    """Pick the next ``n`` signatures that most evenly fill the substrate×style grid.

    Greedy: each step adds to the cell with the fewest tracks, breaking ties by
    the least-covered substrate then style, so the marginals stay balanced too.
    Seeds advance per cell (``variant``) so repeated picks are distinct renders.
    Restrict the grid with ``substrates`` / ``styles`` for targeted coverage. A run
    plans one ``goal`` at a time (a substrate/style not supporting it is dropped from
    the grid, per :func:`_resolve_axes`).
    """
    if n <= 0:
        return []
    subs, stys = _resolve_axes(substrates, styles, goal)
    grid = [(sub, sty) for sub in subs for sty in stys]
    counts = Counter(cell for cell in existing_cells if cell in set(grid))

    def priority(cell: Cell) -> tuple[Any, ...]:
        sub, sty = cell
        return (
            counts[cell],
            sum(counts[(sub, s)] for s in stys),  # substrate marginal
            sum(counts[(u, sty)] for u in subs),  # style marginal
            subs.index(sub),
            stys.index(sty),
        )

    return _fill_cells(
        n,
        grid,
        counts,
        set(used_track_ids),
        lambda cell, variant: build_signature(cell[0], cell[1], goal, duration_s, variant),
        priority,
    )


def fill_to_per_cell(
    target: int,
    duration_s: int,
    *,
    existing_cells: Iterable[Cell] = (),
    used_track_ids: Iterable[str] = (),
    substrates: Sequence[str] | None = None,
    styles: Sequence[str] | None = None,
    goal: str = "relax",
) -> list[Signature]:
    """Coverage guide: bring every selected cell up to ``target`` tracks, for one goal."""
    subs, stys = _resolve_axes(substrates, styles, goal)
    grid = {(sub, sty) for sub in subs for sty in stys}
    counts = Counter(cell for cell in existing_cells if cell in grid)
    needed = sum(max(0, target - counts[cell]) for cell in grid)
    return plan_coverage(
        needed,
        duration_s,
        existing_cells=existing_cells,
        used_track_ids=used_track_ids,
        substrates=substrates,
        styles=styles,
        goal=goal,
    )


# --- Coverage over special groups -------------------------------------------
#
# Special cells get the same treatment, one axis lighter: a keyword belongs to
# exactly one group, so the cells are a ragged list rather than a product, and
# "evenly covered" just means level per-keyword counts (ties to the thinnest group).
# This is the only way to plan more than one seed of a keyword — plain
# ``natural_sounds:rain`` targets are always variant 0.

SpecialCell = Cell  # (group, keyword)


def special_cells(
    groups: Sequence[str] | None = None, keywords: Sequence[str] | None = None
) -> list[SpecialCell]:
    """Every (group, keyword) cell in the named special groups (default: all of them),
    optionally narrowed to specific keyword names within them — the special-cell analogue
    of ``plan_coverage``'s ``substrates``/``styles``, for targeting one or a few keywords
    (e.g. topping up a single new keyword) rather than a whole group."""
    names = list(groups) if groups is not None else list(SPECIAL_GROUPS)
    for name in names:
        if name not in SPECIAL_GROUPS:
            raise ValueError(f"unknown special group {name!r}, expected one of {list(SPECIAL_GROUPS)}")
    cells = [(name, keyword) for name in names for keyword in SPECIAL_GROUPS[name].keywords]
    if keywords is not None:
        wanted = set(keywords)
        known = {keyword for _, keyword in cells}
        unknown = wanted - known
        if unknown:
            raise ValueError(f"unknown keyword(s) {sorted(unknown)}, expected one of {sorted(known)}")
        cells = [cell for cell in cells if cell[1] in wanted]
    return cells


def special_coverage_report(
    existing_cells: Iterable[SpecialCell], groups: Sequence[str] | None = None
) -> dict[str, Any]:
    """Count how many tracks each special cell / group currently has."""
    cells = special_cells(groups)
    counts = Counter(cell for cell in existing_cells if cell in set(cells))
    names = list(dict.fromkeys(group for group, _ in cells))
    return {
        "total": sum(counts.values()),
        "per_cell": {f"{g}:{k}": counts[(g, k)] for g, k in cells},
        "per_group": {name: sum(counts[c] for c in cells if c[0] == name) for name in names},
        "keywords": {name: [k for g, k in cells if g == name] for name in names},
    }


def plan_special_coverage(
    n: int,
    duration_s: int,
    *,
    existing_cells: Iterable[SpecialCell] = (),
    used_track_ids: Iterable[str] = (),
    groups: Sequence[str] | None = None,
    keywords: Sequence[str] | None = None,
) -> list[KeywordSignature]:
    """Pick the next ``n`` keyword signatures that most evenly fill the special groups.

    ``keywords`` narrows to specific keywords the same way ``groups`` narrows to specific
    groups (see :func:`special_cells`) — with both a single group and a single keyword given,
    this fills exactly one cell, e.g. ``n`` more variants of one specific keyword.
    """
    if n <= 0:
        return []
    cells = special_cells(groups, keywords)
    counts = Counter(cell for cell in existing_cells if cell in set(cells))

    def priority(cell: SpecialCell) -> tuple[Any, ...]:
        group, _ = cell
        return (
            counts[cell],
            sum(counts[c] for c in cells if c[0] == group),  # group marginal
            cells.index(cell),
        )

    return _fill_cells(
        n,
        cells,
        counts,
        set(used_track_ids),
        lambda cell, variant: build_keyword_signature(cell[0], cell[1], duration_s, variant),
        priority,
    )


def fill_special_to_per_cell(
    target: int,
    duration_s: int,
    *,
    existing_cells: Iterable[SpecialCell] = (),
    used_track_ids: Iterable[str] = (),
    groups: Sequence[str] | None = None,
    keywords: Sequence[str] | None = None,
) -> list[KeywordSignature]:
    """Coverage guide: bring every keyword of the selected groups up to ``target`` tracks."""
    cells = special_cells(groups, keywords)
    counts = Counter(cell for cell in existing_cells if cell in set(cells))
    needed = sum(max(0, target - counts[cell]) for cell in cells)
    return plan_special_coverage(
        needed,
        duration_s,
        existing_cells=existing_cells,
        used_track_ids=used_track_ids,
        groups=groups,
        keywords=keywords,
    )
