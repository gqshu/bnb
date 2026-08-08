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
from dataclasses import dataclass, field
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


def density_clause(event_driven: bool) -> str:
    """The density limit that suits how this sound is produced."""
    return RESTRAINT if event_driven else STEADINESS


# --- Axis A: substrate (what the sound physically is) -----------------------


@dataclass(frozen=True)
class Substrate:
    """One physical sound type (§2, Axis A).

    ``body`` fills the "[substrate + instrumentation]" slot of the base template;
    the seven axis fields fill the continuous MER slots. ``requested`` is the
    categorical MER coordinate recorded in metadata (§3 ``requested_features``);
    styles may override its ``mode`` / ``nature_bed`` without touching the axis.
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
        harmony="static, minimal movement",
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
    "lofi": Style(
        name="lofi",
        descriptor="lo-fi",
        global_styles=("lofi", "tape texture", "soft keys", "warm", "filtered highs"),
        character=(
            "Warm tape saturation with slow wow and flutter, a dusty vinyl noise floor "
            "and filtered highs, like a worn cassette played late at night"
        ),
        overrides={
            "melodic_instrument": StyleSubstrate(
                body="Soft lo-fi keys with tape hiss and filtered highs, gentle and unhurried",
                instrumentation=("soft_keys", "tape_texture"),
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
    """A fully resolved render spec for one (substrate, style, duration) cell."""

    substrate: Substrate
    style: Style
    duration_s: int
    seed: int
    prompt: str
    negative_prompt: str
    instrumentation: tuple[str, ...]
    composition_plan: dict[str, Any]
    requested_features: dict[str, Any]

    @property
    def track_id(self) -> str:
        return f"{self.style.name}_{self.substrate.short}_seed{self.seed}"

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


# Every track is a bed you sit with for tens of minutes, and the first library came
# back *correct but plain*: the axes were on target and nothing moved. These two
# clauses are the fix, and they're deliberately the only thing loosened — motion at
# the timescale of a breath is what makes a drone worth staying with, and it costs
# nothing on the MER axes because the mean level, brightness and density don't move.
# The engines also simply respond better to this register: the SA3 prompt guide's own
# examples are evocative prose about rooms and gear, not lists of adjectives.
MOTION = (
    "It evolves slowly over minutes — long soft swells that rise and fall like breathing, "
    "faint timbral drift, quiet detail appearing and receding — but it never builds, "
    "resolves, or arrives anywhere"
)


def build_prompt(substrate: Substrate, style: Style, body: str, nature_bed: str) -> str:
    """Fill the §4 base template: substrate body, style descriptor and character, the
    MER axes, and the slow-motion clause that keeps a long listen from going flat."""
    nature = "" if nature_bed == "none" else "A soft natural bed blended gently underneath. "
    character = f"{style.character}. " if style.character else ""
    return (
        f"Instrumental {style.descriptor} soundscape for deep relaxation. "
        f"{body}. "
        f"{MOTION}. "
        f"{density_clause(substrate.event_driven)}. "
        f"{character}"
        f"Tempo {substrate.tempo}, {substrate.energy} energy, very soft dynamics. "
        f"{substrate.timbre} timbre, warm and dark, low spectral brightness. "
        f"{substrate.harmony} harmony, {substrate.texture} texture, {substrate.register} register. "
        f"{nature}"
        "No vocals, no percussion hits, no sudden transitions. "
        "Seamless, calm, continuous."
    )


def build_signature(
    substrate_name: str, style_name: str, duration_s: int, variant: int = 0
) -> Signature:
    """Resolve a (substrate, style) pair into a full render spec.

    Any pair is valid (§2): where the style has a coherent realization for the
    substrate we use it, otherwise the substrate's generic body is coloured by
    the style's global tags. ``variant`` selects a distinct seed within the cell.
    """
    if substrate_name not in SUBSTRATES:
        raise ValueError(f"unknown substrate {substrate_name!r}, expected one of {list(SUBSTRATES)}")
    if style_name not in STYLES:
        raise ValueError(f"unknown style {style_name!r}, expected one of {list(STYLES)}")

    substrate = SUBSTRATES[substrate_name]
    style = STYLES[style_name]
    override = style.overrides.get(substrate_name)

    body = override.body if override else substrate.body
    instrumentation = override.instrumentation if override else substrate.instrumentation
    extra_tags = override.extra_style_tags if override else ()

    # Requested MER coordinate: substrate default, with the style allowed to move
    # only mode/nature_bed (§2: style changes instrumentation/mode, not the axis).
    requested = dict(substrate.requested)
    if style.mode_override is not None:
        requested["mode"] = style.mode_override
    if style.nature_bed_override is not None:
        requested["nature_bed"] = style.nature_bed_override

    positive_global_styles = _dedupe(substrate.style_tags + style.global_styles + extra_tags)

    # One long section == the loop length; empty lines force an instrumental,
    # wordless render (guardrail §6.2). Local styles carry the instrumentation.
    composition_plan: dict[str, Any] = {
        "positive_global_styles": positive_global_styles,
        "negative_global_styles": list(NEGATIVE_GLOBAL_STYLES),
        "sections": [
            {
                "section_name": "loop",
                "positive_local_styles": list(instrumentation),
                "negative_local_styles": list(NEGATIVE_GLOBAL_STYLES),
                "duration_ms": duration_s * 1000,
                "lines": [],
            }
        ],
    }

    return Signature(
        substrate=substrate,
        style=style,
        duration_s=duration_s,
        seed=_seed((style_name, substrate_name), variant),
        prompt=build_prompt(substrate, style, body, requested["nature_bed"]),
        negative_prompt=NEGATIVE_PROMPT,
        instrumentation=instrumentation,
        composition_plan=composition_plan,
        requested_features=requested,
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
    """

    description: str
    event_driven: bool = False


@dataclass(frozen=True)
class SpecialGroup:
    """A keyword-driven category, independent of the substrate x style grid."""

    name: str
    body: str  # the shared prompt shell every keyword in the group layers onto
    keywords: dict[str, KeywordEntry]
    global_styles: tuple[str, ...] = ()  # composition-plan styles common to the group


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
        body=(
            "Calm and unhurried, with no sudden events and no build. Natural stereo "
            "field recording, unprocessed, no music, no instruments, no voices."
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
        },
    ),
}


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
        "negative_global_styles": list(NEGATIVE_GLOBAL_STYLES),
        "sections": [
            {
                "section_name": "loop",
                "positive_local_styles": list(instrumentation),
                "negative_local_styles": list(NEGATIVE_GLOBAL_STYLES),
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
        prompt=f"{entry.description}. {density_clause(entry.event_driven)}. {group.body}",
        negative_prompt=NEGATIVE_PROMPT,
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
    """The default sample set, one signature per SAMPLE_PAIRS entry."""
    return [build_signature(sub, style, duration_s) for style, sub in SAMPLE_PAIRS]


# --- Coverage-driven enrichment ---------------------------------------------
#
# The library grows one (substrate, style) cell at a time. Rather than hand-pick
# what to add next, we let a coverage guide place the next N specs. "Evenly
# distribute across substrates and styles" == keep the per-cell counts as level
# as possible, which also levels the substrate and style marginals.

Cell = tuple[str, str]  # (substrate, style)


def _resolve_axes(
    substrates: Sequence[str] | None, styles: Sequence[str] | None
) -> tuple[list[str], list[str]]:
    subs = list(substrates) if substrates is not None else list(SUBSTRATES)
    stys = list(styles) if styles is not None else list(STYLES)
    for name in subs:
        if name not in SUBSTRATES:
            raise ValueError(f"unknown substrate {name!r}, expected one of {list(SUBSTRATES)}")
    for name in stys:
        if name not in STYLES:
            raise ValueError(f"unknown style {name!r}, expected one of {list(STYLES)}")
    return subs, stys


def coverage_report(
    existing_cells: Iterable[Cell],
    substrates: Sequence[str] | None = None,
    styles: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Count how many tracks each cell / substrate / style currently has."""
    subs, stys = _resolve_axes(substrates, styles)
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
) -> list[Signature]:
    """Pick the next ``n`` signatures that most evenly fill the substrate×style grid.

    Greedy: each step adds to the cell with the fewest tracks, breaking ties by
    the least-covered substrate then style, so the marginals stay balanced too.
    Seeds advance per cell (``variant``) so repeated picks are distinct renders.
    Restrict the grid with ``substrates`` / ``styles`` for targeted coverage.
    """
    if n <= 0:
        return []
    subs, stys = _resolve_axes(substrates, styles)
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
        lambda cell, variant: build_signature(cell[0], cell[1], duration_s, variant),
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
) -> list[Signature]:
    """Coverage guide: bring every selected cell up to ``target`` tracks."""
    subs, stys = _resolve_axes(substrates, styles)
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
    )


# --- Coverage over special groups -------------------------------------------
#
# Special cells get the same treatment, one axis lighter: a keyword belongs to
# exactly one group, so the cells are a ragged list rather than a product, and
# "evenly covered" just means level per-keyword counts (ties to the thinnest group).
# This is the only way to plan more than one seed of a keyword — plain
# ``natural_sounds:rain`` targets are always variant 0.

SpecialCell = Cell  # (group, keyword)


def special_cells(groups: Sequence[str] | None = None) -> list[SpecialCell]:
    """Every (group, keyword) cell in the named special groups (default: all of them)."""
    names = list(groups) if groups is not None else list(SPECIAL_GROUPS)
    for name in names:
        if name not in SPECIAL_GROUPS:
            raise ValueError(f"unknown special group {name!r}, expected one of {list(SPECIAL_GROUPS)}")
    return [(name, keyword) for name in names for keyword in SPECIAL_GROUPS[name].keywords]


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
) -> list[KeywordSignature]:
    """Pick the next ``n`` keyword signatures that most evenly fill the special groups."""
    if n <= 0:
        return []
    cells = special_cells(groups)
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
) -> list[KeywordSignature]:
    """Coverage guide: bring every keyword of the selected groups up to ``target`` tracks."""
    cells = special_cells(groups)
    counts = Counter(cell for cell in existing_cells if cell in set(cells))
    needed = sum(max(0, target - counts[cell]) for cell in cells)
    return plan_special_coverage(
        needed,
        duration_s,
        existing_cells=existing_cells,
        used_track_ids=used_track_ids,
        groups=groups,
    )
