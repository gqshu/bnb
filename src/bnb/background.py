"""Background-soundscape library: the substrate x style taxonomy.

This is the provider-agnostic core of the background media library described in
docs/background_music.md. It turns a (substrate, style) *signature* into the three
things a render needs:

- the ElevenLabs-shaped prompt built from the §4 base template + style modifier,
- the composition plan (positive/negative global styles, one loop section),
- the per-track metadata record (§3 schema), minus the measured MER vector that
  the objective-feature extraction pipeline fills in after the render.

Scope is down-regulation only, so every continuous MER axis (tempo, energy,
brightness, harmony, texture, register, nature-bed) is authored biased low. The
spec is written by scripts/plan_background.py and rendered by
scripts/render_background.py; nothing here imports the ElevenLabs SDK, so the
taxonomy stays importable and testable without an API key.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

# Applies to every down-regulation track (§4). Kept as prose (prompt) and as
# discrete style tags (composition plan) because ElevenLabs consumes both.
NEGATIVE_PROMPT = (
    "bright, harsh, energetic, fast, distorted, drums, buildup, EDM, sudden transitions"
)
NEGATIVE_GLOBAL_STYLES: tuple[str, ...] = (
    "bright",
    "aggressive",
    "drums",
    "vocals",
    "fast",
    "buildup",
)


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
        body="A steady tonal noise wash, like gentle pink or brown noise",
        instrumentation=("noise_texture",),
        style_tags=("noise texture", "steady wash"),
        tempo="arrhythmic",
        energy="very low",
        timbre="smooth, dark",
        harmony="no tonal center",
        texture="flat and even",
        register="low-weighted, full",
        requested={
            "tempo_bpm": None,
            "energy": "very_low",
            "spectral_centroid": "dark_smooth",
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
    overrides: dict[str, StyleSubstrate] = field(default_factory=dict)
    mode_override: str | None = None  # e.g. buddhist -> just_intonation
    nature_bed_override: str | None = None  # e.g. nature_ambient -> present


STYLES: dict[str, Style] = {
    "neutral": Style(
        name="neutral",
        descriptor="ambient",
        global_styles=("instrumental", "ambient", "calm", "minimal"),
    ),
    # One coherent tradition per track, wordless by default (§6 guardrails):
    # the drone override uses a *wordless* hum, never a mantra with real text.
    "buddhist_meditative": Style(
        name="buddhist_meditative",
        descriptor="meditative",
        global_styles=("meditative", "just intonation", "instrumental", "very sparse", "low register"),
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
            "substrate": self.substrate.name,
            "style": self.style.name,
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


def _seed(style_name: str, substrate_name: str, variant: int = 0) -> int:
    """A deterministic per-signature seed so re-renders stay reproducible (§1).

    ``variant`` gives a cell more than one render (the doc wants 3-5 seeds per
    cell, §5). ``variant == 0`` keeps the original one-seed-per-cell identity so
    existing track_ids stay stable.
    """
    key = f"{style_name}:{substrate_name}"
    if variant:
        key = f"{key}:{variant}"
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:4], "big") % 100_000


def _dedupe(tags: tuple[str, ...]) -> list[str]:
    """Order-preserving dedupe for style-tag lists."""
    return list(dict.fromkeys(tags))


def build_prompt(substrate: Substrate, style: Style, body: str, nature_bed: str) -> str:
    """Fill the §4 base template: substrate body + style descriptor + MER axes."""
    nature = "" if nature_bed == "none" else "A soft natural bed blended gently underneath. "
    return (
        f"Instrumental {style.descriptor} soundscape for deep relaxation. "
        f"{body}. "
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
        seed=_seed(style_name, substrate_name, variant),
        prompt=build_prompt(substrate, style, body, requested["nature_bed"]),
        negative_prompt=NEGATIVE_PROMPT,
        instrumentation=instrumentation,
        composition_plan=composition_plan,
        requested_features=requested,
    )


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


def _next_signature(substrate: str, style: str, duration_s: int, used: set[str]) -> Signature:
    """The lowest-variant signature for a cell whose track_id isn't taken yet."""
    variant = 0
    while True:
        sig = build_signature(substrate, style, duration_s, variant)
        if sig.track_id not in used:
            return sig
        variant += 1


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
    grid = {(sub, sty) for sub in subs for sty in stys}
    counts = Counter(cell for cell in existing_cells if cell in grid)
    used = set(used_track_ids)

    signatures: list[Signature] = []
    for _ in range(n):
        sub_marginal = {sub: sum(counts[(sub, sty)] for sty in stys) for sub in subs}
        sty_marginal = {sty: sum(counts[(sub, sty)] for sub in subs) for sty in stys}
        sub, sty = min(
            grid,
            key=lambda cell: (
                counts[cell],
                sub_marginal[cell[0]],
                sty_marginal[cell[1]],
                subs.index(cell[0]),
                stys.index(cell[1]),
            ),
        )
        sig = _next_signature(sub, sty, duration_s, used)
        signatures.append(sig)
        used.add(sig.track_id)
        counts[(sub, sty)] += 1
    return signatures


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
