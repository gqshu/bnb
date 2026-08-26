"""Client-facing *mode profiles* for the app's music-panel selection grid.

A profile is a ready-to-play preset the app renders as a card: presentation
(``title``, ``subtitle``, corner ``badges``, a background ``image`` or ``gradient``)
plus the ``spec`` it expands to on tap. The spec is a **superset of what the client
already sends to** ``/api/backgrounds/random`` (``goal`` and optional ``soundscape``), carrying
in addition the client-side beat-synth parameters the app's WebAudio engine needs
(``mode``, ``beat_hz``, ``carrier``, ``beat_volume``) — the beat is synthesized on the
device, so these describe it rather than the server rendering anything. ``spec.soundscape``
is a *list* of taxonomy keywords and the card's pool is their union
(§ :func:`bnb.background.soundscape_matches`).

The list is **authored, hand-editable content** living at ``assets/profiles.json`` next
to the rest of the asset repository (its images go in ``assets/profiles/``, served by
``/profile/{filename}``). Unlike ``catalog.json`` — derived state — this is source: it's
the one place the product decides which modes exist and how they're pitched, so it's the
exception the asset repo's ``.gitignore`` keeps versioned. It's re-read on every request
(:func:`load_profiles`), so editing the JSON takes effect without a server restart.

Every card declares a ``source``. The mini program ships its **personal** cards built in
— the manual panel and the EEG-driven preset both describe the listener's own hardware
and session, so there is nothing for a catalogue to say about them — and what it fetches
is **community** music, the shared cards. ``scripts/prepare_cloud_assets.py`` publishes
the community ones to the bucket; the served endpoint returns the file whole, since the
server has no bucket to curate. A ``manual`` card carries no spec at all, and a card with
``eeg_driven`` omits ``beat_hz`` (the beat tracks live EEG instead).

:func:`validate_profiles` cross-checks every card against the things that only break at
play time: a ``spec.goal``/``spec.soundscape`` the taxonomy doesn't know, a ``spec.mode`` the
app's WebAudio engine can't build, and a ``gradient`` whose colour contradicts the card's
goal. That last one is presentation, but it is *load-bearing* presentation — the grid uses
warm cards for focus and cool cards for relax, so a card is telling the user what it does
before they read it, and a green one is a lie rather than a taste difference. It holds for
``community`` cards, the ones a listener meets cold; the built-in ``personal`` grid is
colour-coded by mode identity instead and is exempt (:func:`validate_profiles`). Card
background: ``image`` wins when present; otherwise the client uses ``gradient``. Both are
optional — given neither, the client generates a gradient from the card's ``goal``, in the
same hue bands this validates against, seeded by the card id so it stays put between
launches. So authoring a colour is for cards that want *their own* identity; a card content
with the convention can just say what it does and let the grid colour it.
"""

from __future__ import annotations

import colorsys
import json
import re
from pathlib import Path
from typing import Any

from .assets import ASSETS_DIR
from .background import GOALS, soundscape_problems

MAX_BADGES = 3

SOURCES = ("community", "personal")
"""Where a card comes from, and therefore whether it belongs in the bucket.

``personal`` cards are the ones the mini program ships built in (the manual panel, the
EEG-driven preset); ``community`` cards are the shared music the app fetches. Required
rather than defaulted: the cloud build filters on it, so a card that omitted it would
quietly fail to publish, which is exactly the kind of absence nobody notices."""

BEAT_MODES = ("binaural", "monaural", "isochronic")
"""The beat modalities the app's WebAudio engine can actually build (``audio.ts``'s
``BeatMode``).

Worth being strict about, because an unknown value degrades quietly rather than loudly.
``audio.ts`` builds anything that isn't binaural or monaural as the AM path, so a spec
saying ``am_music`` *sounds* like it works — but ``applyBeatLevel`` routes the beat level
by an exact ``=== 'isochronic'`` test, so the level lands on the (disconnected) beat gain
instead of the modulation depth and the entrainment never actually turns on. The card
plays; it just isn't doing anything. ``isochronic`` is this codebase's name for that AM
path — the app labels it 魔改."""

# Focus is warm, relax is cool, and the grid leans on that to say what a card does before
# anyone reads it. Stated as hue *bands* rather than fixed gradients because the cards are
# authored and meant to differ from one another — a teal, an indigo and a slate blue are
# all legibly "relax"; a green is not. Bands are in degrees on the HSV wheel, and a band
# that wraps through 0 is written as a pair.
GOAL_HUES: dict[str, tuple[tuple[float, float], ...]] = {
    "focus": ((330.0, 360.0), (0.0, 40.0)),  # red through red-orange
    "relax": ((185.0, 260.0),),  # cyan-blue through indigo
}

# Colours too washed out or too dark to read as any hue — the near-black end of a gradient
# is a shadow, not a statement, and holding it to the band would just force fake precision.
MIN_GRADIENT_SATURATION = 0.15
MIN_GRADIENT_VALUE = 0.10

_HEX_COLOR = re.compile(r"#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b")


def _hue_degrees(hex_color: str) -> float | None:
    """Hue of a CSS hex colour, or ``None`` if it is too grey or too dark to have one."""
    digits = hex_color.lstrip("#")
    if len(digits) == 3:
        digits = "".join(c * 2 for c in digits)
    r, g, b = (int(digits[i : i + 2], 16) / 255 for i in (0, 2, 4))
    hue, saturation, value = colorsys.rgb_to_hsv(r, g, b)
    if saturation < MIN_GRADIENT_SATURATION or value < MIN_GRADIENT_VALUE:
        return None
    return hue * 360


def gradient_hues(gradient: str) -> list[float]:
    """The hues of every colour stop in a CSS gradient that has one worth checking."""
    found = (_hue_degrees(match.group(0)) for match in _HEX_COLOR.finditer(gradient))
    return [hue for hue in found if hue is not None]


def gradient_matches_goal(gradient: str, goal: str) -> bool:
    """Whether every readable colour stop sits in the band its goal claims.

    Every stop, not just the first: a card that starts red and ends green is exactly the
    drift this exists to catch, and it is invisible until someone opens the app.
    """
    bands = GOAL_HUES.get(goal)
    if bands is None:
        return True
    return all(any(lo <= hue <= hi for lo, hi in bands) for hue in gradient_hues(gradient))


def profiles_path(root: Path = ASSETS_DIR) -> Path:
    """The authored profile catalog file."""
    return root / "profiles.json"


def profiles_image_dir(root: Path = ASSETS_DIR) -> Path:
    """Where profile card background images live (served by ``/profile/{filename}``)."""
    return root / "profiles"


def load_profiles(root: Path = ASSETS_DIR) -> list[dict[str, Any]]:
    """Read and validate the authored profile list from ``assets/profiles.json``.

    Accepts either the served envelope (``{"profiles": [...]}``) or a bare list, so the
    file can be hand-edited in whichever shape is convenient. Read fresh on every call —
    the file is tiny and this is what lets an edit show up without a server restart.
    Raises ``ValueError`` (via :func:`validate_profiles`) on a malformed list, or
    ``OSError`` / ``json.JSONDecodeError`` if the file is missing or unparseable.
    """
    raw = json.loads(profiles_path(root).read_text())
    profiles = raw.get("profiles", []) if isinstance(raw, dict) else raw
    validate_profiles(profiles)
    return profiles


# Backwards-compatible alias: the endpoint and other callers ask for "the profiles list".
list_profiles = load_profiles


def validate_profiles(profiles: list[dict[str, Any]]) -> None:
    """Fail loudly on a malformed profile list, cross-checked against the taxonomy.

    Guards the things a card can get wrong that only surface as an empty grid, a 404, or
    a beat that plays silently: a duplicate/missing id, too many badges, an unknown
    ``source``, a spec naming a goal or soundscape keyword ``/api/backgrounds/random``
    would reject, a ``mode`` the app can't build (:data:`BEAT_MODES`), or — on a community
    card — a gradient whose colour contradicts the goal (:data:`GOAL_HUES`). Raises
    ``ValueError`` listing every problem at once, so one bad hand-edit reports all its issues rather than one per
    fix-and-retry."""
    if not isinstance(profiles, list):
        raise ValueError(f"profiles must be a list, got {type(profiles).__name__}")
    problems: list[str] = []
    seen: set[str] = set()
    for i, p in enumerate(profiles):
        pid = p.get("id")
        where = pid or f"#{i}"
        if not pid:
            problems.append(f"{where}: missing id")
        elif pid in seen:
            problems.append(f"{where}: duplicate id")
        else:
            seen.add(pid)
        if not p.get("title"):
            problems.append(f"{where}: missing title")
        source = p.get("source")
        if source not in SOURCES:
            problems.append(f"{where}: source {source!r} not in {sorted(SOURCES)}")
        badges = p.get("badges") or []
        if len(badges) > MAX_BADGES:
            problems.append(f"{where}: {len(badges)} badges, max {MAX_BADGES}")
        for b in badges:
            if not b.get("text"):
                problems.append(f"{where}: badge with no text")
        spec = p.get("spec")
        if spec is None:
            if not p.get("manual"):
                problems.append(f"{where}: no spec and not manual")
            continue
        goal = spec.get("goal")
        if goal not in GOALS:
            problems.append(f"{where}: spec.goal {goal!r} not in {sorted(GOALS)}")
        # ``type`` was this filter's previous, single-valued shape. Rejected rather than
        # ignored: a leftover ``type`` would silently stop filtering, and "card plays the
        # wrong bed" is exactly the failure nobody reports as a bug.
        if "type" in spec:
            problems.append(f"{where}: spec.type is gone — use spec.soundscape (a list of keywords)")
        soundscape = spec.get("soundscape")
        if soundscape is not None:
            if not isinstance(soundscape, list):
                problems.append(
                    f"{where}: spec.soundscape must be a list of keywords, got "
                    f"{type(soundscape).__name__}"
                )
            else:
                for keyword in soundscape:
                    problems.extend(f"{where}: {p}" for p in soundscape_problems(keyword))
        mode = spec.get("mode")
        if mode is not None and mode not in BEAT_MODES:
            problems.append(f"{where}: spec.mode {mode!r} not in {sorted(BEAT_MODES)}")
        gradient = p.get("gradient")
        # Community cards only. The goal-hue convention earns its keep on cards the
        # listener has never seen — colour tells them what it does before they read it.
        # The built-in grid is the opposite case: six fixed cards seen every session,
        # colour-coded by *mode identity* (电子咖啡 green, 能量小憩 amber), which a listener
        # learns once and then reads faster than any convention. Holding those to the goal
        # bands would cost them that identity to restate what the card already says.
        if source == "personal":
            gradient = None
        if gradient and goal in GOAL_HUES and not gradient_matches_goal(gradient, goal):
            bands = "/".join(f"{lo:g}-{hi:g}°" for lo, hi in GOAL_HUES[goal])
            hues = ", ".join(f"{h:.0f}°" for h in gradient_hues(gradient))
            problems.append(
                f"{where}: gradient hue ({hues}) doesn't read as {goal} — expected {bands}"
            )
    if problems:
        raise ValueError("invalid profiles:\n  " + "\n  ".join(problems))
