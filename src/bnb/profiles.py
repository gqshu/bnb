"""Client-facing *mode profiles* for the app's music-panel selection grid.

A profile is a ready-to-play preset the app renders as a card: presentation
(``title``, ``subtitle``, corner ``badges``, a background ``image`` or ``gradient``)
plus the ``spec`` it expands to on tap. The spec is a **superset of what the client
already sends to** ``/api/backgrounds/random`` (``goal`` and optional ``type``), carrying
in addition the client-side beat-synth parameters the app's WebAudio engine needs
(``mode``, ``beat_hz``, ``carrier``, ``beat_volume``) — the beat is synthesized on the
device, so these describe it rather than the server rendering anything.

The list is **authored, hand-editable content** living at ``assets/profiles.json`` next
to the rest of the asset repository (its images go in ``assets/profiles/``, served by
``/profile/{filename}``). Unlike ``catalog.json`` — derived state — this is source: it's
the one place the product decides which modes exist and how they're pitched, so it's the
exception the asset repo's ``.gitignore`` keeps versioned. It's re-read on every request
(:func:`load_profiles`), so editing the JSON takes effect without a server restart.

The ``manual`` card is special — it opens the manual control panel and carries no spec.
A card with ``eeg_driven`` omits ``beat_hz`` (the beat will track live EEG, once that
lands). :func:`validate_profiles` cross-checks every spec against the real taxonomy
(:mod:`bnb.background`) so a typo'd goal or type fails loudly rather than shipping a card
that silently picks nothing. Card background: ``image`` wins when present; otherwise the
client uses ``gradient``; the client has its own neutral fallback if a profile gives
neither.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .assets import ASSETS_DIR
from .background import GOALS, SPECIAL_GROUPS, SUBSTRATES

MAX_BADGES = 3


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

    Guards the things a card can get wrong that only surface as an empty grid or a
    404 at play time: a duplicate/missing id, too many badges, or a spec naming a goal
    or type that ``/api/backgrounds/random`` would reject. Raises ``ValueError`` listing
    every problem at once, so one bad hand-edit reports all its issues rather than one
    per fix-and-retry."""
    if not isinstance(profiles, list):
        raise ValueError(f"profiles must be a list, got {type(profiles).__name__}")
    known_types = set(SUBSTRATES) | set(SPECIAL_GROUPS)
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
        typ = spec.get("type")
        if typ is not None and typ not in known_types:
            problems.append(f"{where}: spec.type {typ!r} not in {sorted(known_types)}")
    if problems:
        raise ValueError("invalid profiles:\n  " + "\n  ".join(problems))
