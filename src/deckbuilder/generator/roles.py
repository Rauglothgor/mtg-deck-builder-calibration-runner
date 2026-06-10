"""Frozen role detection rules and quota definitions for T9."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final
from uuid import UUID

ROLE_ORDER: Final[tuple[str, ...]] = (
    "lands",
    "ramp",
    "card_draw",
    "removal",
    "board_wipe",
    "win_condition",
    "theme_flex",
)

LANDS = "lands"
RAMP = "ramp"
CARD_DRAW = "card_draw"
REMOVAL = "removal"
BOARD_WIPE = "board_wipe"
WIN_CONDITION = "win_condition"
THEME_FLEX = "theme_flex"


@dataclass(frozen=True, slots=True)
class RoleQuota:
    """Minimum and maximum desired counts for one deck-construction role."""

    minimum: int
    maximum: int


@dataclass(frozen=True, slots=True)
class CardProfile:
    """Minimal card data needed for candidate and role filtering."""

    oracle_id: UUID
    name: str
    color_identity: tuple[str, ...]
    type_line: str
    oracle_text: str
    legality_commander: str
    is_commander_legal_as_commander: bool


ROLE_QUOTAS: Final[dict[str, RoleQuota]] = {
    LANDS: RoleQuota(minimum=36, maximum=38),
    RAMP: RoleQuota(minimum=8, maximum=12),
    CARD_DRAW: RoleQuota(minimum=8, maximum=12),
    REMOVAL: RoleQuota(minimum=6, maximum=10),
    BOARD_WIPE: RoleQuota(minimum=2, maximum=4),
    WIN_CONDITION: RoleQuota(minimum=3, maximum=6),
}

RAMP_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(\badd\s+(?:\{[wubrgc0-9x/]+\}|(?:one|two|three|four|five|six|seven|eight|nine|ten|x|\d+)"
    r"(?:\s+mana)?)|search your library for (?:a|an|up to .*?) land|"
    r"create a treasure token|create .*? treasure token|untap up to .* lands?)",
    re.IGNORECASE | re.DOTALL,
)
CARD_DRAW_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\bdraw (?:a|one|two|three|four|five|six|seven|eight|nine|ten|x|\d+) cards?\b",
    re.IGNORECASE,
)
REMOVAL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(destroy target|exile target|return target .*? to (?:its owner's|their owner's|owner's) hand|"
    r"counter target spell|target player sacrifices)",
    re.IGNORECASE | re.DOTALL,
)
BOARD_WIPE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(destroy all|exile all|each creature gets -\d+/|-x/-x|each other creature)",
    re.IGNORECASE,
)

COMMANDER_WIN_CONDITIONS: Final[dict[str, frozenset[str]]] = {
    "Atraxa, Praetors' Voice": frozenset(
        {
            "Doubling Season",
            "Vorinclex, Monstrous Raider",
            "Tekuthal, Inquiry Dominus",
            "Deepglow Skate",
            "Contagion Engine",
            "Skithiryx, the Blight Dragon",
            "Triumph of the Hordes",
            "Vraska, Betrayal's Sting",
            "Inexorable Tide",
            "Flux Channeler",
        }
    ),
    "Heliod, Sun-Crowned": frozenset(
        {
            "Walking Ballista",
            "Archangel of Thune",
            "Aetherflux Reservoir",
            "Resplendent Angel",
            "Ajani's Pridemate",
        }
    ),
}


def detect_roles(card: CardProfile, commander_name: str) -> set[str]:
    """Return all roles matched by the frozen regex and commander archetype rules."""
    roles: set[str] = set()
    oracle_text = card.oracle_text
    is_land = "Land" in card.type_line
    if is_land:
        roles.add(LANDS)
    if not is_land and RAMP_PATTERN.search(oracle_text):
        roles.add(RAMP)
    if CARD_DRAW_PATTERN.search(oracle_text):
        roles.add(CARD_DRAW)
    if REMOVAL_PATTERN.search(oracle_text):
        roles.add(REMOVAL)
    if BOARD_WIPE_PATTERN.search(oracle_text):
        roles.add(BOARD_WIPE)
        roles.add(REMOVAL)
    if card.name in COMMANDER_WIN_CONDITIONS.get(commander_name, frozenset()):
        roles.add(WIN_CONDITION)
    if not roles:
        roles.add(THEME_FLEX)
    else:
        roles.add(THEME_FLEX)
    return roles


def primary_role(card: CardProfile, commander_name: str) -> str:
    """Return the first matching role in priority order for same-role swaps."""
    roles = detect_roles(card, commander_name)
    for role in ROLE_ORDER:
        if role in roles:
            return role
    return THEME_FLEX


def count_roles(cards: list[CardProfile], commander_name: str) -> dict[str, int]:
    """Count role coverage across a decklist."""
    counts = dict.fromkeys(ROLE_ORDER, 0)
    for card in cards:
        for role in detect_roles(card, commander_name):
            counts[role] += 1
    return counts


def meets_role_minimums(cards: list[CardProfile], commander_name: str) -> bool:
    """Return True when the deck satisfies all Section 9 role minimums."""
    counts = count_roles(cards, commander_name)
    return all(counts[role] >= quota.minimum for role, quota in ROLE_QUOTAS.items())
