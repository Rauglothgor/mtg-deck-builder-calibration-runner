"""Deck-structure diagnostics for cheap pre-Forge validation."""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from uuid import UUID

from sqlalchemy import select

from deckbuilder.db.models import Card
from deckbuilder.db.session import get_session
from deckbuilder.generator.roles import (
    BOARD_WIPE,
    CARD_DRAW,
    LANDS,
    RAMP,
    REMOVAL,
    WIN_CONDITION,
    CardProfile,
    count_roles,
)

DEFAULT_ECMS_TRIALS = 200
DEFAULT_ECMS_TURNS = 7


@dataclass(frozen=True, slots=True)
class StructureCard:
    """Card fields needed for deck-structure diagnostics."""

    oracle_id: UUID
    name: str
    cmc: float
    type_line: str
    oracle_text: str
    color_identity: tuple[str, ...] = ()
    legality_commander: str = "legal"
    is_commander_legal_as_commander: bool = False

    @property
    def is_land(self) -> bool:
        return "Land" in self.type_line

    @property
    def is_ramp(self) -> bool:
        profile = self.to_card_profile()
        return count_roles([profile], "")[RAMP] > 0 or _text_is_ramp(self.oracle_text)

    def to_card_profile(self) -> CardProfile:
        """Convert to the generator role-detection profile shape."""
        return CardProfile(
            oracle_id=self.oracle_id,
            name=self.name,
            color_identity=self.color_identity,
            type_line=self.type_line,
            oracle_text=self.oracle_text,
            legality_commander=self.legality_commander,
            is_commander_legal_as_commander=self.is_commander_legal_as_commander,
        )


@dataclass(frozen=True, slots=True)
class DeckStructureDiagnostics:
    """Cheap structural signals used to explain or filter surrogate scores."""

    card_count: int
    land_count: int
    nonland_count: int
    ramp_count: int
    card_draw_count: int
    removal_count: int
    board_wipe_count: int
    win_condition_count: int
    average_nonland_cmc: float
    median_nonland_cmc: float
    low_curve_nonland_count: int
    high_curve_nonland_count: int
    expected_compounded_mana_spent: float


def _text_is_ramp(oracle_text: str) -> bool:
    text = oracle_text.lower()
    return (
        "add " in text
        or "search your library for" in text
        or "treasure token" in text
        or ("untap" in text and "land" in text)
    )


def _load_structure_cards(card_oracle_ids: list[UUID]) -> list[StructureCard]:
    with get_session() as session:
        rows = session.execute(
            select(
                Card.oracle_id,
                Card.name,
                Card.cmc,
                Card.type_line,
                Card.oracle_text,
                Card.color_identity,
                Card.legality_commander,
                Card.is_commander_legal_as_commander,
            ).where(Card.oracle_id.in_(card_oracle_ids))
        ).all()

    by_id = {
        oracle_id: StructureCard(
            oracle_id=oracle_id,
            name=name,
            cmc=float(cmc or 0.0),
            type_line=type_line,
            oracle_text=oracle_text or "",
            color_identity=tuple(color_identity or ()),
            legality_commander=legality_commander,
            is_commander_legal_as_commander=is_commander_legal_as_commander,
        )
        for (
            oracle_id,
            name,
            cmc,
            type_line,
            oracle_text,
            color_identity,
            legality_commander,
            is_commander_legal_as_commander,
        ) in rows
    }
    missing = [oracle_id for oracle_id in card_oracle_ids if oracle_id not in by_id]
    if missing:
        msg = f"Could not resolve structure cards for oracle ids: {missing}"
        raise RuntimeError(msg)
    return [by_id[oracle_id] for oracle_id in card_oracle_ids]


def _cast_turn_spells(
    hand: list[StructureCard],
    available_mana: int,
) -> tuple[float, int]:
    spent = 0.0
    ramp_added = 0
    castable = sorted(
        [card for card in hand if not card.is_land and card.cmc <= available_mana],
        key=lambda card: (card.is_ramp, card.cmc),
        reverse=True,
    )
    for card in castable:
        cost = int(card.cmc)
        if cost <= 0 or cost > available_mana:
            continue
        available_mana -= cost
        spent += cost
        hand.remove(card)
        if card.is_ramp:
            ramp_added += 1
    return spent, ramp_added


def expected_compounded_mana_spent(
    cards: list[StructureCard],
    *,
    trials: int = DEFAULT_ECMS_TRIALS,
    turns: int = DEFAULT_ECMS_TURNS,
    seed: int = 0,
) -> float:
    """Estimate early-game mana utilization with a deterministic goldfish simulation.

    This intentionally stays approximate: it models one land per turn, simple mana
    availability, and ramp as +1 future mana after being cast. It is meant to be a
    cheap rank/filter signal, not a rules-engine replacement.
    """
    if not cards or trials <= 0 or turns <= 0:
        return 0.0

    rng = random.Random(seed)
    totals: list[float] = []
    for _trial in range(trials):
        library = list(cards)
        rng.shuffle(library)
        hand = library[:7]
        library_index = 7
        lands_in_play = 0
        ramp_mana = 0
        compounded_spent = 0.0

        for turn in range(1, turns + 1):
            if library_index < len(library):
                hand.append(library[library_index])
                library_index += 1

            land = next((card for card in hand if card.is_land), None)
            if land is not None:
                hand.remove(land)
                lands_in_play += 1

            available_mana = lands_in_play + ramp_mana
            spent, ramp_added = _cast_turn_spells(hand, available_mana)
            ramp_mana += ramp_added
            compounded_spent += spent * (turns - turn + 1)

        totals.append(compounded_spent)

    return mean(totals)


def analyze_structure_cards(
    cards: list[StructureCard],
    commander_name: str,
    *,
    ecms_trials: int = DEFAULT_ECMS_TRIALS,
    ecms_turns: int = DEFAULT_ECMS_TURNS,
    ecms_seed: int = 0,
) -> DeckStructureDiagnostics:
    """Compute structural diagnostics for a resolved deck."""
    profiles = [card.to_card_profile() for card in cards]
    roles = count_roles(profiles, commander_name)
    ramp_count = max(
        roles[RAMP],
        sum(1 for card in cards if not card.is_land and card.is_ramp),
    )
    nonlands = [card for card in cards if not card.is_land]
    nonland_cmcs = [card.cmc for card in nonlands]
    low_curve = [card for card in nonlands if card.cmc <= 2]
    high_curve = [card for card in nonlands if card.cmc >= 5]

    return DeckStructureDiagnostics(
        card_count=len(cards),
        land_count=roles[LANDS],
        nonland_count=len(nonlands),
        ramp_count=ramp_count,
        card_draw_count=roles[CARD_DRAW],
        removal_count=roles[REMOVAL],
        board_wipe_count=roles[BOARD_WIPE],
        win_condition_count=roles[WIN_CONDITION],
        average_nonland_cmc=mean(nonland_cmcs) if nonland_cmcs else 0.0,
        median_nonland_cmc=median(nonland_cmcs) if nonland_cmcs else 0.0,
        low_curve_nonland_count=len(low_curve),
        high_curve_nonland_count=len(high_curve),
        expected_compounded_mana_spent=expected_compounded_mana_spent(
            cards,
            trials=ecms_trials,
            turns=ecms_turns,
            seed=ecms_seed,
        ),
    )


def analyze_deck_structure(
    card_oracle_ids: list[UUID],
    commander_name: str,
    *,
    ecms_trials: int = DEFAULT_ECMS_TRIALS,
    ecms_turns: int = DEFAULT_ECMS_TURNS,
    ecms_seed: int = 0,
) -> DeckStructureDiagnostics:
    """Load cards from the database and compute structural diagnostics."""
    cards = _load_structure_cards(card_oracle_ids)
    return analyze_structure_cards(
        cards,
        commander_name,
        ecms_trials=ecms_trials,
        ecms_turns=ecms_turns,
        ecms_seed=ecms_seed,
    )


def write_structure_manifest(
    output_path: Path,
    rows: list[dict[str, str | int | float]],
) -> Path:
    """Write a structural diagnostics sidecar next to a calibration report."""
    manifest_path = output_path.with_suffix(".structure.csv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "generated_deck_id",
        "seed",
        "predicted_win_rate",
        *DeckStructureDiagnostics.__dataclass_fields__.keys(),
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def structure_manifest_row(
    generated_deck_id: UUID,
    seed: int,
    predicted_win_rate: float,
    diagnostics: DeckStructureDiagnostics,
) -> dict[str, str | int | float]:
    """Return one CSV-ready structural diagnostics row."""
    return {
        "generated_deck_id": str(generated_deck_id),
        "seed": seed,
        "predicted_win_rate": predicted_win_rate,
        **asdict(diagnostics),
    }
