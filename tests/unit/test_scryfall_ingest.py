"""Regression tests for Scryfall ingest normalization."""

from __future__ import annotations

import json
from pathlib import Path

from deckbuilder.ingest.scryfall import normalize_card

RAW_SCRYFALL = Path(__file__).resolve().parents[1] / "fixtures" / "scryfall_oracle_subset.json"


def _raw_card_by_name(name: str) -> dict[str, object]:
    cards = json.loads(RAW_SCRYFALL.read_text(encoding="utf-8"))
    for card in cards:
        if card.get("name") == name:
            return card
    msg = f"Card not found in raw Scryfall oracle file: {name}"
    raise AssertionError(msg)


def test_normalize_card_keeps_commander_legal_cards_even_when_mtgo_only() -> None:
    """Commander-legal cards must survive oracle-card representative print quirks."""
    for name in ["Savannah", "Plague Myr", "Sol Ring"]:
        raw_card = _raw_card_by_name(name)
        normalized = normalize_card(raw_card)
        assert normalized is not None
        assert normalized["name"] == name
        assert normalized["legality_commander"] == "legal"
