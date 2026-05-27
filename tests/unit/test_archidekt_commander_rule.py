"""Tests for Archidekt commander detection."""

from __future__ import annotations

import json
from pathlib import Path

from deckbuilder.ingest.archidekt import is_commander

ARCHIDEKT_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "archidekt"
ATRAXA = "Atraxa, Praetors' Voice"

POSITIVE_DECK_IDS = [
    22723343,
    22724227,
    22717217,
    22723289,
    14172391,
    22720649,
    2035732,
    22677354,
]
NEGATIVE_DECK_IDS = [22715374, 22690244]


def load_deck(deck_id: int) -> dict[str, object]:
    """Load a saved Phase 2A deck detail fixture from disk."""
    return json.loads((ARCHIDEKT_FIXTURES / f"deck_{deck_id}.json").read_text(encoding="utf-8"))


def test_is_commander_matches_all_positive_phase2a_decks() -> None:
    """Atraxa should be tagged as commander in the eight positive fixtures."""
    results = [is_commander(load_deck(deck_id), ATRAXA) for deck_id in POSITIVE_DECK_IDS]
    assert results == [True] * 8


def test_is_commander_excludes_phase2a_ninety_nine_only_decks() -> None:
    """Atraxa in the 99 should not be mistaken for commander."""
    results = [is_commander(load_deck(deck_id), ATRAXA) for deck_id in NEGATIVE_DECK_IDS]
    assert results == [False, False]


def test_is_commander_yields_eight_positive_and_two_negative_results() -> None:
    """The full Phase 2A sample should resolve to 8 positives and 2 negatives."""
    deck_ids = POSITIVE_DECK_IDS + NEGATIVE_DECK_IDS
    results = {deck_id: is_commander(load_deck(deck_id), ATRAXA) for deck_id in deck_ids}
    assert sum(results.values()) == 8
    assert len(results) - sum(results.values()) == 2
