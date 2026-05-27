"""Unit tests for the embedding encoder helpers."""

from uuid import uuid4

from deckbuilder.db.models import Card
from deckbuilder.embedding.encoder import build_embedding_text


def test_build_embedding_text_handles_missing_oracle_text() -> None:
    """Embedding text should remain stable when oracle text is missing."""
    card = Card(
        oracle_id=uuid4(),
        name="Sol Ring",
        mana_cost="{1}",
        cmc=1.0,
        type_line="Artifact",
        oracle_text=None,
        color_identity=[],
        legality_commander="legal",
        is_commander_legal_as_commander=False,
        scryfall_uri="https://scryfall.com/card/cmm/1/sol-ring",
    )

    assert build_embedding_text(card) == "Sol Ring Artifact"
