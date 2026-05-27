"""Constraint filtering implementation for Task T8."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select

from deckbuilder.db.models import Card
from deckbuilder.db.session import get_session
from deckbuilder.generator.roles import CardProfile


def _to_profile(card: Card) -> CardProfile:
    return CardProfile(
        oracle_id=card.oracle_id,
        name=card.name,
        color_identity=tuple(card.color_identity),
        type_line=card.type_line,
        oracle_text=card.oracle_text or "",
        legality_commander=card.legality_commander,
        is_commander_legal_as_commander=card.is_commander_legal_as_commander,
    )


def commander_profile(commander_oracle_id: UUID) -> CardProfile:
    """Load the commander card profile for downstream color filtering."""
    with get_session() as session:
        commander = session.get(Card, commander_oracle_id)
    if commander is None:
        msg = f"Commander not found: {commander_oracle_id}"
        raise RuntimeError(msg)
    return _to_profile(commander)


def candidate_pool(commander_oracle_id: UUID) -> list[CardProfile]:
    """Return commander-legal cards within the commander's color identity."""
    commander = commander_profile(commander_oracle_id)
    commander_colors = set(commander.color_identity)
    with get_session() as session:
        rows = session.execute(select(Card).where(Card.legality_commander == "legal")).scalars()
        pool = []
        for card in rows:
            if card.oracle_id == commander_oracle_id:
                continue
            if not set(card.color_identity).issubset(commander_colors):
                continue
            pool.append(_to_profile(card))
    pool.sort(key=lambda card: (card.name, str(card.oracle_id)))
    return pool
