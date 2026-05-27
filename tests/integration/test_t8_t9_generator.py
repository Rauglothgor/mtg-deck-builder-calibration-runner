"""Integration checks for T8 candidate filtering and T9 deck generation."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select

from deckbuilder.db.models import AwrCoefficient, Card
from deckbuilder.db.session import get_session
from deckbuilder.generator.csp import candidate_pool
from deckbuilder.generator.roles import ROLE_QUOTAS, count_roles
from deckbuilder.generator.search import generate_deck


def _oracle_id_for(name: str) -> UUID:
    with get_session() as session:
        return session.execute(select(Card.oracle_id).where(Card.name == name)).scalar_one()


def _latest_fit_run_id(commander_oracle_id: UUID) -> UUID:
    with get_session() as session:
        return session.execute(
            select(AwrCoefficient.fit_run_id)
            .where(AwrCoefficient.commander_oracle_id == commander_oracle_id)
            .order_by(AwrCoefficient.created_at.desc())
            .limit(1)
        ).scalar_one()


def test_candidate_pool_acceptance_checks() -> None:
    atraxa_id = _oracle_id_for("Atraxa, Praetors' Voice")
    heliod_id = _oracle_id_for("Heliod, Sun-Crowned")

    atraxa_pool = candidate_pool(atraxa_id)
    heliod_pool = candidate_pool(heliod_id)

    assert 10000 <= len(atraxa_pool) <= 25000
    assert len(heliod_pool) < len(atraxa_pool)
    assert all("R" not in card.color_identity for card in atraxa_pool)


def test_generate_deck_acceptance_checks() -> None:
    atraxa_id = _oracle_id_for("Atraxa, Praetors' Voice")
    fit_run_id = _latest_fit_run_id(atraxa_id)
    deck_one = generate_deck(atraxa_id, fit_run_id, seed=42)
    deck_two = generate_deck(atraxa_id, fit_run_id, seed=42)

    assert len(deck_one) == 99
    assert len(set(deck_one)) == 99
    assert deck_one == deck_two

    pool_by_id = {card.oracle_id: card for card in candidate_pool(atraxa_id)}
    cards = [pool_by_id[oracle_id] for oracle_id in deck_one]
    counts = count_roles(cards, "Atraxa, Praetors' Voice")

    assert all(set(card.color_identity).issubset({"W", "U", "B", "G"}) for card in cards)
    for role, quota in ROLE_QUOTAS.items():
        assert counts[role] >= quota.minimum
