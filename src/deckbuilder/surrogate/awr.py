"""AWR surrogate fitting and scoring."""

from __future__ import annotations

import math
import random
import uuid
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import cast
from uuid import UUID

import numpy as np
from sklearn.linear_model import LogisticRegression
from sqlalchemy import delete, select

from deckbuilder.db.models import AwrCoefficient, AwrSynergy, Card, TrainingDeck
from deckbuilder.db.session import get_session

LAPLACE_ALPHA = 1.0
PMI_MIN_SUPPORT = 1
DEFAULT_ACCEPTANCE_SEED = 42


@dataclass(slots=True)
class FitResult:
    """Persisted AWR fit outputs for one commander."""

    fit_run_id: UUID
    commander_oracle_id: UUID
    commander_name: str
    deck_count: int
    unique_card_count: int
    inclusion_counts: dict[UUID, int]
    strength_intercepts: dict[UUID, float]
    joint_counts: dict[tuple[UUID, UUID], int]
    synergy_scores: dict[tuple[UUID, UUID], float]


@dataclass(slots=True)
class ScoredDeck:
    """A scored training deck record."""

    source: str
    score: float


@dataclass(slots=True)
class AcceptanceResult:
    """Outcome for the T7 acceptance check."""

    passed: bool
    top_quartile_source: str
    top_quartile_score: float
    random_score: float
    random_seed: int
    random_sample_size: int


def sigmoid(value: float) -> float:
    """Stable logistic squashing for summed surrogate logits."""
    if value >= 0:
        denominator = 1.0 + math.exp(-value)
        return 1.0 / denominator
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_commander(commander_name: str) -> tuple[UUID, str]:
    with get_session() as session:
        row = session.execute(
            select(Card.oracle_id, Card.name).where(Card.name == commander_name)
        ).one_or_none()
    if row is None:
        msg = f"Commander not found in cards table: {commander_name}"
        raise RuntimeError(msg)
    return cast(UUID, row[0]), cast(str, row[1])


def _load_training_decks(commander_oracle_id: UUID) -> list[tuple[str, list[UUID]]]:
    with get_session() as session:
        rows = session.execute(
            select(TrainingDeck.source, TrainingDeck.card_oracle_ids).where(
                TrainingDeck.commander_oracle_id == commander_oracle_id
            )
        ).all()
    decks = [(cast(str, source), list(cast(list[UUID], card_ids))) for source, card_ids in rows]
    if not decks:
        msg = f"No training decks found for commander oracle_id={commander_oracle_id}"
        raise RuntimeError(msg)
    return decks


def fit_awr(commander_name: str) -> FitResult:
    """Fit log-inclusion intercepts and PMI synergies for one commander corpus."""
    commander_oracle_id, canonical_name = _resolve_commander(commander_name)
    training_decks = _load_training_decks(commander_oracle_id)
    deck_count = len(training_decks)
    unique_decks = [(source, sorted(set(card_ids))) for source, card_ids in training_decks]

    inclusion_counts: Counter[UUID] = Counter()
    joint_counts: Counter[tuple[UUID, UUID]] = Counter()
    for _source, card_ids in unique_decks:
        inclusion_counts.update(card_ids)
        joint_counts.update(combinations(card_ids, 2))

    strength_intercepts = {
        oracle_id: math.log((count + LAPLACE_ALPHA) / (deck_count + 2.0 * LAPLACE_ALPHA))
        for oracle_id, count in inclusion_counts.items()
    }

    synergy_scores: dict[tuple[UUID, UUID], float] = {}
    for pair, joint_count in joint_counts.items():
        if joint_count < PMI_MIN_SUPPORT:
            continue
        card_a, card_b = pair
        probability_ab = joint_count / deck_count
        probability_a = inclusion_counts[card_a] / deck_count
        probability_b = inclusion_counts[card_b] / deck_count
        synergy_scores[pair] = math.log(probability_ab / (probability_a * probability_b))

    fit_run_id = uuid.uuid4()
    with get_session() as session:
        coefficient_rows = [
            AwrCoefficient(
                commander_oracle_id=commander_oracle_id,
                oracle_id=oracle_id,
                fit_run_id=fit_run_id,
                strength_intercept=value,
            )
            for oracle_id, value in strength_intercepts.items()
        ]
        synergy_rows = [
            AwrSynergy(
                commander_oracle_id=commander_oracle_id,
                card_a_oracle_id=pair[0],
                card_b_oracle_id=pair[1],
                fit_run_id=fit_run_id,
                synergy=value,
            )
            for pair, value in synergy_scores.items()
        ]
        session.add_all(coefficient_rows)
        session.add_all(synergy_rows)
        session.commit()

    return FitResult(
        fit_run_id=fit_run_id,
        commander_oracle_id=commander_oracle_id,
        commander_name=canonical_name,
        deck_count=deck_count,
        unique_card_count=len(inclusion_counts),
        inclusion_counts=dict(inclusion_counts),
        strength_intercepts=strength_intercepts,
        joint_counts=dict(joint_counts),
        synergy_scores=synergy_scores,
    )


def _load_fit(
    commander_name: str,
    fit_run_id: UUID | None = None,
) -> tuple[UUID, UUID, dict[UUID, float], dict[tuple[UUID, UUID], float]]:
    commander_oracle_id, _canonical_name = _resolve_commander(commander_name)
    with get_session() as session:
        if fit_run_id is None:
            fit_run_id = session.execute(
                select(AwrCoefficient.fit_run_id)
                .where(AwrCoefficient.commander_oracle_id == commander_oracle_id)
                .order_by(AwrCoefficient.created_at.desc())
                .limit(1)
            ).scalar_one_or_none()
        if fit_run_id is None:
            msg = f"No AWR fit found for commander {commander_name}"
            raise RuntimeError(msg)

        coefficient_rows = session.execute(
            select(AwrCoefficient.oracle_id, AwrCoefficient.strength_intercept).where(
                AwrCoefficient.commander_oracle_id == commander_oracle_id,
                AwrCoefficient.fit_run_id == fit_run_id,
            )
        ).all()
        synergy_rows = session.execute(
            select(
                AwrSynergy.card_a_oracle_id,
                AwrSynergy.card_b_oracle_id,
                AwrSynergy.synergy,
            ).where(
                AwrSynergy.commander_oracle_id == commander_oracle_id,
                AwrSynergy.fit_run_id == fit_run_id,
            )
        ).all()

    coefficients = {cast(UUID, row[0]): cast(float, row[1]) for row in coefficient_rows}
    synergies = {
        (cast(UUID, row[0]), cast(UUID, row[1])): cast(float, row[2]) for row in synergy_rows
    }
    return commander_oracle_id, fit_run_id, coefficients, synergies


def _raw_score_from_model(
    card_oracle_ids: Iterable[UUID],
    coefficients: dict[UUID, float],
    synergies: dict[tuple[UUID, UUID], float],
) -> float:
    unique_cards = sorted(set(card_oracle_ids))
    logit = sum(coefficients.get(oracle_id, 0.0) for oracle_id in unique_cards)
    for pair in combinations(unique_cards, 2):
        logit += synergies.get(pair, 0.0)
    return logit


def _score_from_model(
    card_oracle_ids: Iterable[UUID],
    coefficients: dict[UUID, float],
    synergies: dict[tuple[UUID, UUID], float],
    calibration_slope: float,
    calibration_intercept: float,
) -> float:
    raw_score = _raw_score_from_model(card_oracle_ids, coefficients, synergies)
    return sigmoid((calibration_slope * raw_score) + calibration_intercept)


@lru_cache(maxsize=8)
def _calibration_parameters(
    commander_name: str,
    fit_run_id: UUID | None,
) -> tuple[float, float]:
    commander_oracle_id, resolved_fit_run_id, coefficients, synergies = _load_fit(
        commander_name,
        fit_run_id=fit_run_id,
    )
    training_decks = _load_training_decks(commander_oracle_id)
    card_universe = sorted({oracle_id for _source, deck in training_decks for oracle_id in deck})
    rng = random.Random(f"{resolved_fit_run_id}:{DEFAULT_ACCEPTANCE_SEED}")

    positive_scores = [
        _raw_score_from_model(deck, coefficients, synergies) for _source, deck in training_decks
    ]
    negative_scores = []
    for _source, deck in training_decks:
        sample_size = min(len(set(deck)), len(card_universe))
        negative_deck = rng.sample(card_universe, k=sample_size)
        negative_scores.append(_raw_score_from_model(negative_deck, coefficients, synergies))

    features = np.array(positive_scores + negative_scores, dtype=float).reshape(-1, 1)
    labels = np.array(([1] * len(positive_scores)) + ([0] * len(negative_scores)), dtype=int)
    model = LogisticRegression(random_state=DEFAULT_ACCEPTANCE_SEED, max_iter=1000)
    model.fit(features, labels)
    slope = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])
    return slope, intercept


def score_deck(
    commander_name: str,
    card_oracle_ids: Iterable[UUID],
    fit_run_id: UUID | None = None,
) -> float:
    """Score a deck using the fitted intercept and pairwise synergy terms."""
    _commander_oracle_id, resolved_fit_run_id, coefficients, synergies = _load_fit(
        commander_name,
        fit_run_id=fit_run_id,
    )
    slope, intercept = _calibration_parameters(commander_name, resolved_fit_run_id)
    return _score_from_model(card_oracle_ids, coefficients, synergies, slope, intercept)


def score_training_decks(
    commander_name: str,
    fit_run_id: UUID | None = None,
) -> list[ScoredDeck]:
    """Score all training decks for a commander using one fitted run."""
    commander_oracle_id, resolved_fit_run_id, coefficients, synergies = _load_fit(
        commander_name,
        fit_run_id=fit_run_id,
    )
    slope, intercept = _calibration_parameters(commander_name, resolved_fit_run_id)
    decks = _load_training_decks(commander_oracle_id)
    return [
        ScoredDeck(
            source=source,
            score=_score_from_model(card_ids, coefficients, synergies, slope, intercept),
        )
        for source, card_ids in decks
    ]


def run_acceptance_check(
    commander_name: str,
    fit_run_id: UUID | None = None,
    seed: int = DEFAULT_ACCEPTANCE_SEED,
) -> AcceptanceResult:
    """Assert a top-quartile corpus deck beats a random 99-card sample."""
    commander_oracle_id, resolved_fit_run_id, coefficients, synergies = _load_fit(
        commander_name,
        fit_run_id=fit_run_id,
    )
    slope, intercept = _calibration_parameters(commander_name, resolved_fit_run_id)
    scored_decks = score_training_decks(commander_name, fit_run_id=resolved_fit_run_id)
    if len(scored_decks) < 4:
        msg = "Need at least 4 training decks for quartile acceptance check"
        raise RuntimeError(msg)
    ranked = sorted(scored_decks, key=lambda item: item.score, reverse=True)
    top_quartile_index = max(0, math.ceil(len(ranked) * 0.25) - 1)
    top_quartile_deck = ranked[top_quartile_index]

    all_training_decks = _load_training_decks(commander_oracle_id)
    card_universe = sorted(
        {oracle_id for _source, deck in all_training_decks for oracle_id in deck}
    )
    random_sample_size = min(99, len(card_universe))
    random_sample = random.Random(seed).sample(card_universe, k=random_sample_size)
    random_score = _score_from_model(random_sample, coefficients, synergies, slope, intercept)
    passed = top_quartile_deck.score > random_score

    return AcceptanceResult(
        passed=passed,
        top_quartile_source=top_quartile_deck.source,
        top_quartile_score=top_quartile_deck.score,
        random_score=random_score,
        random_seed=seed,
        random_sample_size=random_sample_size,
    )


def clear_commander_fit(commander_name: str) -> None:
    """Delete all fit rows for one commander. Useful for tests only."""
    commander_oracle_id, _canonical_name = _resolve_commander(commander_name)
    with get_session() as session:
        session.execute(
            delete(AwrSynergy).where(AwrSynergy.commander_oracle_id == commander_oracle_id)
        )
        session.execute(
            delete(AwrCoefficient).where(AwrCoefficient.commander_oracle_id == commander_oracle_id)
        )
        session.commit()
