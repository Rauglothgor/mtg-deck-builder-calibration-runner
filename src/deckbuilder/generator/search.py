"""Local-search deck generation for Task T9."""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from uuid import UUID

from deckbuilder.generator.csp import candidate_pool, commander_profile
from deckbuilder.generator.roles import (
    BOARD_WIPE,
    CARD_DRAW,
    LANDS,
    RAMP,
    REMOVAL,
    ROLE_QUOTAS,
    THEME_FLEX,
    WIN_CONDITION,
    CardProfile,
    count_roles,
    detect_roles,
    primary_role,
)
from deckbuilder.surrogate.awr import (
    _calibration_parameters,
    _load_fit,
    _score_from_model,
)

TARGET_DECK_SIZE = 99
LAND_TARGET = 37
MAX_NON_IMPROVEMENTS = 50
MAX_ITERATIONS = 500
ROLE_BUILD_ORDER = [LANDS, RAMP, CARD_DRAW, REMOVAL, BOARD_WIPE, WIN_CONDITION]
DEFAULT_INIT_TEMPERATURE = 1.0
HARD_ROLE_MAXIMUMS = {LANDS, RAMP}


@dataclass(frozen=True, slots=True)
class LocalSearchStats:
    """Summary metrics from one local-search generation run."""

    initial_score: float
    final_score: float
    iterations: int
    non_improvements_at_stop: int


@dataclass(frozen=True, slots=True)
class GeneratedDeckResult:
    """Full generated deck artifact for reporting and validation."""

    deck: list[UUID]
    stats: LocalSearchStats


def _build_pool_maps(
    commander_oracle_id: UUID,
    commander_name: str,
) -> tuple[list[CardProfile], dict[UUID, CardProfile], dict[str, list[CardProfile]]]:
    pool = candidate_pool(commander_oracle_id)
    by_id = {card.oracle_id: card for card in pool}
    by_role: dict[str, list[CardProfile]] = {
        LANDS: [],
        RAMP: [],
        CARD_DRAW: [],
        REMOVAL: [],
        BOARD_WIPE: [],
        WIN_CONDITION: [],
        THEME_FLEX: [],
    }
    for card in pool:
        roles = detect_roles(card, commander_name)
        for role in roles:
            by_role.setdefault(role, []).append(card)
    return pool, by_id, by_role


def _score_sort_key(
    card: CardProfile,
    coefficients: dict[UUID, float],
) -> tuple[float, str, str]:
    return (-coefficients.get(card.oracle_id, -999.0), card.name, str(card.oracle_id))


def _current_cards(deck_ids: set[UUID], by_id: dict[UUID, CardProfile]) -> list[CardProfile]:
    return [by_id[oracle_id] for oracle_id in sorted(deck_ids, key=str)]


def _meets_role_constraints(cards: list[CardProfile], commander_name: str) -> bool:
    counts = count_roles(cards, commander_name)
    return all(counts[role] >= quota.minimum for role, quota in ROLE_QUOTAS.items()) and all(
        counts[role] <= ROLE_QUOTAS[role].maximum for role in HARD_ROLE_MAXIMUMS
    )


def _can_add_without_role_overflow(
    deck_ids: set[UUID],
    candidate: CardProfile,
    by_id: dict[UUID, CardProfile],
    commander_name: str,
) -> bool:
    trial_ids = set(deck_ids)
    trial_ids.add(candidate.oracle_id)
    counts = count_roles(_current_cards(trial_ids, by_id), commander_name)
    return all(counts[role] <= ROLE_QUOTAS[role].maximum for role in HARD_ROLE_MAXIMUMS)


def _init_temperature() -> float:
    raw_value = os.environ.get("INIT_TEMPERATURE", str(DEFAULT_INIT_TEMPERATURE))
    try:
        temperature = float(raw_value)
    except ValueError as exc:
        msg = f"INIT_TEMPERATURE must be a positive float, got {raw_value!r}"
        raise RuntimeError(msg) from exc
    if temperature <= 0:
        msg = f"INIT_TEMPERATURE must be positive, got {temperature}"
        raise RuntimeError(msg)
    return temperature


def _sample_weighted_card(
    candidates: list[CardProfile],
    coefficients: dict[UUID, float],
    temperature: float,
    rng: random.Random,
) -> CardProfile:
    scores = [coefficients.get(card.oracle_id, -999.0) / temperature for card in candidates]
    max_score = max(scores)
    weights = [math.exp(score - max_score) for score in scores]
    return rng.choices(candidates, weights=weights, k=1)[0]


def _initialize_deck(
    commander_oracle_id: UUID,
    fit_run_id: UUID,
    seed: int,
) -> set[UUID]:
    commander = commander_profile(commander_oracle_id)
    _resolved_commander_oracle_id, _resolved_fit_run_id, coefficients, _synergies = _load_fit(
        commander.name,
        fit_run_id=fit_run_id,
    )
    pool, by_id, by_role = _build_pool_maps(commander_oracle_id, commander.name)
    del pool

    deck_ids: set[UUID] = set()
    rng = random.Random(seed)
    temperature = _init_temperature()

    while True:
        land_count = sum(1 for oid in deck_ids if LANDS in detect_roles(by_id[oid], commander.name))
        if land_count >= LAND_TARGET:
            break
        available_lands = [card for card in by_role[LANDS] if card.oracle_id not in deck_ids]
        if not available_lands:
            msg = "Initialization ran out of available land candidates"
            raise RuntimeError(msg)
        chosen_land = _sample_weighted_card(available_lands, coefficients, temperature, rng)
        deck_ids.add(chosen_land.oracle_id)

    for role in [RAMP, CARD_DRAW, REMOVAL, BOARD_WIPE, WIN_CONDITION]:
        while True:
            current_counts = count_roles(_current_cards(deck_ids, by_id), commander.name)
            if current_counts[role] >= ROLE_QUOTAS[role].minimum:
                break
            available_cards = [
                card
                for card in by_role[role]
                if card.oracle_id not in deck_ids
                and _can_add_without_role_overflow(deck_ids, card, by_id, commander.name)
            ]
            if not available_cards:
                msg = f"Initialization ran out of available candidates for role {role}"
                raise RuntimeError(msg)
            chosen_card = _sample_weighted_card(
                available_cards,
                coefficients,
                temperature,
                rng,
            )
            deck_ids.add(chosen_card.oracle_id)

    while len(deck_ids) < TARGET_DECK_SIZE:
        available_flex = [
            card
            for card in by_role[THEME_FLEX]
            if card.oracle_id not in deck_ids
            and LANDS not in detect_roles(card, commander.name)
            and _can_add_without_role_overflow(deck_ids, card, by_id, commander.name)
        ]
        if not available_flex:
            msg = "Initialization ran out of available flex candidates"
            raise RuntimeError(msg)
        chosen_flex = _sample_weighted_card(available_flex, coefficients, temperature, rng)
        deck_ids.add(chosen_flex.oracle_id)

    if len(deck_ids) != TARGET_DECK_SIZE:
        msg = f"Initialization failed to reach {TARGET_DECK_SIZE} unique cards; got {len(deck_ids)}"
        raise RuntimeError(msg)
    if not _meets_role_constraints(_current_cards(deck_ids, by_id), commander.name):
        msg = "Initialized deck does not satisfy role constraints"
        raise RuntimeError(msg)
    return deck_ids


def _propose_swap(
    deck_ids: set[UUID],
    by_id: dict[UUID, CardProfile],
    by_role: dict[str, list[CardProfile]],
    commander_name: str,
    coefficients: dict[UUID, float],
    rng: random.Random,
) -> tuple[UUID, UUID] | None:
    selected_cards = [by_id[oracle_id] for oracle_id in sorted(deck_ids, key=str)]
    remove_card = rng.choice(selected_cards)
    role = primary_role(remove_card, commander_name)
    replacement_pool = [
        card
        for card in by_role.get(role, [])
        if card.oracle_id not in deck_ids
        and coefficients.get(card.oracle_id, -999.0)
        > coefficients.get(remove_card.oracle_id, -999.0)
    ]
    replacement_pool.sort(key=lambda card: _score_sort_key(card, coefficients))
    for replacement in replacement_pool:
        trial_ids = set(deck_ids)
        trial_ids.remove(remove_card.oracle_id)
        trial_ids.add(replacement.oracle_id)
        if not _meets_role_constraints(_current_cards(trial_ids, by_id), commander_name):
            continue
        return remove_card.oracle_id, replacement.oracle_id
    return None


def generate_deck_with_stats(
    commander_oracle_id: UUID,
    fit_run_id: UUID,
    seed: int = 42,
) -> GeneratedDeckResult:
    """Generate a 99-card deck and return search stats for reporting."""
    commander = commander_profile(commander_oracle_id)
    _resolved_commander_oracle_id, resolved_fit_run_id, coefficients, synergies = _load_fit(
        commander.name,
        fit_run_id=fit_run_id,
    )
    calibration_slope, calibration_intercept = _calibration_parameters(
        commander.name,
        resolved_fit_run_id,
    )
    _pool, by_id, by_role = _build_pool_maps(commander_oracle_id, commander.name)
    deck_ids = _initialize_deck(commander_oracle_id, resolved_fit_run_id, seed)
    initial_score = _score_from_model(
        deck_ids,
        coefficients,
        synergies,
        calibration_slope,
        calibration_intercept,
    )

    rng = random.Random(seed)
    best_ids = set(deck_ids)
    best_score = initial_score
    non_improvements = 0
    iterations = 0
    while iterations < MAX_ITERATIONS and non_improvements < MAX_NON_IMPROVEMENTS:
        iterations += 1
        proposal = _propose_swap(
            best_ids,
            by_id,
            by_role,
            commander.name,
            coefficients,
            rng,
        )
        if proposal is None:
            non_improvements += 1
            continue
        remove_id, add_id = proposal
        trial_ids = set(best_ids)
        trial_ids.remove(remove_id)
        trial_ids.add(add_id)
        trial_score = _score_from_model(
            trial_ids,
            coefficients,
            synergies,
            calibration_slope,
            calibration_intercept,
        )
        if trial_score > best_score:
            best_ids = trial_ids
            best_score = trial_score
            non_improvements = 0
        else:
            non_improvements += 1

    final_cards = _current_cards(best_ids, by_id)
    if len(best_ids) != TARGET_DECK_SIZE:
        msg = f"Generated deck has {len(best_ids)} cards instead of {TARGET_DECK_SIZE}"
        raise RuntimeError(msg)
    if not _meets_role_constraints(final_cards, commander.name):
        msg = "Generated deck failed role constraints after local search"
        raise RuntimeError(msg)
    if not all(
        set(card.color_identity).issubset(set(commander.color_identity)) for card in final_cards
    ):
        msg = "Generated deck contains cards outside commander color identity"
        raise RuntimeError(msg)

    return GeneratedDeckResult(
        deck=sorted(best_ids, key=str),
        stats=LocalSearchStats(
            initial_score=initial_score,
            final_score=best_score,
            iterations=iterations,
            non_improvements_at_stop=non_improvements,
        ),
    )


def generate_deck(
    commander_oracle_id: UUID,
    fit_run_id: UUID,
    seed: int = 42,
) -> list[UUID]:
    """Generate a deterministic 99-card deck for one commander fit run."""
    return generate_deck_with_stats(commander_oracle_id, fit_run_id, seed=seed).deck


def role_sorted_cards(
    commander_oracle_id: UUID,
    deck_ids: list[UUID],
) -> list[tuple[str, CardProfile]]:
    """Return cards paired with their primary role for reporting."""
    commander = commander_profile(commander_oracle_id)
    pool = candidate_pool(commander_oracle_id)
    by_id = {card.oracle_id: card for card in pool}
    cards = [by_id[oracle_id] for oracle_id in deck_ids]
    return sorted(
        ((primary_role(card, commander.name), card) for card in cards),
        key=lambda item: (item[0], item[1].name),
    )
