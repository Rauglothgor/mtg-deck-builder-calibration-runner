"""Experiment orchestration for the v0.5 calibration run."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from deckbuilder.config import get_settings
from deckbuilder.db.models import (
    AwrCoefficient,
    Card,
    ExperimentRun,
    GeneratedDeck,
)
from deckbuilder.db.models import (
    SimResult as SimRow,
)
from deckbuilder.db.session import get_session
from deckbuilder.experiment.forge_calibrator import (
    EmpiricalForgeCalibrator,
    ForgeOutcomeModel,
    outcome_features_from_diagnostics,
)
from deckbuilder.experiment.metrics import CalibrationReport, compute_calibration
from deckbuilder.experiment.structure import (
    DeckStructureDiagnostics,
    analyze_deck_structure,
    structural_adjusted_score,
    structural_score_penalty,
    structure_manifest_row,
    write_structure_manifest,
)
from deckbuilder.forge.decklist import to_dck_format
from deckbuilder.forge.parser import SimResult
from deckbuilder.forge.runner import run_sim
from deckbuilder.generator.search import generate_deck
from deckbuilder.report.render import ExperimentCase, render_experiment_report
from deckbuilder.surrogate.awr import score_deck

DEFAULT_OPPONENT_NAME = "alela.dck"
DEFAULT_ATTEMPT_CAP = 500
ELITE_THRESHOLD = 0.70
THEME_BOOST_PER_TAG = 0.005
THEME_BOOST_CAP = 0.03
PRESET_LANE_TAGS = {
    "balanced": ("value", "interaction", "ramp"),
    "proliferate-counters": ("proliferate", "counter", "loyalty"),
    "poison": ("poison", "toxic", "infect", "proliferate"),
    "superfriends": ("planeswalker", "loyalty", "proliferate"),
    "lifegain-value": ("lifegain", "draw", "recursion"),
}


def _bundled_deck_dir() -> Path:
    return get_settings().forge_bundled_deck_dir


@dataclass(frozen=True, slots=True)
class ExperimentOutcome:
    """Summary of one completed experiment orchestration run."""

    experiment_run_id: UUID
    calibration: CalibrationReport
    report_path: Path
    generated_deck_ids: tuple[UUID, ...]
    sim_result_ids: tuple[UUID, ...]
    retry_count: int


@dataclass(frozen=True, slots=True)
class CandidateDeck:
    """Generated candidate deck before Forge validation."""

    seed: int
    card_oracle_ids: list[UUID]
    predicted_win_rate: float
    selection_score: float | None = None
    structure_penalty: float = 0.0
    diagnostics: DeckStructureDiagnostics | None = None
    preset_lane: str = ""
    theme_tags: tuple[str, ...] = ()
    theme_boost: float = 0.0
    pre_rerank_selection_score: float | None = None
    rerank_matches_played: int = 0
    rerank_wins: int = 0
    rerank_losses: int = 0
    rerank_draws: int = 0
    rerank_sim_win_rate: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True, slots=True)
class SelectedCandidateDeck:
    """Candidate selected from one score band for Forge validation."""

    seed: int
    card_oracle_ids: list[UUID]
    predicted_win_rate: float
    selection_score: float
    structure_penalty: float
    score_band: int
    band_min_score: float
    band_max_score: float
    diagnostics: DeckStructureDiagnostics | None = None
    preset_lane: str = ""
    theme_tags: tuple[str, ...] = ()
    theme_boost: float = 0.0
    pre_rerank_selection_score: float | None = None
    rerank_matches_played: int = 0
    rerank_wins: int = 0
    rerank_losses: int = 0
    rerank_draws: int = 0
    rerank_sim_win_rate: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True, slots=True)
class SimulationRerankOutcome:
    """Candidate pool after optional lightweight Forge reranking."""

    candidates: list[CandidateDeck]
    manifest_rows: list[dict[str, str | int | float | bool]]
    retry_count: int


def _candidate_selection_score(candidate: CandidateDeck) -> float:
    return (
        candidate.selection_score
        if candidate.selection_score is not None
        else candidate.predicted_win_rate
    )


def _normalize_theme_tags(raw_tags: str | tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if raw_tags is None:
        return ()
    if isinstance(raw_tags, str):
        candidates = re.split(r"[,;\n|]", raw_tags)
    else:
        candidates = list(raw_tags)
    tags: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        tag = " ".join(str(candidate).strip().lower().split())
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tuple(tags)


def _theme_tags_for_preset(
    preset_lane: str | None,
    theme_tags: str | tuple[str, ...] | list[str] | None,
) -> tuple[str, tuple[str, ...]]:
    lane = (preset_lane or "").strip().lower()
    if lane and lane not in PRESET_LANE_TAGS:
        msg = f"Unsupported preset_lane={preset_lane!r}"
        raise ValueError(msg)
    return lane, _normalize_theme_tags(
        [*PRESET_LANE_TAGS.get(lane, ()), *_normalize_theme_tags(theme_tags)]
    )


def _theme_match_boost_from_texts(
    searchable_cards: list[str], theme_tags: tuple[str, ...]
) -> float:
    if not theme_tags:
        return 0.0
    matched_tags = {
        tag for tag in theme_tags if any(tag in searchable for searchable in searchable_cards)
    }
    return min(THEME_BOOST_CAP, len(matched_tags) * THEME_BOOST_PER_TAG)


def _theme_match_boost(card_oracle_ids: list[UUID], theme_tags: tuple[str, ...]) -> float:
    if not theme_tags:
        return 0.0
    with get_session() as session:
        rows = session.execute(
            select(Card.name, Card.type_line, Card.oracle_text).where(
                Card.oracle_id.in_(card_oracle_ids)
            )
        ).all()
    searchable_cards = [" ".join(str(value or "") for value in row).lower() for row in rows]
    return _theme_match_boost_from_texts(searchable_cards, theme_tags)


def _selection_pool_after_optional_rerank(candidates: list[CandidateDeck]) -> list[CandidateDeck]:
    reranked = [
        candidate for candidate in candidates if candidate.pre_rerank_selection_score is not None
    ]
    return reranked if reranked else candidates


def _resolve_commander(session: Session, commander_name: str) -> UUID:
    row = session.execute(
        select(Card.oracle_id).where(Card.name == commander_name)
    ).scalar_one_or_none()
    if row is None:
        msg = f"Commander not found: {commander_name}"
        raise RuntimeError(msg)
    return row


def _resolve_latest_fit_run_id(session: Session, commander_oracle_id: UUID) -> UUID:
    fit_run_id = session.execute(
        select(AwrCoefficient.fit_run_id)
        .where(AwrCoefficient.commander_oracle_id == commander_oracle_id)
        .order_by(AwrCoefficient.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    if fit_run_id is None:
        msg = f"No AWR fit found for commander oracle id {commander_oracle_id}"
        raise RuntimeError(msg)
    return fit_run_id


def _actual_win_rate_from_counts(wins: int, draws: int, matches_played: int) -> float:
    if matches_played <= 0:
        return 0.0
    return (wins + 0.5 * draws) / matches_played


def _resolve_opponent_path(opponent: str | Path) -> Path:
    opponent_path = Path(opponent)
    if opponent_path.is_file():
        return opponent_path
    bundled = _bundled_deck_dir() / str(opponent)
    if bundled.is_file():
        return bundled
    msg = f"Opponent deck not found: {opponent}"
    raise FileNotFoundError(msg)


def _create_sim_result_row(generated_deck_id: UUID, opponent_name: str) -> UUID:
    with get_session() as session:
        row = SimRow(
            generated_deck_id=generated_deck_id,
            opponent_deck_name=opponent_name,
            matches_played=0,
            wins=0,
            losses=0,
            draws=0,
            actual_win_rate=0.0,
            forge_log_path="in-progress",
        )
        session.add(row)
        session.commit()
        return row.id


def _record_match_result(sim_result_id: UUID, result: SimResult, note: str) -> None:
    with get_session() as session:
        row = session.get(SimRow, sim_result_id)
        if row is None:
            msg = f"Sim result row disappeared: {sim_result_id}"
            raise RuntimeError(msg)
        row.matches_played += result.matches_played
        row.wins += result.wins
        row.losses += result.losses
        row.draws += result.draws
        row.actual_win_rate = _actual_win_rate_from_counts(
            row.wins,
            row.draws,
            row.matches_played,
        )
        row.forge_log_path = note
        session.commit()


def _mark_sim_result_failure(sim_result_id: UUID, note: str) -> None:
    with get_session() as session:
        row = session.get(SimRow, sim_result_id)
        if row is None:
            return
        row.forge_log_path = note
        session.commit()


def _load_report_cases(experiment_run_id: UUID) -> list[ExperimentCase]:
    with get_session() as session:
        rows = session.execute(
            select(
                GeneratedDeck.id,
                GeneratedDeck.predicted_win_rate,
                SimRow.actual_win_rate,
                SimRow.wins,
                SimRow.losses,
                SimRow.draws,
                SimRow.opponent_deck_name,
            )
            .join(SimRow, SimRow.generated_deck_id == GeneratedDeck.id)
            .where(GeneratedDeck.experiment_run_id == experiment_run_id)
            .order_by(GeneratedDeck.created_at.asc())
        ).all()
    return [
        ExperimentCase(
            generated_deck_id=str(deck_id),
            predicted_win_rate=predicted_win_rate,
            actual_win_rate=actual_win_rate,
            wins=wins,
            losses=losses,
            draws=draws,
            opponent_deck_name=opponent_name,
        )
        for deck_id, predicted_win_rate, actual_win_rate, wins, losses, draws, opponent_name in rows
    ]


def _load_sim_row(sim_result_id: UUID) -> SimRow:
    with get_session() as session:
        row = session.get(SimRow, sim_result_id)
        if row is None:
            msg = f"Sim result row disappeared: {sim_result_id}"
            raise RuntimeError(msg)
        session.expunge(row)
        return row


def _record_run_retry_count(experiment_run_id: UUID, retry_count: int) -> None:
    with get_session() as session:
        experiment_run = session.get(ExperimentRun, experiment_run_id)
        if experiment_run is None:
            return
        experiment_run.retry_count = retry_count
        session.commit()


def _select_evenly_from_bucket(
    bucket: list[CandidateDeck],
    count: int,
) -> list[CandidateDeck]:
    if count <= 0:
        return []
    if count > len(bucket):
        msg = f"Cannot select {count} candidates from score band with {len(bucket)} candidates"
        raise RuntimeError(msg)
    if count == 1:
        return [bucket[len(bucket) // 2]]
    selected: list[CandidateDeck] = []
    used_indexes: set[int] = set()
    for index in range(count):
        bucket_index = round(index * (len(bucket) - 1) / (count - 1))
        while bucket_index in used_indexes and bucket_index + 1 < len(bucket):
            bucket_index += 1
        while bucket_index in used_indexes and bucket_index > 0:
            bucket_index -= 1
        used_indexes.add(bucket_index)
        selected.append(bucket[bucket_index])
    return selected


def _select_score_band_candidates(
    candidates: list[CandidateDeck],
    n_decks: int,
    band_count: int,
) -> list[SelectedCandidateDeck]:
    """Select candidates evenly from rank-based surrogate score bands."""
    if n_decks <= 0:
        msg = f"n_decks must be positive, got {n_decks}"
        raise ValueError(msg)
    if band_count <= 0:
        msg = f"band_count must be positive, got {band_count}"
        raise ValueError(msg)
    if len(candidates) < n_decks:
        msg = f"Need at least {n_decks} candidates, got {len(candidates)}"
        raise RuntimeError(msg)

    ranked = sorted(candidates, key=lambda item: (_candidate_selection_score(item), item.seed))
    base_count = n_decks // band_count
    remainder = n_decks % band_count
    selected: list[SelectedCandidateDeck] = []

    for band_index in range(band_count):
        start = round(band_index * len(ranked) / band_count)
        end = round((band_index + 1) * len(ranked) / band_count)
        bucket = ranked[start:end]
        select_count = base_count + (1 if band_index < remainder else 0)
        if select_count == 0:
            continue
        for candidate in _select_evenly_from_bucket(bucket, select_count):
            selected.append(
                SelectedCandidateDeck(
                    seed=candidate.seed,
                    card_oracle_ids=candidate.card_oracle_ids,
                    predicted_win_rate=candidate.predicted_win_rate,
                    selection_score=_candidate_selection_score(candidate),
                    structure_penalty=candidate.structure_penalty,
                    score_band=band_index,
                    band_min_score=_candidate_selection_score(bucket[0]),
                    band_max_score=_candidate_selection_score(bucket[-1]),
                    diagnostics=candidate.diagnostics,
                    preset_lane=candidate.preset_lane,
                    theme_tags=candidate.theme_tags,
                    theme_boost=candidate.theme_boost,
                    pre_rerank_selection_score=candidate.pre_rerank_selection_score,
                    rerank_matches_played=candidate.rerank_matches_played,
                    rerank_wins=candidate.rerank_wins,
                    rerank_losses=candidate.rerank_losses,
                    rerank_draws=candidate.rerank_draws,
                    rerank_sim_win_rate=candidate.rerank_sim_win_rate,
                    rerank_score=candidate.rerank_score,
                )
            )

    selected.sort(key=lambda item: (item.score_band, item.selection_score, item.seed))
    return selected


def _build_candidate_pool(
    commander_name: str,
    commander_oracle_id: UUID,
    fit_run_id: UUID,
    seed_start: int,
    candidate_pool_size: int,
    forge_calibrator: EmpiricalForgeCalibrator | None = None,
    forge_outcome_model: ForgeOutcomeModel | None = None,
    preset_lane: str | None = None,
    theme_tags: str | tuple[str, ...] | list[str] | None = None,
) -> list[CandidateDeck]:
    resolved_preset_lane, resolved_theme_tags = _theme_tags_for_preset(preset_lane, theme_tags)
    candidates: list[CandidateDeck] = []
    seen_decks: set[tuple[str, ...]] = set()
    for offset in range(candidate_pool_size):
        seed = seed_start + offset
        deck_ids = generate_deck(commander_oracle_id, fit_run_id, seed=seed)
        signature = tuple(sorted(str(oracle_id) for oracle_id in deck_ids))
        if signature in seen_decks:
            continue
        seen_decks.add(signature)
        predicted = score_deck(commander_name, deck_ids, fit_run_id=fit_run_id)
        diagnostics = analyze_deck_structure(deck_ids, commander_name, ecms_seed=seed)
        structure_penalty = structural_score_penalty(diagnostics)
        structural_selection_score = structural_adjusted_score(predicted, diagnostics)
        if forge_outcome_model is not None:
            selection_score = forge_outcome_model.predict_features(
                outcome_features_from_diagnostics(
                    predicted_win_rate=predicted,
                    selection_score=structural_selection_score,
                    structure_penalty=structure_penalty,
                    diagnostics=diagnostics,
                )
            )
        elif forge_calibrator is not None:
            selection_score = forge_calibrator.predict(structural_selection_score)
        else:
            selection_score = structural_selection_score
        theme_boost = _theme_match_boost(deck_ids, resolved_theme_tags)
        selection_score = min(1.0, selection_score + theme_boost)
        candidates.append(
            CandidateDeck(
                seed=seed,
                card_oracle_ids=deck_ids,
                predicted_win_rate=predicted,
                selection_score=selection_score,
                structure_penalty=structure_penalty,
                diagnostics=diagnostics,
                preset_lane=resolved_preset_lane,
                theme_tags=resolved_theme_tags,
                theme_boost=theme_boost,
            )
        )
        print(
            f"candidate {len(candidates)} generated: "
            f"seed={seed} predicted_win_rate={predicted:.4f} "
            f"selection_score={selection_score:.4f} "
            f"structure_penalty={structure_penalty:.4f}",
            flush=True,
        )
    return candidates


def _simulation_rerank_score(
    *,
    prior_score: float,
    wins: int,
    draws: int,
    matches_played: int,
    prior_weight: float,
) -> float:
    """Blend learned prior score with a lightweight Forge observation."""
    if prior_weight < 0:
        msg = f"prior_weight must be non-negative, got {prior_weight}"
        raise ValueError(msg)
    if matches_played <= 0:
        return prior_score
    sim_win_rate = _actual_win_rate_from_counts(wins, draws, matches_played)
    denominator = prior_weight + matches_played
    if denominator == 0:
        return sim_win_rate
    return ((prior_score * prior_weight) + (sim_win_rate * matches_played)) / denominator


def _simulation_rerank_candidates(
    *,
    candidates: list[CandidateDeck],
    commander_oracle_id: UUID,
    opponent_path: Path,
    tmp_root: Path,
    shortlist_size: int,
    matches: int,
    prior_weight: float,
) -> SimulationRerankOutcome:
    """Run lightweight Forge sims for the top model-scored candidates and rerank."""
    if shortlist_size <= 0 or matches <= 0:
        return SimulationRerankOutcome(
            candidates=candidates,
            manifest_rows=[],
            retry_count=0,
        )
    if prior_weight < 0:
        msg = f"prior_weight must be non-negative, got {prior_weight}"
        raise ValueError(msg)

    ranked = sorted(
        candidates,
        key=lambda candidate: (_candidate_selection_score(candidate), -candidate.seed),
        reverse=True,
    )
    shortlist = ranked[: min(shortlist_size, len(ranked))]
    pre_rank_by_seed = {candidate.seed: index for index, candidate in enumerate(ranked, start=1)}
    reranked_by_seed: dict[int, CandidateDeck] = {}
    manifest_rows: list[dict[str, str | int | float | bool]] = []
    retry_count = 0

    for candidate in shortlist:
        prior_score = _candidate_selection_score(candidate)
        deck_path = tmp_root / f"rerank-{candidate.seed}.dck"
        to_dck_format(commander_oracle_id, candidate.card_oracle_ids, deck_path)
        wins = 0
        losses = 0
        draws = 0
        failures = 0
        for match_index in range(1, matches + 1):
            try:
                result = run_sim(
                    deck_path,
                    opponent_path,
                    n_matches=1,
                    seed=candidate.seed,
                )
            except Exception as exc:
                failures += 1
                retry_count += 1
                print(
                    f"rerank seed={candidate.seed} match {match_index} failed: {exc}",
                    flush=True,
                )
                continue
            wins += result.wins
            losses += result.losses
            draws += result.draws

        matches_played = wins + losses + draws
        sim_win_rate = (
            _actual_win_rate_from_counts(wins, draws, matches_played) if matches_played else None
        )
        rerank_score = _simulation_rerank_score(
            prior_score=prior_score,
            wins=wins,
            draws=draws,
            matches_played=matches_played,
            prior_weight=prior_weight,
        )
        reranked_candidate = replace(
            candidate,
            selection_score=rerank_score,
            pre_rerank_selection_score=prior_score,
            rerank_matches_played=matches_played,
            rerank_wins=wins,
            rerank_losses=losses,
            rerank_draws=draws,
            rerank_sim_win_rate=sim_win_rate,
            rerank_score=rerank_score,
        )
        reranked_by_seed[candidate.seed] = reranked_candidate
        manifest_rows.append(
            {
                "seed": candidate.seed,
                "pre_rerank_rank": pre_rank_by_seed[candidate.seed],
                "predicted_win_rate": candidate.predicted_win_rate,
                "model_selection_score": prior_score,
                "structure_penalty": candidate.structure_penalty,
                "preset_lane": candidate.preset_lane,
                "theme_tags": "|".join(candidate.theme_tags),
                "theme_boost": candidate.theme_boost,
                "rerank_matches_requested": matches,
                "rerank_matches_played": matches_played,
                "rerank_wins": wins,
                "rerank_losses": losses,
                "rerank_draws": draws,
                "rerank_sim_win_rate": sim_win_rate if sim_win_rate is not None else "",
                "rerank_prior_weight": prior_weight,
                "rerank_score": rerank_score,
                "rerank_failures": failures,
                "selected": False,
                "generated_deck_id": "",
                "score_band": "",
            }
        )
        print(
            "rerank complete: "
            f"seed={candidate.seed} prior={prior_score:.4f} "
            f"matches_played={matches_played}/{matches} "
            f"sim_win_rate={sim_win_rate if sim_win_rate is not None else 'n/a'} "
            f"rerank_score={rerank_score:.4f}",
            flush=True,
        )

    reranked_candidates = [
        reranked_by_seed.get(candidate.seed, candidate) for candidate in candidates
    ]
    return SimulationRerankOutcome(
        candidates=reranked_candidates,
        manifest_rows=manifest_rows,
        retry_count=retry_count,
    )


def _write_score_band_manifest(
    output_path: Path,
    rows: list[dict[str, str | int | float]],
) -> Path:
    manifest_path = output_path.with_suffix(".selection.csv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "generated_deck_id",
                "seed",
                "score_band",
                "band_min_score",
                "band_max_score",
                "predicted_win_rate",
                "selection_score",
                "structure_penalty",
                "preset_lane",
                "theme_tags",
                "theme_boost",
                "pre_rerank_selection_score",
                "rerank_matches_played",
                "rerank_wins",
                "rerank_losses",
                "rerank_draws",
                "rerank_sim_win_rate",
                "rerank_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def _write_sim_rerank_manifest(
    output_path: Path,
    rows: list[dict[str, str | int | float | bool]],
) -> Path | None:
    if not rows:
        return None
    manifest_path = output_path.with_suffix(".rerank.csv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "seed",
                "pre_rerank_rank",
                "predicted_win_rate",
                "model_selection_score",
                "structure_penalty",
                "preset_lane",
                "theme_tags",
                "theme_boost",
                "rerank_matches_requested",
                "rerank_matches_played",
                "rerank_wins",
                "rerank_losses",
                "rerank_draws",
                "rerank_sim_win_rate",
                "rerank_prior_weight",
                "rerank_score",
                "rerank_failures",
                "selected",
                "generated_deck_id",
                "score_band",
                "forge_ai_profile",
                "forge_build_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def run_experiment(
    commander_name: str,
    n_decks: int,
    matches: int,
    opponent: str | Path = DEFAULT_OPPONENT_NAME,
    output: str | Path = "/tmp/deckbuilder-calibration.md",
    seed_start: int = 42,
    attempt_cap: int = DEFAULT_ATTEMPT_CAP,
) -> ExperimentOutcome:
    """Run the experiment orchestration pipeline."""
    if n_decks <= 0:
        msg = f"n_decks must be positive, got {n_decks}"
        raise ValueError(msg)
    if matches <= 0:
        msg = f"matches must be positive, got {matches}"
        raise ValueError(msg)
    if attempt_cap < n_decks:
        msg = f"attempt_cap must be at least n_decks, got {attempt_cap} < {n_decks}"
        raise ValueError(msg)

    output_path = Path(output)
    opponent_path = _resolve_opponent_path(opponent)
    settings = get_settings()

    with get_session() as session:
        commander_oracle_id = _resolve_commander(session, commander_name)
        fit_run_id = _resolve_latest_fit_run_id(session, commander_oracle_id)
        experiment_run = ExperimentRun(
            commander_oracle_id=commander_oracle_id,
            n_decks=n_decks,
            matches_per_deck=matches,
            status="running",
            retry_count=0,
            forge_ai_profile=settings.forge_ai_profile,
            forge_build_id=settings.forge_build_id,
        )
        session.add(experiment_run)
        session.commit()
        experiment_run_id = experiment_run.id

    generated_ids: list[UUID] = []
    sim_result_ids: list[UUID] = []
    pairs: list[tuple[float, float]] = []
    structure_rows: list[dict[str, str | int | float]] = []
    retry_count = 0

    try:
        with TemporaryDirectory(prefix="deckbuilder-experiment-") as tmp_dir:
            tmp_root = Path(tmp_dir)
            attempts = 0
            seed = seed_start
            while len(generated_ids) < n_decks and attempts < attempt_cap:
                deck_ids = generate_deck(commander_oracle_id, fit_run_id, seed=seed)
                predicted = score_deck(commander_name, deck_ids, fit_run_id=fit_run_id)
                attempts += 1
                current_seed = seed
                seed += 1
                if predicted < ELITE_THRESHOLD:
                    retry_count += 1
                    _record_run_retry_count(experiment_run_id, retry_count)
                    print(
                        "seed "
                        f"{current_seed} skipped: predicted_win_rate={predicted:.4f} "
                        f"< {ELITE_THRESHOLD:.2f}",
                        flush=True,
                    )
                    continue

                deck_number = len(generated_ids) + 1
                generated_deck = GeneratedDeck(
                    commander_oracle_id=commander_oracle_id,
                    card_oracle_ids=deck_ids,
                    predicted_win_rate=predicted,
                    experiment_run_id=experiment_run_id,
                )
                with get_session() as session:
                    session.add(generated_deck)
                    session.commit()
                    generated_deck_id = generated_deck.id
                generated_ids.append(generated_deck_id)
                structure_rows.append(
                    structure_manifest_row(
                        generated_deck_id=generated_deck_id,
                        seed=current_seed,
                        predicted_win_rate=predicted,
                        diagnostics=analyze_deck_structure(
                            deck_ids,
                            commander_name,
                            ecms_seed=current_seed,
                        ),
                    )
                )

                sim_result_id = _create_sim_result_row(generated_deck_id, opponent_path.name)
                sim_result_ids.append(sim_result_id)

                deck_path = tmp_root / f"{generated_deck_id}.dck"
                to_dck_format(commander_oracle_id, deck_ids, deck_path)
                print(
                    f"deck {deck_number} of {n_decks} starting: "
                    f"seed={current_seed} predicted_win_rate={predicted:.4f}",
                    flush=True,
                )

                deck_failure_count = 0
                for match_index in range(1, matches + 1):
                    try:
                        match_result = run_sim(
                            deck_path,
                            opponent_path,
                            n_matches=1,
                            seed=current_seed,
                        )
                    except Exception as exc:
                        deck_failure_count += 1
                        retry_count += 1
                        _record_run_retry_count(experiment_run_id, retry_count)
                        note = f"match {match_index} failed: {exc}"
                        _mark_sim_result_failure(sim_result_id, note)
                        print(
                            f"deck {deck_number} of {n_decks} match {match_index} failed: {exc}",
                            flush=True,
                        )
                        continue

                    _record_match_result(
                        sim_result_id,
                        match_result,
                        note=f"match {match_index} complete",
                    )
                    sim_row = _load_sim_row(sim_result_id)
                    print(
                        f"deck {deck_number} of {n_decks} match {match_index} "
                        f"of {matches} complete "
                        f"(w={sim_row.wins} l={sim_row.losses} d={sim_row.draws})",
                        flush=True,
                    )

                sim_row = _load_sim_row(sim_result_id)
                if sim_row.matches_played == 0:
                    note = f"incomplete simulation: played 0 of {matches} matches"
                    _mark_sim_result_failure(sim_result_id, note)
                    msg = f"Deck {deck_number} simulation produced no completed matches"
                    raise RuntimeError(msg)
                if sim_row.matches_played != matches:
                    note = (
                        f"incomplete simulation: played {sim_row.matches_played} "
                        f"of {matches} matches"
                    )
                    _mark_sim_result_failure(sim_result_id, note)
                    print(
                        f"deck {deck_number} incomplete: "
                        f"played {sim_row.matches_played} of {matches} matches; "
                        "including partial result",
                        flush=True,
                    )
                pairs.append((predicted, sim_row.actual_win_rate))
                print(
                    f"deck {deck_number} finished: "
                    f"matches_played={sim_row.matches_played}/{matches} "
                    f"failures={deck_failure_count} "
                    f"actual_win_rate={sim_row.actual_win_rate:.4f}",
                    flush=True,
                )

            if len(generated_ids) < n_decks:
                msg = (
                    f"Could not find {n_decks} elite decks with predicted win rate >= "
                    f"{ELITE_THRESHOLD} within {attempt_cap} attempts"
                )
                raise RuntimeError(msg)

        calibration = compute_calibration(pairs)
        cases = _load_report_cases(experiment_run_id)
        with get_session() as session:
            persisted_run = session.get(ExperimentRun, experiment_run_id)
            if persisted_run is None:
                msg = f"Experiment run disappeared: {experiment_run_id}"
                raise RuntimeError(msg)
            persisted_run.retry_count = retry_count
            persisted_run.mean_absolute_deviation = calibration.mean_absolute_deviation
            persisted_run.max_deviation = calibration.max_deviation
            persisted_run.adversarial_rate = calibration.adversarial_rate
            persisted_run.decision = calibration.decision
            persisted_run.report_path = str(output_path)
            persisted_run.status = "completed"
            persisted_run.completed_at = datetime.now(UTC).replace(tzinfo=None)
            session.commit()
            session.refresh(persisted_run)
            report_path = render_experiment_report(
                persisted_run,
                cases,
                commander_name=commander_name,
                output_path=output_path,
            )
        structure_path = write_structure_manifest(output_path, structure_rows)
        print(f"structure_manifest={structure_path}", flush=True)
        return ExperimentOutcome(
            experiment_run_id=experiment_run_id,
            calibration=calibration,
            report_path=report_path,
            generated_deck_ids=tuple(generated_ids),
            sim_result_ids=tuple(sim_result_ids),
            retry_count=retry_count,
        )
    except Exception:
        with get_session() as session:
            failed_run = session.get(ExperimentRun, experiment_run_id)
            if failed_run is not None:
                failed_run.status = "failed"
                failed_run.retry_count = retry_count
                failed_run.completed_at = datetime.now(UTC).replace(tzinfo=None)
                session.commit()
        raise


def run_score_band_experiment(
    commander_name: str,
    n_decks: int,
    matches: int,
    opponent: str | Path = DEFAULT_OPPONENT_NAME,
    output: str | Path = "/tmp/deckbuilder-score-band-calibration.md",
    seed_start: int = 42,
    candidate_pool_size: int = 100,
    band_count: int = 5,
    forge_calibrator: EmpiricalForgeCalibrator | None = None,
    forge_outcome_model: ForgeOutcomeModel | None = None,
    rerank_shortlist_size: int = 0,
    rerank_matches: int = 0,
    rerank_prior_weight: float = 20.0,
    preset_lane: str | None = None,
    theme_tags: str | tuple[str, ...] | list[str] | None = None,
) -> ExperimentOutcome:
    """Run calibration by sampling generated decks across surrogate score bands."""
    if n_decks <= 0:
        msg = f"n_decks must be positive, got {n_decks}"
        raise ValueError(msg)
    if matches <= 0:
        msg = f"matches must be positive, got {matches}"
        raise ValueError(msg)
    if candidate_pool_size < n_decks:
        msg = f"candidate_pool_size must be at least n_decks, got {candidate_pool_size} < {n_decks}"
        raise ValueError(msg)
    if rerank_shortlist_size < 0:
        msg = f"rerank_shortlist_size must be non-negative, got {rerank_shortlist_size}"
        raise ValueError(msg)
    if rerank_matches < 0:
        msg = f"rerank_matches must be non-negative, got {rerank_matches}"
        raise ValueError(msg)
    if rerank_prior_weight < 0:
        msg = f"rerank_prior_weight must be non-negative, got {rerank_prior_weight}"
        raise ValueError(msg)

    output_path = Path(output)
    opponent_path = _resolve_opponent_path(opponent)
    settings = get_settings()

    with get_session() as session:
        commander_oracle_id = _resolve_commander(session, commander_name)
        fit_run_id = _resolve_latest_fit_run_id(session, commander_oracle_id)
        experiment_run = ExperimentRun(
            commander_oracle_id=commander_oracle_id,
            n_decks=n_decks,
            matches_per_deck=matches,
            status="running",
            retry_count=0,
            forge_ai_profile=settings.forge_ai_profile,
            forge_build_id=settings.forge_build_id,
        )
        session.add(experiment_run)
        session.commit()
        experiment_run_id = experiment_run.id

    generated_ids: list[UUID] = []
    sim_result_ids: list[UUID] = []
    pairs: list[tuple[float, float]] = []
    manifest_rows: list[dict[str, str | int | float]] = []
    rerank_rows: list[dict[str, str | int | float | bool]] = []
    structure_rows: list[dict[str, str | int | float]] = []
    report_cases: list[ExperimentCase] = []
    retry_count = 0

    try:
        with TemporaryDirectory(prefix="deckbuilder-score-bands-") as tmp_dir:
            tmp_root = Path(tmp_dir)
            candidates = _build_candidate_pool(
                commander_name=commander_name,
                commander_oracle_id=commander_oracle_id,
                fit_run_id=fit_run_id,
                seed_start=seed_start,
                candidate_pool_size=candidate_pool_size,
                forge_calibrator=forge_calibrator,
                forge_outcome_model=forge_outcome_model,
                preset_lane=preset_lane,
                theme_tags=theme_tags,
            )
            rerank_outcome = _simulation_rerank_candidates(
                candidates=candidates,
                commander_oracle_id=commander_oracle_id,
                opponent_path=opponent_path,
                tmp_root=tmp_root,
                shortlist_size=rerank_shortlist_size,
                matches=rerank_matches,
                prior_weight=rerank_prior_weight,
            )
            candidates = rerank_outcome.candidates
            rerank_rows = rerank_outcome.manifest_rows
            retry_count += rerank_outcome.retry_count
            if rerank_outcome.retry_count:
                _record_run_retry_count(experiment_run_id, retry_count)
            selection_pool = _selection_pool_after_optional_rerank(candidates)
            selected_candidates = _select_score_band_candidates(
                selection_pool,
                n_decks=n_decks,
                band_count=band_count,
            )
            print(
                f"selected {len(selected_candidates)} candidates from "
                f"{len(selection_pool)} eligible candidates "
                f"({len(candidates)} generated) across {band_count} score bands",
                flush=True,
            )

            for deck_index, selected in enumerate(selected_candidates, start=1):
                generated_deck = GeneratedDeck(
                    commander_oracle_id=commander_oracle_id,
                    card_oracle_ids=selected.card_oracle_ids,
                    predicted_win_rate=selected.predicted_win_rate,
                    experiment_run_id=experiment_run_id,
                )
                with get_session() as session:
                    session.add(generated_deck)
                    session.commit()
                    generated_deck_id = generated_deck.id
                generated_ids.append(generated_deck_id)
                manifest_rows.append(
                    {
                        "generated_deck_id": str(generated_deck_id),
                        "seed": selected.seed,
                        "score_band": selected.score_band,
                        "band_min_score": selected.band_min_score,
                        "band_max_score": selected.band_max_score,
                        "predicted_win_rate": selected.predicted_win_rate,
                        "selection_score": selected.selection_score,
                        "structure_penalty": selected.structure_penalty,
                        "preset_lane": selected.preset_lane,
                        "theme_tags": "|".join(selected.theme_tags),
                        "theme_boost": selected.theme_boost,
                        "pre_rerank_selection_score": (
                            selected.pre_rerank_selection_score
                            if selected.pre_rerank_selection_score is not None
                            else ""
                        ),
                        "rerank_matches_played": selected.rerank_matches_played,
                        "rerank_wins": selected.rerank_wins,
                        "rerank_losses": selected.rerank_losses,
                        "rerank_draws": selected.rerank_draws,
                        "rerank_sim_win_rate": (
                            selected.rerank_sim_win_rate
                            if selected.rerank_sim_win_rate is not None
                            else ""
                        ),
                        "rerank_score": (
                            selected.rerank_score if selected.rerank_score is not None else ""
                        ),
                    }
                )
                for row in rerank_rows:
                    if row["seed"] != selected.seed:
                        continue
                    row["selected"] = True
                    row["generated_deck_id"] = str(generated_deck_id)
                    row["score_band"] = selected.score_band
                diagnostics = selected.diagnostics or analyze_deck_structure(
                    selected.card_oracle_ids,
                    commander_name,
                    ecms_seed=selected.seed,
                )
                structure_rows.append(
                    {
                        **structure_manifest_row(
                            generated_deck_id=generated_deck_id,
                            seed=selected.seed,
                            predicted_win_rate=selected.predicted_win_rate,
                            diagnostics=diagnostics,
                        ),
                        "preset_lane": selected.preset_lane,
                        "theme_tags": "|".join(selected.theme_tags),
                        "theme_boost": selected.theme_boost,
                    }
                )

                sim_result_id = _create_sim_result_row(generated_deck_id, opponent_path.name)
                sim_result_ids.append(sim_result_id)

                deck_path = tmp_root / f"{generated_deck_id}.dck"
                to_dck_format(commander_oracle_id, selected.card_oracle_ids, deck_path)
                print(
                    f"deck {deck_index} of {n_decks} starting: "
                    f"seed={selected.seed} score_band={selected.score_band} "
                    f"predicted_win_rate={selected.predicted_win_rate:.4f} "
                    f"selection_score={selected.selection_score:.4f}",
                    flush=True,
                )

                deck_failure_count = 0
                for match_index in range(1, matches + 1):
                    try:
                        match_result = run_sim(
                            deck_path,
                            opponent_path,
                            n_matches=1,
                            seed=selected.seed,
                        )
                    except Exception as exc:
                        deck_failure_count += 1
                        retry_count += 1
                        _record_run_retry_count(experiment_run_id, retry_count)
                        note = f"match {match_index} failed: {exc}"
                        _mark_sim_result_failure(sim_result_id, note)
                        print(
                            f"deck {deck_index} of {n_decks} match {match_index} failed: {exc}",
                            flush=True,
                        )
                        continue

                    _record_match_result(
                        sim_result_id,
                        match_result,
                        note=f"match {match_index} complete",
                    )
                    sim_row = _load_sim_row(sim_result_id)
                    print(
                        f"deck {deck_index} of {n_decks} match {match_index} "
                        f"of {matches} complete "
                        f"(w={sim_row.wins} l={sim_row.losses} d={sim_row.draws})",
                        flush=True,
                    )

                sim_row = _load_sim_row(sim_result_id)
                if sim_row.matches_played == 0:
                    note = f"incomplete simulation: played 0 of {matches} matches"
                    _mark_sim_result_failure(sim_result_id, note)
                    msg = f"Deck {deck_index} simulation produced no completed matches"
                    raise RuntimeError(msg)
                if sim_row.matches_played != matches:
                    note = (
                        f"incomplete simulation: played {sim_row.matches_played} "
                        f"of {matches} matches"
                    )
                    _mark_sim_result_failure(sim_result_id, note)
                    print(
                        f"deck {deck_index} incomplete: "
                        f"played {sim_row.matches_played} of {matches} matches; "
                        "including partial result",
                        flush=True,
                    )
                report_score = selected.selection_score
                pairs.append((report_score, sim_row.actual_win_rate))
                report_cases.append(
                    ExperimentCase(
                        generated_deck_id=str(generated_deck_id),
                        predicted_win_rate=report_score,
                        actual_win_rate=sim_row.actual_win_rate,
                        wins=sim_row.wins,
                        losses=sim_row.losses,
                        draws=sim_row.draws,
                        opponent_deck_name=sim_row.opponent_deck_name,
                    )
                )
                print(
                    f"deck {deck_index} finished: "
                    f"matches_played={sim_row.matches_played}/{matches} "
                    f"failures={deck_failure_count} "
                    f"actual_win_rate={sim_row.actual_win_rate:.4f}",
                    flush=True,
                )
            for row in rerank_rows:
                row["forge_ai_profile"] = settings.forge_ai_profile
                row["forge_build_id"] = settings.forge_build_id

        calibration = compute_calibration(pairs)
        cases = report_cases
        with get_session() as session:
            persisted_run = session.get(ExperimentRun, experiment_run_id)
            if persisted_run is None:
                msg = f"Experiment run disappeared: {experiment_run_id}"
                raise RuntimeError(msg)
            persisted_run.retry_count = retry_count
            persisted_run.mean_absolute_deviation = calibration.mean_absolute_deviation
            persisted_run.max_deviation = calibration.max_deviation
            persisted_run.adversarial_rate = calibration.adversarial_rate
            persisted_run.decision = calibration.decision
            persisted_run.report_path = str(output_path)
            persisted_run.status = "completed"
            persisted_run.completed_at = datetime.now(UTC).replace(tzinfo=None)
            session.commit()
            session.refresh(persisted_run)
            report_path = render_experiment_report(
                persisted_run,
                cases,
                commander_name=commander_name,
                output_path=output_path,
            )
        manifest_path = _write_score_band_manifest(output_path, manifest_rows)
        rerank_manifest_path = _write_sim_rerank_manifest(output_path, rerank_rows)
        structure_path = write_structure_manifest(output_path, structure_rows)
        print(f"selection_manifest={manifest_path}", flush=True)
        if rerank_manifest_path is not None:
            print(f"rerank_manifest={rerank_manifest_path}", flush=True)
        print(f"structure_manifest={structure_path}", flush=True)
        return ExperimentOutcome(
            experiment_run_id=experiment_run_id,
            calibration=calibration,
            report_path=report_path,
            generated_deck_ids=tuple(generated_ids),
            sim_result_ids=tuple(sim_result_ids),
            retry_count=retry_count,
        )
    except Exception:
        with get_session() as session:
            failed_run = session.get(ExperimentRun, experiment_run_id)
            if failed_run is not None:
                failed_run.status = "failed"
                failed_run.retry_count = retry_count
                failed_run.completed_at = datetime.now(UTC).replace(tzinfo=None)
                session.commit()
        raise
