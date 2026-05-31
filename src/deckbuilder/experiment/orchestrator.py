"""Experiment orchestration for the v0.5 calibration run."""

from __future__ import annotations

import csv
from dataclasses import dataclass
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
from deckbuilder.experiment.metrics import CalibrationReport, compute_calibration
from deckbuilder.forge.decklist import to_dck_format
from deckbuilder.forge.parser import SimResult
from deckbuilder.forge.runner import run_sim
from deckbuilder.generator.search import generate_deck
from deckbuilder.report.render import ExperimentCase, render_experiment_report
from deckbuilder.surrogate.awr import score_deck

DEFAULT_OPPONENT_NAME = "alela.dck"
DEFAULT_ATTEMPT_CAP = 500
ELITE_THRESHOLD = 0.70


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


@dataclass(frozen=True, slots=True)
class SelectedCandidateDeck:
    """Candidate selected from one score band for Forge validation."""

    seed: int
    card_oracle_ids: list[UUID]
    predicted_win_rate: float
    score_band: int
    band_min_score: float
    band_max_score: float


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

    ranked = sorted(candidates, key=lambda item: (item.predicted_win_rate, item.seed))
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
                    score_band=band_index,
                    band_min_score=bucket[0].predicted_win_rate,
                    band_max_score=bucket[-1].predicted_win_rate,
                )
            )

    selected.sort(key=lambda item: (item.score_band, item.predicted_win_rate, item.seed))
    return selected


def _build_candidate_pool(
    commander_name: str,
    commander_oracle_id: UUID,
    fit_run_id: UUID,
    seed_start: int,
    candidate_pool_size: int,
) -> list[CandidateDeck]:
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
        candidates.append(
            CandidateDeck(
                seed=seed,
                card_oracle_ids=deck_ids,
                predicted_win_rate=predicted,
            )
        )
        print(
            f"candidate {len(candidates)} generated: "
            f"seed={seed} predicted_win_rate={predicted:.4f}",
            flush=True,
        )
    return candidates


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

    with get_session() as session:
        commander_oracle_id = _resolve_commander(session, commander_name)
        fit_run_id = _resolve_latest_fit_run_id(session, commander_oracle_id)
        experiment_run = ExperimentRun(
            commander_oracle_id=commander_oracle_id,
            n_decks=n_decks,
            matches_per_deck=matches,
            status="running",
            retry_count=0,
        )
        session.add(experiment_run)
        session.commit()
        experiment_run_id = experiment_run.id

    generated_ids: list[UUID] = []
    sim_result_ids: list[UUID] = []
    pairs: list[tuple[float, float]] = []
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

    output_path = Path(output)
    opponent_path = _resolve_opponent_path(opponent)

    with get_session() as session:
        commander_oracle_id = _resolve_commander(session, commander_name)
        fit_run_id = _resolve_latest_fit_run_id(session, commander_oracle_id)
        experiment_run = ExperimentRun(
            commander_oracle_id=commander_oracle_id,
            n_decks=n_decks,
            matches_per_deck=matches,
            status="running",
            retry_count=0,
        )
        session.add(experiment_run)
        session.commit()
        experiment_run_id = experiment_run.id

    generated_ids: list[UUID] = []
    sim_result_ids: list[UUID] = []
    pairs: list[tuple[float, float]] = []
    manifest_rows: list[dict[str, str | int | float]] = []
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
            )
            selected_candidates = _select_score_band_candidates(
                candidates,
                n_decks=n_decks,
                band_count=band_count,
            )
            print(
                f"selected {len(selected_candidates)} candidates from "
                f"{len(candidates)} generated candidates across {band_count} score bands",
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
                    }
                )

                sim_result_id = _create_sim_result_row(generated_deck_id, opponent_path.name)
                sim_result_ids.append(sim_result_id)

                deck_path = tmp_root / f"{generated_deck_id}.dck"
                to_dck_format(commander_oracle_id, selected.card_oracle_ids, deck_path)
                print(
                    f"deck {deck_index} of {n_decks} starting: "
                    f"seed={selected.seed} score_band={selected.score_band} "
                    f"predicted_win_rate={selected.predicted_win_rate:.4f}",
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
                pairs.append((selected.predicted_win_rate, sim_row.actual_win_rate))
                print(
                    f"deck {deck_index} finished: "
                    f"matches_played={sim_row.matches_played}/{matches} "
                    f"failures={deck_failure_count} "
                    f"actual_win_rate={sim_row.actual_win_rate:.4f}",
                    flush=True,
                )

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
        manifest_path = _write_score_band_manifest(output_path, manifest_rows)
        print(f"selection_manifest={manifest_path}", flush=True)
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
