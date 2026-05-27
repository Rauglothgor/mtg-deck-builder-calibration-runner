from __future__ import annotations

import uuid
from pathlib import Path
from uuid import UUID

from _pytest.monkeypatch import MonkeyPatch
from sqlalchemy import delete, select

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
from deckbuilder.experiment.orchestrator import run_experiment
from deckbuilder.forge.parser import SimResult


def test_mocked_experiment_pipeline_persists_rows_and_renders_report(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    commander_id = uuid.uuid4()
    fit_run_id = uuid.uuid4()
    deck_score_by_first_card: dict[UUID, float] = {
        uuid.uuid5(uuid.NAMESPACE_DNS, "deck-42-0"): 0.65,
        uuid.uuid5(uuid.NAMESPACE_DNS, "deck-43-0"): 0.82,
        uuid.uuid5(uuid.NAMESPACE_DNS, "deck-44-0"): 0.79,
        uuid.uuid5(uuid.NAMESPACE_DNS, "deck-45-0"): 0.76,
        uuid.uuid5(uuid.NAMESPACE_DNS, "deck-46-0"): 0.74,
        uuid.uuid5(uuid.NAMESPACE_DNS, "deck-47-0"): 0.71,
    }
    per_match_outcomes = [
        [(1, 0)] * 8 + [(0, 1)] * 2,
        [(1, 0)] * 7 + [(0, 1)] * 3,
        [(1, 0)] * 3 + [(0, 1)] * 7,
        [(1, 0)] * 9 + [(0, 1)] * 1,
        [(1, 0)] * 2 + [(0, 1)] * 8,
    ]
    match_results: list[SimResult] = []
    for deck in per_match_outcomes:
        for win, loss in deck:
            match_results.append(
                SimResult(
                    wins=win,
                    losses=loss,
                    draws=0,
                    game_durations_ms=(1000,),
                    game_winners=("Ai(1)-candidate" if win else "Ai(2)-alela",),
                    raw_output="",
                )
            )
    result_iter = iter(match_results)
    opponent_path = tmp_path / "alela.dck"
    opponent_path.write_text(
        "[metadata]\nName=Alela\n[Commander]\n1 Alela\n[Main]\n1 Sol Ring\n",
        encoding="utf-8",
    )

    def fake_generate_deck(
        _commander_oracle_id: UUID,
        _fit_run_id: UUID,
        seed: int = 42,
    ) -> list[UUID]:
        return [uuid.uuid5(uuid.NAMESPACE_DNS, f"deck-{seed}-{i}") for i in range(3)]

    def fake_score_deck(
        _commander_name: str,
        card_oracle_ids: list[UUID],
        fit_run_id: UUID | None = None,
    ) -> float:
        del fit_run_id
        return deck_score_by_first_card[card_oracle_ids[0]]

    def fake_to_dck_format(
        _commander_oracle_id: UUID,
        _card_oracle_ids: list[UUID],
        output_path: str | Path,
    ) -> Path:
        path = Path(output_path)
        path.write_text(
            "[metadata]\nName=Mock\n[Commander]\n1 Mock Commander\n[Main]\n1 Mock Card\n",
            encoding="utf-8",
        )
        return path

    def fake_run_sim(
        _deck_path: str | Path,
        _opponent_path: str | Path,
        n_matches: int,
        seed: int,
    ) -> SimResult:
        del _deck_path, _opponent_path, seed
        assert n_matches == 1
        return next(result_iter)

    with get_session() as session:
        session.add(
            Card(
                oracle_id=commander_id,
                name="Test Commander",
                mana_cost=None,
                cmc=4.0,
                type_line="Legendary Creature - Test",
                oracle_text="",
                color_identity=["G"],
                legality_commander="legal",
                is_commander_legal_as_commander=True,
                scryfall_uri=None,
            )
        )
        session.commit()
        session.add(
            AwrCoefficient(
                commander_oracle_id=commander_id,
                oracle_id=commander_id,
                fit_run_id=fit_run_id,
                strength_intercept=1.0,
            )
        )
        session.commit()

    monkeypatch.setattr("deckbuilder.experiment.orchestrator.generate_deck", fake_generate_deck)
    monkeypatch.setattr("deckbuilder.experiment.orchestrator.score_deck", fake_score_deck)
    monkeypatch.setattr("deckbuilder.experiment.orchestrator.to_dck_format", fake_to_dck_format)
    monkeypatch.setattr("deckbuilder.experiment.orchestrator.run_sim", fake_run_sim)

    try:
        outcome = run_experiment(
            commander_name="Test Commander",
            n_decks=5,
            matches=10,
            opponent=opponent_path,
            output=tmp_path / "smoke-report.md",
        )

        report_text = outcome.report_path.read_text(encoding="utf-8")
        assert "## Metadata" in report_text
        assert "## Summary Metrics" in report_text
        assert "## Decision Recommendation" in report_text
        assert "## Top 10 Adversarial Cases" in report_text
        assert "## ASCII Scatter" in report_text

        with get_session() as session:
            run_row = session.get(ExperimentRun, outcome.experiment_run_id)
            assert run_row is not None
            assert run_row.status == "completed"
            assert run_row.retry_count == 1
            assert run_row.decision == outcome.calibration.decision
            persisted_sim_results = (
                session.execute(
                    select(SimRow)
                    .join(GeneratedDeck)
                    .where(GeneratedDeck.experiment_run_id == outcome.experiment_run_id)
                )
                .scalars()
                .all()
            )
            assert len(persisted_sim_results) == 5
            assert sum(row.matches_played for row in persisted_sim_results) == 50
    finally:
        with get_session() as session:
            generated_ids = []
            if "outcome" in locals():
                generated_ids = (
                    session.execute(
                        select(GeneratedDeck.id).where(
                            GeneratedDeck.experiment_run_id == outcome.experiment_run_id
                        )
                    )
                    .scalars()
                    .all()
                )
            if generated_ids:
                session.execute(delete(SimRow).where(SimRow.generated_deck_id.in_(generated_ids)))
                session.execute(delete(GeneratedDeck).where(GeneratedDeck.id.in_(generated_ids)))
            if "outcome" in locals():
                session.execute(
                    delete(ExperimentRun).where(ExperimentRun.id == outcome.experiment_run_id)
                )
            session.execute(delete(AwrCoefficient).where(AwrCoefficient.fit_run_id == fit_run_id))
            session.execute(delete(Card).where(Card.oracle_id == commander_id))
            session.commit()
