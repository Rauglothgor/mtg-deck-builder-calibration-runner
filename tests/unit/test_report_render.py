from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from deckbuilder.db.models import ExperimentRun
from deckbuilder.report.render import ExperimentCase, render_experiment_report


def test_render_experiment_report_contains_required_sections(tmp_path: Path) -> None:
    experiment_run = ExperimentRun(
        id=uuid4(),
        commander_oracle_id=uuid4(),
        n_decks=5,
        matches_per_deck=100,
        status="completed",
        retry_count=2,
        started_at=datetime(2026, 5, 19, 19, 0, 0, tzinfo=UTC),
        completed_at=datetime(2026, 5, 19, 19, 5, 0, tzinfo=UTC),
        mean_absolute_deviation=0.12,
        max_deviation=0.41,
        adversarial_rate=0.20,
        decision="frequent_validation",
    )
    cases = [
        ExperimentCase(
            generated_deck_id=str(uuid4()),
            predicted_win_rate=0.81,
            actual_win_rate=0.20,
            wins=2,
            losses=8,
            draws=0,
            opponent_deck_name="alela.dck",
        ),
        ExperimentCase(
            generated_deck_id=str(uuid4()),
            predicted_win_rate=0.74,
            actual_win_rate=0.62,
            wins=6,
            losses=4,
            draws=0,
            opponent_deck_name="alela.dck",
        ),
    ]

    report_path = render_experiment_report(
        experiment_run,
        cases,
        commander_name="Atraxa, Praetors' Voice",
        output_path=tmp_path / "report.md",
    )
    text = report_path.read_text(encoding="utf-8")

    for heading in [
        "## Metadata",
        "## Summary Metrics",
        "## Decision Recommendation",
        "## Top 10 Adversarial Cases",
        "## ASCII Scatter",
        "## All Cases",
        "## Recommendation Rationale",
    ]:
        assert heading in text
    assert "Decision: **frequent_validation**" in text
