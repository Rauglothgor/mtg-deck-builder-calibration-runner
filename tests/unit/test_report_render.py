from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from deckbuilder.db.models import ExperimentRun
from deckbuilder.report.render import (
    ExperimentCase,
    ForgeRunIdentity,
    build_report_context,
    render_experiment_report,
    require_single_forge_identity,
)


def test_render_experiment_report_contains_required_sections(tmp_path: Path) -> None:
    experiment_run = ExperimentRun(
        id=uuid4(),
        commander_oracle_id=uuid4(),
        n_decks=5,
        matches_per_deck=100,
        status="completed",
        retry_count=2,
        forge_ai_profile="forge-daily-snapshot",
        forge_build_id="2.0.13-SNAPSHOT-06.11-2026-06-11T19:12:03Z",
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
        "## Top 10 Overconfident Cases",
        "## ASCII Scatter",
        "## All Cases",
        "## Recommendation Rationale",
    ]:
        assert heading in text
    assert "Decision: **pivot**" in text
    assert "Mean calibration bias" in text
    assert "Overconfidence rate > 0.20" in text
    assert "Forge AI profile: `forge-daily-snapshot`" in text
    assert "Forge build ID: `2.0.13-SNAPSHOT-06.11-2026-06-11T19:12:03Z`" in text
    assert "| Deck ID | Predicted | Actual | Bias | Deviation |" in text


def test_report_context_keeps_forge_ai_profiles_distinguishable() -> None:
    shared_kwargs = {
        "commander_oracle_id": uuid4(),
        "n_decks": 5,
        "matches_per_deck": 100,
        "status": "completed",
        "retry_count": 0,
    }
    baseline_run = ExperimentRun(
        id=uuid4(),
        forge_ai_profile="forge-baseline",
        forge_build_id="unknown",
        **shared_kwargs,
    )
    snapshot_run = ExperimentRun(
        id=uuid4(),
        forge_ai_profile="forge-daily-snapshot",
        forge_build_id="2.0.13-SNAPSHOT-06.11-2026-06-11T19:12:03Z",
        **shared_kwargs,
    )

    baseline_context = build_report_context(baseline_run, [], "Atraxa, Praetors' Voice")
    snapshot_context = build_report_context(snapshot_run, [], "Atraxa, Praetors' Voice")

    assert baseline_context["metadata"]["forge_ai_profile"] == "forge-baseline"
    assert baseline_context["metadata"]["forge_build_id"] == "unknown"
    assert snapshot_context["metadata"]["forge_ai_profile"] == "forge-daily-snapshot"
    assert (
        snapshot_context["metadata"]["forge_build_id"]
        == "2.0.13-SNAPSHOT-06.11-2026-06-11T19:12:03Z"
    )


def test_require_single_forge_identity_rejects_mixed_builds() -> None:
    shared_kwargs = {
        "commander_oracle_id": uuid4(),
        "n_decks": 5,
        "matches_per_deck": 100,
        "status": "completed",
        "retry_count": 0,
    }
    baseline_run = ExperimentRun(
        id=uuid4(),
        forge_ai_profile="forge-baseline",
        forge_build_id="unknown",
        **shared_kwargs,
    )
    snapshot_run = ExperimentRun(
        id=uuid4(),
        forge_ai_profile="forge-daily-snapshot",
        forge_build_id="2.0.13-SNAPSHOT-06.11-2026-06-11T19:12:03Z",
        **shared_kwargs,
    )

    with pytest.raises(ValueError, match="Mixed Forge AI builds"):
        require_single_forge_identity([baseline_run, snapshot_run])

    assert require_single_forge_identity([baseline_run, snapshot_run], allow_mixed=True) is None


def test_require_single_forge_identity_returns_normalized_identity() -> None:
    run = ExperimentRun(
        id=uuid4(),
        commander_oracle_id=uuid4(),
        n_decks=5,
        matches_per_deck=100,
        status="completed",
        retry_count=0,
        forge_ai_profile="",
        forge_build_id="",
    )

    assert require_single_forge_identity([run]) == ForgeRunIdentity(
        forge_ai_profile="forge-baseline",
        forge_build_id="unknown",
    )
