"""Markdown report rendering for experiment calibration runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from deckbuilder.db.models import ExperimentRun
from deckbuilder.experiment.metrics import CalibrationReport, compute_calibration


@dataclass(frozen=True, slots=True)
class ExperimentCase:
    """One predicted-vs-actual experiment case for report rendering."""

    generated_deck_id: str
    predicted_win_rate: float
    actual_win_rate: float
    wins: int
    losses: int
    draws: int
    opponent_deck_name: str

    @property
    def deviation(self) -> float:
        return abs(self.predicted_win_rate - self.actual_win_rate)

    @property
    def calibration_bias(self) -> float:
        return self.predicted_win_rate - self.actual_win_rate

    @property
    def is_adversarial(self) -> bool:
        return self.predicted_win_rate >= 0.70 and self.actual_win_rate < 0.35

    @property
    def is_overconfident(self) -> bool:
        return self.calibration_bias > 0.20


def _template_environment() -> Environment:
    template_dir = Path(__file__).resolve().parent / "templates"
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(enabled_extensions=("html", "xml")),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _format_datetime(value: datetime | None) -> str:
    if value is None:
        return "n/a"
    return value.isoformat(sep=" ", timespec="seconds")


def _ascii_scatter(cases: list[ExperimentCase], width: int = 32, height: int = 12) -> str:
    if not cases:
        return "(no points)"
    grid = [[" " for _ in range(width)] for _ in range(height)]
    for case in cases:
        x = min(width - 1, max(0, round(case.predicted_win_rate * (width - 1))))
        y = min(height - 1, max(0, round((1.0 - case.actual_win_rate) * (height - 1))))
        grid[y][x] = "*"
    rows = ["actual ^"]
    for row in grid:
        rows.append("|" + "".join(row) + "|")
    rows.append("+" + "-" * width + "+")
    rows.append(" predicted -> 0.0" + " " * max(1, width - 10) + "1.0")
    return "\n".join(rows)


def _top_adversarial_cases(cases: list[ExperimentCase], limit: int = 10) -> list[ExperimentCase]:
    ranked = [case for case in cases if case.is_adversarial]
    ranked.sort(
        key=lambda case: (
            -case.predicted_win_rate,
            case.actual_win_rate,
            case.generated_deck_id,
        )
    )
    return ranked[:limit]


def _top_overconfident_cases(cases: list[ExperimentCase], limit: int = 10) -> list[ExperimentCase]:
    ranked = [case for case in cases if case.is_overconfident]
    ranked.sort(
        key=lambda case: (
            -case.calibration_bias,
            -case.predicted_win_rate,
            case.generated_deck_id,
        )
    )
    return ranked[:limit]


def _pairs_from_cases(cases: list[ExperimentCase]) -> list[tuple[float, float]]:
    return [(case.predicted_win_rate, case.actual_win_rate) for case in cases]


def _summary_from_cases(
    experiment_run: ExperimentRun,
    cases: list[ExperimentCase],
) -> CalibrationReport:
    if cases:
        return compute_calibration(_pairs_from_cases(cases))
    return CalibrationReport(
        pair_count=0,
        mean_absolute_deviation=experiment_run.mean_absolute_deviation or 0.0,
        max_deviation=experiment_run.max_deviation or 0.0,
        mean_calibration_bias=0.0,
        overconfidence_rate_20=0.0,
        overconfidence_rate_30=0.0,
        brier_score=0.0,
        adversarial_rate=experiment_run.adversarial_rate or 0.0,
        decision="proceed",
    )


def _decision_from_summary(
    experiment_run: ExperimentRun,
    calibration: CalibrationReport,
    cases: list[ExperimentCase],
) -> str:
    if cases:
        return calibration.decision
    return experiment_run.decision or calibration.decision


def render_experiment_report(
    experiment_run: ExperimentRun,
    cases: list[ExperimentCase],
    commander_name: str,
    output_path: str | Path,
) -> Path:
    """Render a markdown calibration report and write it to disk."""
    environment = _template_environment()
    template = environment.get_template("calibration.md.j2")
    calibration = _summary_from_cases(experiment_run, cases)
    decision = _decision_from_summary(experiment_run, calibration, cases)
    rendered = template.render(
        commander_name=commander_name,
        experiment_run=experiment_run,
        metadata={
            "run_id": str(experiment_run.id),
            "commander_name": commander_name,
            "status": experiment_run.status,
            "n_decks": experiment_run.n_decks,
            "matches_per_deck": experiment_run.matches_per_deck,
            "retry_count": experiment_run.retry_count,
            "started_at": _format_datetime(experiment_run.started_at),
            "completed_at": _format_datetime(experiment_run.completed_at),
        },
        summary={
            "mean_absolute_deviation": calibration.mean_absolute_deviation,
            "max_deviation": calibration.max_deviation,
            "mean_calibration_bias": calibration.mean_calibration_bias,
            "overconfidence_rate_20": calibration.overconfidence_rate_20,
            "overconfidence_rate_30": calibration.overconfidence_rate_30,
            "brier_score": calibration.brier_score,
            "adversarial_rate": calibration.adversarial_rate,
            "case_count": calibration.pair_count,
        },
        decision=decision,
        top_adversarial_cases=_top_adversarial_cases(cases),
        top_overconfident_cases=_top_overconfident_cases(cases),
        all_cases=cases,
        ascii_scatter=_ascii_scatter(cases),
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
    return destination


def build_report_context(
    experiment_run: ExperimentRun,
    cases: list[ExperimentCase],
    commander_name: str,
) -> dict[str, Any]:
    """Return the context pieces used by the markdown report template."""
    calibration = _summary_from_cases(experiment_run, cases)
    decision = _decision_from_summary(experiment_run, calibration, cases)
    return {
        "metadata": {
            "run_id": str(experiment_run.id),
            "commander_name": commander_name,
            "status": experiment_run.status,
            "n_decks": experiment_run.n_decks,
            "matches_per_deck": experiment_run.matches_per_deck,
            "retry_count": experiment_run.retry_count,
        },
        "summary": {
            "mean_absolute_deviation": calibration.mean_absolute_deviation,
            "max_deviation": calibration.max_deviation,
            "mean_calibration_bias": calibration.mean_calibration_bias,
            "overconfidence_rate_20": calibration.overconfidence_rate_20,
            "overconfidence_rate_30": calibration.overconfidence_rate_30,
            "brier_score": calibration.brier_score,
            "adversarial_rate": calibration.adversarial_rate,
            "case_count": calibration.pair_count,
        },
        "decision": decision,
        "top_adversarial_cases": _top_adversarial_cases(cases),
        "top_overconfident_cases": _top_overconfident_cases(cases),
        "ascii_scatter": _ascii_scatter(cases),
    }
