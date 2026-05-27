"""Markdown report rendering for experiment calibration runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from deckbuilder.db.models import ExperimentRun


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
    def is_adversarial(self) -> bool:
        return self.predicted_win_rate >= 0.70 and self.actual_win_rate < 0.35


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


def render_experiment_report(
    experiment_run: ExperimentRun,
    cases: list[ExperimentCase],
    commander_name: str,
    output_path: str | Path,
) -> Path:
    """Render a markdown calibration report and write it to disk."""
    environment = _template_environment()
    template = environment.get_template("calibration.md.j2")
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
            "mean_absolute_deviation": experiment_run.mean_absolute_deviation,
            "max_deviation": experiment_run.max_deviation,
            "adversarial_rate": experiment_run.adversarial_rate,
            "case_count": len(cases),
        },
        decision=experiment_run.decision or "n/a",
        top_adversarial_cases=_top_adversarial_cases(cases),
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
            "mean_absolute_deviation": experiment_run.mean_absolute_deviation,
            "max_deviation": experiment_run.max_deviation,
            "adversarial_rate": experiment_run.adversarial_rate,
        },
        "decision": experiment_run.decision,
        "top_adversarial_cases": _top_adversarial_cases(cases),
        "ascii_scatter": _ascii_scatter(cases),
    }
