"""Fit empirical calibrators from Forge validation artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Literal, cast

ScoreField = Literal["predicted_win_rate", "selection_score"]


@dataclass(frozen=True, slots=True)
class CalibrationObservation:
    """One generated deck with surrogate scores and Forge-observed win rate."""

    generated_deck_id: str
    score_band: int
    predicted_win_rate: float
    selection_score: float
    structure_penalty: float
    actual_win_rate: float


@dataclass(frozen=True, slots=True)
class EmpiricalCalibrationBin:
    """A score interval mapped to an empirical Forge win rate."""

    score_min: float
    score_max: float
    count: int
    mean_score: float
    observed_win_rate: float
    calibrated_win_rate: float


@dataclass(frozen=True, slots=True)
class EmpiricalForgeCalibrator:
    """Post-hoc score calibrator fit from Forge outcomes."""

    score_field: ScoreField
    source_case_count: int
    bins: list[EmpiricalCalibrationBin]

    def predict(self, score: float) -> float:
        """Return the calibrated Forge win-rate estimate for a score."""
        if not self.bins:
            return score
        if score <= self.bins[0].score_min:
            return self.bins[0].calibrated_win_rate
        for calibration_bin in self.bins:
            if calibration_bin.score_min <= score <= calibration_bin.score_max:
                return calibration_bin.calibrated_win_rate
        return self.bins[-1].calibrated_win_rate

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "score_field": self.score_field,
            "source_case_count": self.source_case_count,
            "bins": [asdict(calibration_bin) for calibration_bin in self.bins],
        }


def load_observations_from_artifacts(artifacts_dir: Path) -> list[CalibrationObservation]:
    """Load calibration observations from markdown and selection CSV artifacts."""
    actual_by_id = _load_actual_win_rates(artifacts_dir)
    observations: list[CalibrationObservation] = []
    for selection_path in artifacts_dir.rglob("*.selection.csv"):
        with selection_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                generated_deck_id = row["generated_deck_id"]
                actual_win_rate = actual_by_id.get(generated_deck_id)
                if actual_win_rate is None:
                    continue
                observations.append(
                    CalibrationObservation(
                        generated_deck_id=generated_deck_id,
                        score_band=int(row["score_band"]),
                        predicted_win_rate=float(row["predicted_win_rate"]),
                        selection_score=float(row["selection_score"]),
                        structure_penalty=float(row["structure_penalty"]),
                        actual_win_rate=actual_win_rate,
                    )
                )
    return observations


def fit_empirical_calibrator(
    observations: list[CalibrationObservation],
    *,
    score_field: ScoreField = "selection_score",
    bin_count: int = 5,
) -> EmpiricalForgeCalibrator:
    """Fit a monotonic empirical calibrator from Forge observations."""
    if bin_count <= 0:
        msg = f"bin_count must be positive, got {bin_count}"
        raise ValueError(msg)
    if not observations:
        msg = "at least one observation is required"
        raise ValueError(msg)

    sorted_observations = sorted(
        observations,
        key=lambda observation: (
            _score_for(observation, score_field),
            observation.generated_deck_id,
        ),
    )
    raw_bins: list[EmpiricalCalibrationBin] = []
    actual_bin_count = min(bin_count, len(sorted_observations))
    for index in range(actual_bin_count):
        start = round(index * len(sorted_observations) / actual_bin_count)
        end = round((index + 1) * len(sorted_observations) / actual_bin_count)
        bucket = sorted_observations[start:end]
        scores = [_score_for(observation, score_field) for observation in bucket]
        observed_win_rate = mean(observation.actual_win_rate for observation in bucket)
        raw_bins.append(
            EmpiricalCalibrationBin(
                score_min=min(scores),
                score_max=max(scores),
                count=len(bucket),
                mean_score=mean(scores),
                observed_win_rate=observed_win_rate,
                calibrated_win_rate=observed_win_rate,
            )
        )

    calibrated_rates = _monotonic_non_decreasing(
        [calibration_bin.observed_win_rate for calibration_bin in raw_bins],
        [calibration_bin.count for calibration_bin in raw_bins],
    )
    bins = [
        EmpiricalCalibrationBin(
            score_min=calibration_bin.score_min,
            score_max=calibration_bin.score_max,
            count=calibration_bin.count,
            mean_score=calibration_bin.mean_score,
            observed_win_rate=calibration_bin.observed_win_rate,
            calibrated_win_rate=calibrated_rate,
        )
        for calibration_bin, calibrated_rate in zip(raw_bins, calibrated_rates, strict=True)
    ]
    return EmpiricalForgeCalibrator(
        score_field=score_field,
        source_case_count=len(observations),
        bins=bins,
    )


def write_empirical_calibrator(calibrator: EmpiricalForgeCalibrator, output_path: Path) -> Path:
    """Write an empirical calibrator JSON artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(calibrator.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_empirical_calibrator(calibrator_path: Path) -> EmpiricalForgeCalibrator:
    """Load an empirical calibrator JSON artifact."""
    payload = json.loads(calibrator_path.read_text(encoding="utf-8"))
    score_field = payload["score_field"]
    if score_field not in {"predicted_win_rate", "selection_score"}:
        msg = f"Unsupported score_field in calibrator: {score_field}"
        raise ValueError(msg)
    bins = [
        EmpiricalCalibrationBin(
            score_min=float(item["score_min"]),
            score_max=float(item["score_max"]),
            count=int(item["count"]),
            mean_score=float(item["mean_score"]),
            observed_win_rate=float(item["observed_win_rate"]),
            calibrated_win_rate=float(item["calibrated_win_rate"]),
        )
        for item in payload["bins"]
    ]
    return EmpiricalForgeCalibrator(
        score_field=cast(ScoreField, score_field),
        source_case_count=int(payload["source_case_count"]),
        bins=bins,
    )


def _load_actual_win_rates(artifacts_dir: Path) -> dict[str, float]:
    actual_by_id: dict[str, float] = {}
    for report_path in artifacts_dir.rglob("*.md"):
        in_cases = False
        for line in report_path.read_text(encoding="utf-8").splitlines():
            if line.strip() == "## All Cases":
                in_cases = True
                continue
            if in_cases and line.startswith("## "):
                break
            if in_cases and line.startswith("| `"):
                parts = [part.strip() for part in line.strip("|").split("|")]
                actual_by_id[parts[0].strip("`")] = float(parts[2])
    return actual_by_id


def _score_for(observation: CalibrationObservation, score_field: ScoreField) -> float:
    if score_field == "predicted_win_rate":
        return observation.predicted_win_rate
    return observation.selection_score


def _monotonic_non_decreasing(values: list[float], weights: list[int]) -> list[float]:
    blocks = [
        {"total": value * weight, "weight": float(weight), "count": 1}
        for value, weight in zip(values, weights, strict=True)
    ]
    index = 0
    while index < len(blocks) - 1:
        current = blocks[index]["total"] / blocks[index]["weight"]
        next_value = blocks[index + 1]["total"] / blocks[index + 1]["weight"]
        if current <= next_value:
            index += 1
            continue
        blocks[index]["total"] += blocks[index + 1]["total"]
        blocks[index]["weight"] += blocks[index + 1]["weight"]
        blocks[index]["count"] += blocks[index + 1]["count"]
        del blocks[index + 1]
        if index > 0:
            index -= 1

    calibrated: list[float] = []
    for block in blocks:
        block_average = block["total"] / block["weight"]
        calibrated.extend([block_average] * int(block["count"]))
    return calibrated
