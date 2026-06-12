"""Fit empirical calibrators from Forge validation artifacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Literal, cast

ScoreField = Literal["predicted_win_rate", "selection_score"]

STRUCTURE_FEATURE_NAMES: tuple[str, ...] = (
    "land_count",
    "ramp_count",
    "card_draw_count",
    "removal_count",
    "board_wipe_count",
    "win_condition_count",
    "average_nonland_cmc",
    "median_nonland_cmc",
    "low_curve_nonland_count",
    "high_curve_nonland_count",
    "expected_compounded_mana_spent",
)

OUTCOME_FEATURE_NAMES: tuple[str, ...] = (
    "predicted_win_rate",
    "selection_score",
    "structure_penalty",
    *STRUCTURE_FEATURE_NAMES,
)


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
class ForgeOutcomeObservation:
    """One generated deck with model features and Forge-observed win rate."""

    generated_deck_id: str
    predicted_win_rate: float
    selection_score: float
    structure_penalty: float
    actual_win_rate: float
    features: dict[str, float]


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


@dataclass(frozen=True, slots=True)
class ForgeOutcomeModel:
    """Residual Forge-outcome model layered on top of a calibrated base score."""

    source_case_count: int
    feature_names: list[str]
    feature_means: dict[str, float]
    feature_scales: dict[str, float]
    coefficients: dict[str, float]
    intercept: float
    base_score_field: ScoreField
    base_calibrator: EmpiricalForgeCalibrator | None
    l2_regularization: float
    training_mad: float
    training_bias: float

    def predict_features(self, features: Mapping[str, float]) -> float:
        """Predict Forge win rate from a complete feature row."""
        base = _base_prediction(
            features,
            score_field=self.base_score_field,
            calibrator=self.base_calibrator,
        )
        residual = self.intercept
        for feature_name in self.feature_names:
            value = features[feature_name]
            centered = value - self.feature_means[feature_name]
            scaled = centered / self.feature_scales[feature_name]
            residual += self.coefficients[feature_name] * scaled
        return min(1.0, max(0.0, base + residual))

    def predict_observation(self, observation: ForgeOutcomeObservation) -> float:
        """Predict Forge win rate for a loaded observation."""
        return self.predict_features(observation.features)

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "source_case_count": self.source_case_count,
            "feature_names": self.feature_names,
            "feature_means": self.feature_means,
            "feature_scales": self.feature_scales,
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "base_score_field": self.base_score_field,
            "base_calibrator": (
                self.base_calibrator.to_json_dict() if self.base_calibrator is not None else None
            ),
            "l2_regularization": self.l2_regularization,
            "training_mad": self.training_mad,
            "training_bias": self.training_bias,
        }


@dataclass(frozen=True, slots=True)
class OutcomeModelEvaluation:
    """Compact quality metrics for a Forge-outcome model."""

    case_count: int
    mean_prediction: float
    mean_actual: float
    mean_absolute_deviation: float
    bias: float
    overconfidence_rate_20: float
    overconfidence_rate_30: float
    underconfidence_rate_20: float
    underconfidence_rate_30: float
    pearson: float
    spearman: float


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


def load_outcome_observations_from_artifacts(artifacts_dir: Path) -> list[ForgeOutcomeObservation]:
    """Load complete Forge-outcome model rows from report, selection, and structure artifacts."""
    actual_by_id = _load_actual_win_rates(artifacts_dir)
    selection_by_id = _load_selection_rows(artifacts_dir)
    structure_by_id = _load_structure_rows(artifacts_dir)

    observations: list[ForgeOutcomeObservation] = []
    for generated_deck_id, selection_row in selection_by_id.items():
        actual_win_rate = actual_by_id.get(generated_deck_id)
        structure_row = structure_by_id.get(generated_deck_id)
        if actual_win_rate is None or structure_row is None:
            continue
        structural_selection_score = _structural_adjusted_score_from_penalty(
            selection_row["predicted_win_rate"],
            selection_row["structure_penalty"],
        )
        features = {
            "predicted_win_rate": selection_row["predicted_win_rate"],
            "selection_score": structural_selection_score,
            "structure_penalty": selection_row["structure_penalty"],
            **{name: structure_row[name] for name in STRUCTURE_FEATURE_NAMES},
        }
        observations.append(
            ForgeOutcomeObservation(
                generated_deck_id=generated_deck_id,
                predicted_win_rate=selection_row["predicted_win_rate"],
                selection_score=structural_selection_score,
                structure_penalty=selection_row["structure_penalty"],
                actual_win_rate=actual_win_rate,
                features=features,
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


def fit_forge_outcome_model(
    observations: list[ForgeOutcomeObservation],
    *,
    base_score_field: ScoreField = "selection_score",
    base_calibrator: EmpiricalForgeCalibrator | None = None,
    feature_names: tuple[str, ...] = OUTCOME_FEATURE_NAMES,
    l2_regularization: float = 5.0,
) -> ForgeOutcomeModel:
    """Fit a ridge residual model for Forge outcomes."""
    if not observations:
        msg = "at least one observation is required"
        raise ValueError(msg)
    if l2_regularization < 0:
        msg = f"l2_regularization must be non-negative, got {l2_regularization}"
        raise ValueError(msg)

    features = list(feature_names)
    rows = [[observation.features[name] for name in features] for observation in observations]
    feature_means = {
        feature_name: mean(row[index] for row in rows)
        for index, feature_name in enumerate(features)
    }
    feature_scales: dict[str, float] = {}
    for index, feature_name in enumerate(features):
        variance = mean((row[index] - feature_means[feature_name]) ** 2 for row in rows)
        feature_scales[feature_name] = variance**0.5 or 1.0

    design_rows = [
        [
            (row[index] - feature_means[feature_name]) / feature_scales[feature_name]
            for index, feature_name in enumerate(features)
        ]
        for row in rows
    ]
    targets = [
        observation.actual_win_rate
        - _base_prediction(
            observation.features,
            score_field=base_score_field,
            calibrator=base_calibrator,
        )
        for observation in observations
    ]
    intercept, coefficients = _fit_ridge(design_rows, targets, l2_regularization)
    coefficient_by_feature = dict(zip(features, coefficients, strict=True))

    provisional = ForgeOutcomeModel(
        source_case_count=len(observations),
        feature_names=features,
        feature_means=feature_means,
        feature_scales=feature_scales,
        coefficients=coefficient_by_feature,
        intercept=intercept,
        base_score_field=base_score_field,
        base_calibrator=base_calibrator,
        l2_regularization=l2_regularization,
        training_mad=0.0,
        training_bias=0.0,
    )
    predictions = [provisional.predict_observation(observation) for observation in observations]
    diffs = [
        prediction - observation.actual_win_rate
        for prediction, observation in zip(predictions, observations, strict=True)
    ]
    return ForgeOutcomeModel(
        source_case_count=provisional.source_case_count,
        feature_names=provisional.feature_names,
        feature_means=provisional.feature_means,
        feature_scales=provisional.feature_scales,
        coefficients=provisional.coefficients,
        intercept=provisional.intercept,
        base_score_field=provisional.base_score_field,
        base_calibrator=provisional.base_calibrator,
        l2_regularization=provisional.l2_regularization,
        training_mad=mean(abs(diff) for diff in diffs),
        training_bias=mean(diffs),
    )


def outcome_features_from_diagnostics(
    *,
    predicted_win_rate: float,
    selection_score: float,
    structure_penalty: float,
    diagnostics: object,
) -> dict[str, float]:
    """Return model features from candidate diagnostics."""
    return {
        "predicted_win_rate": predicted_win_rate,
        "selection_score": selection_score,
        "structure_penalty": structure_penalty,
        **{name: float(getattr(diagnostics, name)) for name in STRUCTURE_FEATURE_NAMES},
    }


def evaluate_outcome_model(
    model: ForgeOutcomeModel,
    observations: list[ForgeOutcomeObservation],
) -> OutcomeModelEvaluation:
    """Evaluate a Forge-outcome model against observations."""
    if not observations:
        msg = "at least one observation is required"
        raise ValueError(msg)
    predictions = [model.predict_observation(observation) for observation in observations]
    actuals = [observation.actual_win_rate for observation in observations]
    diffs = [prediction - actual for prediction, actual in zip(predictions, actuals, strict=True)]
    return OutcomeModelEvaluation(
        case_count=len(observations),
        mean_prediction=mean(predictions),
        mean_actual=mean(actuals),
        mean_absolute_deviation=mean(abs(diff) for diff in diffs),
        bias=mean(diffs),
        overconfidence_rate_20=sum(diff > 0.20 for diff in diffs) / len(diffs),
        overconfidence_rate_30=sum(diff > 0.30 for diff in diffs) / len(diffs),
        underconfidence_rate_20=sum(diff < -0.20 for diff in diffs) / len(diffs),
        underconfidence_rate_30=sum(diff < -0.30 for diff in diffs) / len(diffs),
        pearson=_pearson(predictions, actuals),
        spearman=_spearman(predictions, actuals),
    )


def write_empirical_calibrator(calibrator: EmpiricalForgeCalibrator, output_path: Path) -> Path:
    """Write an empirical calibrator JSON artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(calibrator.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_forge_outcome_model(model: ForgeOutcomeModel, output_path: Path) -> Path:
    """Write a Forge-outcome model JSON artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(model.to_json_dict(), indent=2, sort_keys=True) + "\n",
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


def load_forge_outcome_model(model_path: Path) -> ForgeOutcomeModel:
    """Load a Forge-outcome model JSON artifact."""
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    base_score_field = payload["base_score_field"]
    if base_score_field not in {"predicted_win_rate", "selection_score"}:
        msg = f"Unsupported base_score_field in outcome model: {base_score_field}"
        raise ValueError(msg)
    base_calibrator_payload = payload.get("base_calibrator")
    base_calibrator = (
        _empirical_calibrator_from_json_dict(cast(dict[str, object], base_calibrator_payload))
        if base_calibrator_payload is not None
        else None
    )
    feature_names = [str(item) for item in payload["feature_names"]]
    return ForgeOutcomeModel(
        source_case_count=int(payload["source_case_count"]),
        feature_names=feature_names,
        feature_means={str(key): float(value) for key, value in payload["feature_means"].items()},
        feature_scales={str(key): float(value) for key, value in payload["feature_scales"].items()},
        coefficients={str(key): float(value) for key, value in payload["coefficients"].items()},
        intercept=float(payload["intercept"]),
        base_score_field=cast(ScoreField, base_score_field),
        base_calibrator=base_calibrator,
        l2_regularization=float(payload["l2_regularization"]),
        training_mad=float(payload["training_mad"]),
        training_bias=float(payload["training_bias"]),
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


def _load_selection_rows(artifacts_dir: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for selection_path in artifacts_dir.rglob("*.selection.csv"):
        with selection_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows[row["generated_deck_id"]] = {
                    "seed": float(row["seed"]),
                    "score_band": float(row["score_band"]),
                    "band_min_score": float(row["band_min_score"]),
                    "band_max_score": float(row["band_max_score"]),
                    "predicted_win_rate": float(row["predicted_win_rate"]),
                    "selection_score": float(row["selection_score"]),
                    "structure_penalty": float(row["structure_penalty"]),
                }
    return rows


def _load_structure_rows(artifacts_dir: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for structure_path in artifacts_dir.rglob("*.structure.csv"):
        with structure_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows[row["generated_deck_id"]] = {
                    key: float(value) for key, value in row.items() if key != "generated_deck_id"
                }
    return rows


def _empirical_calibrator_from_json_dict(payload: dict[str, object]) -> EmpiricalForgeCalibrator:
    score_field = payload["score_field"]
    if score_field not in {"predicted_win_rate", "selection_score"}:
        msg = f"Unsupported score_field in calibrator: {score_field}"
        raise ValueError(msg)
    bins = [
        EmpiricalCalibrationBin(
            score_min=_json_float(item["score_min"]),
            score_max=_json_float(item["score_max"]),
            count=_json_int(item["count"]),
            mean_score=_json_float(item["mean_score"]),
            observed_win_rate=_json_float(item["observed_win_rate"]),
            calibrated_win_rate=_json_float(item["calibrated_win_rate"]),
        )
        for item in cast(list[dict[str, object]], payload["bins"])
    ]
    return EmpiricalForgeCalibrator(
        score_field=cast(ScoreField, score_field),
        source_case_count=_json_int(payload["source_case_count"]),
        bins=bins,
    )


def _json_float(value: object) -> float:
    if isinstance(value, str | int | float):
        return float(value)
    msg = f"Expected JSON number-compatible value, got {type(value).__name__}"
    raise TypeError(msg)


def _json_int(value: object) -> int:
    if isinstance(value, str | int | float):
        return int(value)
    msg = f"Expected JSON integer-compatible value, got {type(value).__name__}"
    raise TypeError(msg)


def _score_for(observation: CalibrationObservation, score_field: ScoreField) -> float:
    if score_field == "predicted_win_rate":
        return observation.predicted_win_rate
    return observation.selection_score


def _base_prediction(
    features: Mapping[str, float],
    *,
    score_field: ScoreField,
    calibrator: EmpiricalForgeCalibrator | None,
) -> float:
    score = features[score_field]
    if calibrator is None:
        return score
    return calibrator.predict(score)


def _structural_adjusted_score_from_penalty(
    predicted_win_rate: float,
    structure_penalty: float,
) -> float:
    penalty_scale = min(1.0, max(0.0, (predicted_win_rate - 0.5) / 0.5))
    return min(1.0, max(0.0, predicted_win_rate - structure_penalty * penalty_scale))


def _fit_ridge(
    design_rows: list[list[float]],
    targets: list[float],
    l2_regularization: float,
) -> tuple[float, list[float]]:
    feature_count = len(design_rows[0])
    matrix_size = feature_count + 1
    gram = [[0.0 for _ in range(matrix_size)] for _ in range(matrix_size)]
    rhs = [0.0 for _ in range(matrix_size)]

    for row, target in zip(design_rows, targets, strict=True):
        augmented = [1.0, *row]
        for outer_index, outer_value in enumerate(augmented):
            rhs[outer_index] += outer_value * target
            for inner_index, inner_value in enumerate(augmented):
                gram[outer_index][inner_index] += outer_value * inner_value

    for index in range(1, matrix_size):
        gram[index][index] += l2_regularization

    solution = _solve_linear_system(gram, rhs)
    return solution[0], solution[1:]


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    size = len(rhs)
    augmented = [row[:] + [rhs_value] for row, rhs_value in zip(matrix, rhs, strict=True)]
    for pivot_index in range(size):
        pivot_row = max(
            range(pivot_index, size),
            key=lambda row_index: abs(augmented[row_index][pivot_index]),
        )
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            msg = "Could not solve ridge system; matrix is singular"
            raise RuntimeError(msg)
        augmented[pivot_index], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[pivot_index],
        )
        pivot = augmented[pivot_index][pivot_index]
        augmented[pivot_index] = [value / pivot for value in augmented[pivot_index]]
        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            augmented[row_index] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row_index],
                    augmented[pivot_index],
                    strict=True,
                )
            ]
    return [row[-1] for row in augmented]


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean = mean(left)
    right_mean = mean(right)
    left_deltas = [value - left_mean for value in left]
    right_deltas = [value - right_mean for value in right]
    denominator = (
        sum(value * value for value in left_deltas) * sum(value * value for value in right_deltas)
    ) ** 0.5
    if denominator == 0:
        return 0.0
    return float(
        sum(
            left_delta * right_delta
            for left_delta, right_delta in zip(left_deltas, right_deltas, strict=True)
        )
        / denominator
    )


def _spearman(left: list[float], right: list[float]) -> float:
    return _pearson(_ranks(left), _ranks(right))


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        rank = (index + end) / 2 + 1
        for order_index in range(index, end + 1):
            ranks[order[order_index]] = rank
        index = end + 1
    return ranks


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
