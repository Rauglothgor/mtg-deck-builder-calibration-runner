"""Calibration metrics for experiment evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Aggregate calibration metrics for predicted-vs-actual win-rate pairs."""

    pair_count: int
    mean_absolute_deviation: float
    max_deviation: float
    mean_calibration_bias: float
    overconfidence_rate_20: float
    overconfidence_rate_30: float
    brier_score: float
    adversarial_rate: float
    decision: Literal["proceed", "frequent_validation", "pivot"]


def mean_absolute_deviation(pairs: list[tuple[float, float]]) -> float:
    """Return the mean absolute deviation between predicted and actual win rates."""
    if not pairs:
        return 0.0
    return sum(abs(predicted - actual) for predicted, actual in pairs) / len(pairs)


def max_deviation(pairs: list[tuple[float, float]]) -> float:
    """Return the maximum absolute deviation between predicted and actual win rates."""
    if not pairs:
        return 0.0
    return max(abs(predicted - actual) for predicted, actual in pairs)


def mean_calibration_bias(pairs: list[tuple[float, float]]) -> float:
    """Return mean signed error, where positive values mean overconfidence."""
    if not pairs:
        return 0.0
    return sum(predicted - actual for predicted, actual in pairs) / len(pairs)


def overconfidence_rate(pairs: list[tuple[float, float]], threshold: float = 0.20) -> float:
    """Return the fraction of cases where prediction exceeds actual by threshold."""
    if not pairs:
        return 0.0
    overconfident_count = sum(1 for predicted, actual in pairs if predicted - actual > threshold)
    return overconfident_count / len(pairs)


def brier_score(pairs: list[tuple[float, float]]) -> float:
    """Return the mean squared probability error against observed win rates."""
    if not pairs:
        return 0.0
    return sum((predicted - actual) ** 2 for predicted, actual in pairs) / len(pairs)


def adversarial_rate(pairs: list[tuple[float, float]]) -> float:
    """Return the fraction of adversarial cases in the calibration sample."""
    if not pairs:
        return 0.0
    adversarial_count = sum(1 for predicted, actual in pairs if predicted >= 0.70 and actual < 0.35)
    return adversarial_count / len(pairs)


def calibration_decision(
    adversarial_fraction: float,
    mean_abs_deviation: float = 0.0,
    overconfidence_fraction_20: float = 0.0,
) -> Literal["proceed", "frequent_validation", "pivot"]:
    """Map calibration metrics to the v0.5 decision recommendation."""
    if (
        adversarial_fraction > 0.30
        or mean_abs_deviation > 0.25
        or overconfidence_fraction_20 > 0.50
    ):
        return "pivot"
    if (
        adversarial_fraction >= 0.10
        or mean_abs_deviation > 0.10
        or overconfidence_fraction_20 > 0.20
    ):
        return "frequent_validation"
    return "proceed"


def compute_calibration(pairs: list[tuple[float, float]]) -> CalibrationReport:
    """Compute the full calibration report for predicted-vs-actual pairs."""
    mad = mean_absolute_deviation(pairs)
    max_dev = max_deviation(pairs)
    bias = mean_calibration_bias(pairs)
    overconf_20 = overconfidence_rate(pairs, threshold=0.20)
    overconf_30 = overconfidence_rate(pairs, threshold=0.30)
    brier = brier_score(pairs)
    adv_rate = adversarial_rate(pairs)
    return CalibrationReport(
        pair_count=len(pairs),
        mean_absolute_deviation=mad,
        max_deviation=max_dev,
        mean_calibration_bias=bias,
        overconfidence_rate_20=overconf_20,
        overconfidence_rate_30=overconf_30,
        brier_score=brier,
        adversarial_rate=adv_rate,
        decision=calibration_decision(adv_rate, mad, overconf_20),
    )
