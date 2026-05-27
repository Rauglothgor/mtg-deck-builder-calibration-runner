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


def adversarial_rate(pairs: list[tuple[float, float]]) -> float:
    """Return the fraction of adversarial cases in the calibration sample."""
    if not pairs:
        return 0.0
    adversarial_count = sum(1 for predicted, actual in pairs if predicted >= 0.70 and actual < 0.35)
    return adversarial_count / len(pairs)


def calibration_decision(
    adversarial_fraction: float,
) -> Literal["proceed", "frequent_validation", "pivot"]:
    """Map adversarial rate to the PRD decision recommendation."""
    if adversarial_fraction < 0.10:
        return "proceed"
    if adversarial_fraction <= 0.30:
        return "frequent_validation"
    return "pivot"


def compute_calibration(pairs: list[tuple[float, float]]) -> CalibrationReport:
    """Compute the full calibration report for predicted-vs-actual pairs."""
    mad = mean_absolute_deviation(pairs)
    max_dev = max_deviation(pairs)
    adv_rate = adversarial_rate(pairs)
    return CalibrationReport(
        pair_count=len(pairs),
        mean_absolute_deviation=mad,
        max_deviation=max_dev,
        adversarial_rate=adv_rate,
        decision=calibration_decision(adv_rate),
    )
