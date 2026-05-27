import pytest

from deckbuilder.experiment.metrics import (
    adversarial_rate,
    calibration_decision,
    compute_calibration,
    max_deviation,
    mean_absolute_deviation,
)


def test_metric_functions_on_synthetic_data() -> None:
    pairs = [(0.80, 0.20), (0.60, 0.50), (0.30, 0.40), (0.75, 0.80)]

    assert mean_absolute_deviation(pairs) == pytest.approx(0.2125)
    assert max_deviation(pairs) == pytest.approx(0.60)
    assert adversarial_rate(pairs) == 0.25


def test_compute_calibration_returns_decision() -> None:
    pairs = [(0.90, 0.10), (0.85, 0.20), (0.40, 0.45)]
    report = compute_calibration(pairs)

    assert report.pair_count == 3
    assert report.mean_absolute_deviation == pytest.approx(0.5)
    assert report.max_deviation == pytest.approx(0.8)
    assert report.adversarial_rate == 2 / 3
    assert report.decision == "pivot"


def test_calibration_decision_thresholds() -> None:
    assert calibration_decision(0.05) == "proceed"
    assert calibration_decision(0.10) == "frequent_validation"
    assert calibration_decision(0.30) == "frequent_validation"
    assert calibration_decision(0.31) == "pivot"
