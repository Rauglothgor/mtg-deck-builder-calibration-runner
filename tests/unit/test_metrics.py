import pytest

from deckbuilder.experiment.metrics import (
    adversarial_rate,
    brier_score,
    calibration_decision,
    compute_calibration,
    max_deviation,
    mean_absolute_deviation,
    mean_calibration_bias,
    overconfidence_rate,
)


def test_metric_functions_on_synthetic_data() -> None:
    pairs = [(0.80, 0.20), (0.60, 0.50), (0.30, 0.40), (0.75, 0.80)]

    assert mean_absolute_deviation(pairs) == pytest.approx(0.2125)
    assert max_deviation(pairs) == pytest.approx(0.60)
    assert mean_calibration_bias(pairs) == pytest.approx(0.1375)
    assert overconfidence_rate(pairs, threshold=0.20) == 0.25
    assert brier_score(pairs) == pytest.approx(0.095625)
    assert adversarial_rate(pairs) == 0.25


def test_compute_calibration_returns_decision() -> None:
    pairs = [(0.90, 0.10), (0.85, 0.20), (0.40, 0.45)]
    report = compute_calibration(pairs)

    assert report.pair_count == 3
    assert report.mean_absolute_deviation == pytest.approx(0.5)
    assert report.max_deviation == pytest.approx(0.8)
    assert report.mean_calibration_bias == pytest.approx(0.4666666666666666)
    assert report.overconfidence_rate_20 == pytest.approx(2 / 3)
    assert report.overconfidence_rate_30 == pytest.approx(2 / 3)
    assert report.brier_score == pytest.approx(0.355)
    assert report.adversarial_rate == 2 / 3
    assert report.decision == "pivot"


def test_calibration_decision_thresholds() -> None:
    assert calibration_decision(0.05) == "proceed"
    assert calibration_decision(0.10) == "frequent_validation"
    assert calibration_decision(0.30) == "frequent_validation"
    assert calibration_decision(0.31) == "pivot"
    assert calibration_decision(0.00, mean_abs_deviation=0.11) == "frequent_validation"
    assert calibration_decision(0.00, mean_abs_deviation=0.26) == "pivot"
    assert calibration_decision(0.00, overconfidence_fraction_20=0.21) == "frequent_validation"
    assert calibration_decision(0.00, overconfidence_fraction_20=0.51) == "pivot"
