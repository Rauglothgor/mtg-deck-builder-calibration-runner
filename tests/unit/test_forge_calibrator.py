from pathlib import Path

import pytest

from deckbuilder.experiment.forge_calibrator import (
    CalibrationObservation,
    fit_empirical_calibrator,
    load_empirical_calibrator,
    load_observations_from_artifacts,
    write_empirical_calibrator,
)


def test_load_observations_from_artifacts_pairs_report_and_selection_csv(tmp_path: Path) -> None:
    report_path = tmp_path / "v0_5_calibration_shard_0.md"
    report_path.write_text(
        "\n".join(
            [
                "# Calibration Report",
                "## All Cases",
                "| Deck ID | Predicted | Actual | Bias | Deviation |",
                "|---|---:|---:|---:|---:|",
                "| `deck-a` | 0.100 | 0.500 | -0.400 | 0.400 |",
                "| `deck-b` | 0.900 | 0.600 | 0.300 | 0.300 |",
            ]
        ),
        encoding="utf-8",
    )
    selection_path = tmp_path / "v0_5_calibration_shard_0.selection.csv"
    selection_path.write_text(
        "\n".join(
            [
                "generated_deck_id,seed,score_band,band_min_score,band_max_score,predicted_win_rate,selection_score,structure_penalty",
                "deck-a,1,0,0.0,0.2,0.1,0.1,0.0",
                "deck-b,2,1,0.8,1.0,0.9,0.7,0.2",
            ]
        ),
        encoding="utf-8",
    )

    observations = load_observations_from_artifacts(tmp_path)

    assert observations == [
        CalibrationObservation(
            generated_deck_id="deck-a",
            score_band=0,
            predicted_win_rate=0.1,
            selection_score=0.1,
            structure_penalty=0.0,
            actual_win_rate=0.5,
        ),
        CalibrationObservation(
            generated_deck_id="deck-b",
            score_band=1,
            predicted_win_rate=0.9,
            selection_score=0.7,
            structure_penalty=0.2,
            actual_win_rate=0.6,
        ),
    ]


def test_fit_empirical_calibrator_pools_non_monotonic_bins() -> None:
    observations = [
        CalibrationObservation("a", 0, 0.1, 0.1, 0.0, 0.50),
        CalibrationObservation("b", 1, 0.4, 0.4, 0.0, 0.70),
        CalibrationObservation("c", 2, 0.7, 0.7, 0.0, 0.60),
        CalibrationObservation("d", 3, 0.9, 0.9, 0.0, 0.80),
    ]

    calibrator = fit_empirical_calibrator(observations, bin_count=4)

    assert calibrator.source_case_count == 4
    assert [bin.calibrated_win_rate for bin in calibrator.bins] == pytest.approx(
        [0.50, 0.65, 0.65, 0.80]
    )
    assert calibrator.predict(0.05) == pytest.approx(0.50)
    assert calibrator.predict(0.7) == pytest.approx(0.65)
    assert calibrator.predict(1.0) == pytest.approx(0.80)


def test_fit_empirical_calibrator_rejects_empty_observations() -> None:
    with pytest.raises(ValueError, match="at least one observation"):
        fit_empirical_calibrator([])


def test_write_and_load_empirical_calibrator_round_trips(tmp_path: Path) -> None:
    observations = [
        CalibrationObservation("a", 0, 0.1, 0.1, 0.0, 0.50),
        CalibrationObservation("b", 1, 0.9, 0.7, 0.2, 0.60),
    ]
    calibrator = fit_empirical_calibrator(observations, bin_count=2)
    output_path = tmp_path / "calibrator.json"

    write_empirical_calibrator(calibrator, output_path)
    loaded = load_empirical_calibrator(output_path)

    assert loaded == calibrator
    assert loaded.predict(0.7) == pytest.approx(0.60)
