from pathlib import Path
from types import SimpleNamespace

import pytest

from deckbuilder.experiment.forge_calibrator import (
    OUTCOME_FEATURE_NAMES,
    CalibrationObservation,
    EmpiricalCalibrationBin,
    EmpiricalForgeCalibrator,
    ForgeOutcomeObservation,
    evaluate_outcome_model,
    fit_empirical_calibrator,
    fit_forge_outcome_model,
    load_empirical_calibrator,
    load_forge_outcome_model,
    load_observations_from_artifacts,
    load_outcome_observations_from_artifacts,
    outcome_features_from_diagnostics,
    write_empirical_calibrator,
    write_forge_outcome_model,
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


def test_load_outcome_observations_requires_report_selection_and_structure(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "v0_5_calibration_shard_0.md"
    report_path.write_text(
        "\n".join(
            [
                "# Calibration Report",
                "## All Cases",
                "| Deck ID | Predicted | Actual | Bias | Deviation |",
                "|---|---:|---:|---:|---:|",
                "| `deck-a` | 0.700 | 0.550 | 0.150 | 0.150 |",
            ]
        ),
        encoding="utf-8",
    )
    selection_path = tmp_path / "v0_5_calibration_shard_0.selection.csv"
    selection_path.write_text(
        "\n".join(
            [
                "generated_deck_id,seed,score_band,band_min_score,band_max_score,predicted_win_rate,selection_score,structure_penalty",
                "deck-a,1,0,0.0,1.0,0.7,0.6,0.1",
            ]
        ),
        encoding="utf-8",
    )
    structure_path = tmp_path / "v0_5_calibration_shard_0.structure.csv"
    structure_path.write_text(
        "\n".join(
            [
                "generated_deck_id,seed,predicted_win_rate,card_count,land_count,nonland_count,ramp_count,card_draw_count,removal_count,board_wipe_count,win_condition_count,average_nonland_cmc,median_nonland_cmc,low_curve_nonland_count,high_curve_nonland_count,expected_compounded_mana_spent",
                "deck-a,1,0.7,99,38,61,12,14,10,2,6,2.8,3.0,26,7,68.0",
            ]
        ),
        encoding="utf-8",
    )

    observations = load_outcome_observations_from_artifacts(tmp_path)

    assert len(observations) == 1
    observation = observations[0]
    assert observation.generated_deck_id == "deck-a"
    assert observation.predicted_win_rate == 0.7
    assert observation.selection_score == pytest.approx(0.66)
    assert observation.structure_penalty == 0.1
    assert observation.actual_win_rate == 0.55
    assert observation.features == {
        "predicted_win_rate": 0.7,
        "selection_score": pytest.approx(0.66),
        "structure_penalty": 0.1,
        "land_count": 38.0,
        "ramp_count": 12.0,
        "card_draw_count": 14.0,
        "removal_count": 10.0,
        "board_wipe_count": 2.0,
        "win_condition_count": 6.0,
        "average_nonland_cmc": 2.8,
        "median_nonland_cmc": 3.0,
        "low_curve_nonland_count": 26.0,
        "high_curve_nonland_count": 7.0,
        "expected_compounded_mana_spent": 68.0,
    }


def test_forge_outcome_model_fits_residual_over_empirical_guardrail(
    tmp_path: Path,
) -> None:
    base_calibrator = EmpiricalForgeCalibrator(
        score_field="selection_score",
        source_case_count=2,
        bins=[
            EmpiricalCalibrationBin(
                score_min=0.0,
                score_max=1.0,
                count=2,
                mean_score=0.5,
                observed_win_rate=0.55,
                calibrated_win_rate=0.55,
            )
        ],
    )
    observations = [
        _outcome_observation("a", selection_score=0.3, actual_win_rate=0.45, ramp_count=8.0),
        _outcome_observation("b", selection_score=0.7, actual_win_rate=0.65, ramp_count=14.0),
    ]

    model = fit_forge_outcome_model(
        observations,
        base_calibrator=base_calibrator,
        l2_regularization=1.0,
    )
    output_path = tmp_path / "outcome-model.json"

    write_forge_outcome_model(model, output_path)
    loaded = load_forge_outcome_model(output_path)
    evaluation = evaluate_outcome_model(loaded, observations)

    assert loaded.base_calibrator == base_calibrator
    assert loaded.source_case_count == 2
    assert evaluation.mean_absolute_deviation < 0.10
    assert loaded.predict_observation(observations[1]) > loaded.predict_observation(observations[0])


def test_outcome_features_from_diagnostics_extracts_model_feature_row() -> None:
    diagnostics = SimpleNamespace(
        land_count=38,
        ramp_count=12,
        card_draw_count=14,
        removal_count=10,
        board_wipe_count=2,
        win_condition_count=6,
        average_nonland_cmc=2.8,
        median_nonland_cmc=3.0,
        low_curve_nonland_count=26,
        high_curve_nonland_count=7,
        expected_compounded_mana_spent=68.0,
    )

    features = outcome_features_from_diagnostics(
        predicted_win_rate=0.8,
        selection_score=0.7,
        structure_penalty=0.1,
        diagnostics=diagnostics,
    )

    assert set(features) == set(OUTCOME_FEATURE_NAMES)
    assert features["predicted_win_rate"] == 0.8
    assert features["selection_score"] == 0.7
    assert features["expected_compounded_mana_spent"] == 68.0


def _outcome_observation(
    generated_deck_id: str,
    *,
    selection_score: float,
    actual_win_rate: float,
    ramp_count: float,
) -> ForgeOutcomeObservation:
    features = {
        "predicted_win_rate": selection_score,
        "selection_score": selection_score,
        "structure_penalty": 0.0,
        "land_count": 38.0,
        "ramp_count": ramp_count,
        "card_draw_count": 14.0,
        "removal_count": 10.0,
        "board_wipe_count": 2.0,
        "win_condition_count": 6.0,
        "average_nonland_cmc": 2.8,
        "median_nonland_cmc": 3.0,
        "low_curve_nonland_count": 26.0,
        "high_curve_nonland_count": 7.0,
        "expected_compounded_mana_spent": 68.0,
    }
    return ForgeOutcomeObservation(
        generated_deck_id=generated_deck_id,
        predicted_win_rate=selection_score,
        selection_score=selection_score,
        structure_penalty=0.0,
        actual_win_rate=actual_win_rate,
        features=features,
    )
