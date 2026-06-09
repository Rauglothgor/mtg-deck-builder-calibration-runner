from uuid import uuid4

import pytest

from deckbuilder.experiment.orchestrator import (
    CandidateDeck,
    _select_score_band_candidates,
)


def test_select_score_band_candidates_samples_each_band() -> None:
    candidates = [
        CandidateDeck(
            seed=seed,
            card_oracle_ids=[uuid4()],
            predicted_win_rate=score / 100,
        )
        for seed, score in enumerate(range(10, 110, 10), start=1)
    ]

    selected = _select_score_band_candidates(candidates, n_decks=5, band_count=5)

    assert [item.score_band for item in selected] == [0, 1, 2, 3, 4]
    assert [item.predicted_win_rate for item in selected] == pytest.approx(
        [0.2, 0.4, 0.6, 0.8, 1.0]
    )
    assert selected[0].band_min_score == pytest.approx(0.1)
    assert selected[0].band_max_score == pytest.approx(0.2)


def test_select_score_band_candidates_rejects_too_small_pool() -> None:
    candidates = [
        CandidateDeck(seed=1, card_oracle_ids=[uuid4()], predicted_win_rate=0.5),
    ]

    with pytest.raises(RuntimeError, match="Need at least 2 candidates"):
        _select_score_band_candidates(candidates, n_decks=2, band_count=5)


def test_select_score_band_candidates_ranks_by_selection_score_when_present() -> None:
    low_raw_clean = CandidateDeck(
        seed=1,
        card_oracle_ids=[uuid4()],
        predicted_win_rate=0.6,
        selection_score=0.6,
    )
    high_raw_penalized = CandidateDeck(
        seed=2,
        card_oracle_ids=[uuid4()],
        predicted_win_rate=1.0,
        selection_score=0.35,
        structure_penalty=0.65,
    )

    selected = _select_score_band_candidates(
        [low_raw_clean, high_raw_penalized],
        n_decks=2,
        band_count=2,
    )

    assert [item.seed for item in selected] == [2, 1]
    assert [item.predicted_win_rate for item in selected] == [1.0, 0.6]
    assert [item.selection_score for item in selected] == [0.35, 0.6]
    assert selected[0].structure_penalty == 0.65
