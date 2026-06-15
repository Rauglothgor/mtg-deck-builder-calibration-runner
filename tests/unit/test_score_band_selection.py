from pathlib import Path
from uuid import UUID, uuid4

import pytest
from _pytest.monkeypatch import MonkeyPatch

from deckbuilder.experiment.orchestrator import (
    CandidateDeck,
    _select_score_band_candidates,
    _selection_pool_after_optional_rerank,
    _simulation_rerank_candidates,
    _simulation_rerank_score,
    _theme_match_boost_from_texts,
    _theme_tags_for_preset,
    _write_score_band_manifest,
    _write_sim_rerank_manifest,
)
from deckbuilder.forge.parser import SimResult


def test_theme_tags_for_preset_merges_and_deduplicates() -> None:
    lane, tags = _theme_tags_for_preset(
        "poison",
        "Proliferate, draw",
    )

    assert lane == "poison"
    assert tags == ("poison", "toxic", "infect", "proliferate", "draw")


def test_theme_match_boost_is_bounded() -> None:
    searchable_cards = [
        "evolution sage creature landfall proliferate",
        "ichor rats creature poison counter",
        "sol ring artifact ramp",
    ]

    assert _theme_match_boost_from_texts(
        searchable_cards,
        ("proliferate", "poison", "ramp", "counter", "missing", "artifact", "creature"),
    ) == pytest.approx(0.03)
    assert _theme_match_boost_from_texts(searchable_cards, ("missing",)) == 0.0


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


def test_simulation_rerank_score_blends_prior_and_observation() -> None:
    assert _simulation_rerank_score(
        prior_score=0.80,
        wins=0,
        draws=0,
        matches_played=2,
        prior_weight=2.0,
    ) == pytest.approx(0.40)
    assert _simulation_rerank_score(
        prior_score=0.80,
        wins=0,
        draws=0,
        matches_played=0,
        prior_weight=2.0,
    ) == pytest.approx(0.80)


def test_simulation_rerank_candidates_updates_shortlist_scores(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    opponent_path = tmp_path / "alela.dck"
    opponent_path.write_text("[metadata]\nName=Alela\n", encoding="utf-8")
    fake_forge = _FakeForgeRunner(
        [
            SimResult(
                wins=0,
                losses=1,
                draws=0,
                game_durations_ms=(1000,),
                game_winners=("Ai(2)-opponent",),
                raw_output="fake",
            ),
            SimResult(
                wins=0,
                losses=1,
                draws=0,
                game_durations_ms=(1000,),
                game_winners=("Ai(2)-opponent",),
                raw_output="fake",
            ),
            SimResult(
                wins=1,
                losses=0,
                draws=0,
                game_durations_ms=(1000,),
                game_winners=("Ai(1)-candidate",),
                raw_output="fake",
            ),
            SimResult(
                wins=1,
                losses=0,
                draws=0,
                game_durations_ms=(1000,),
                game_winners=("Ai(1)-candidate",),
                raw_output="fake",
            ),
        ]
    )

    def fake_to_dck_format(
        _commander_oracle_id: UUID,
        _card_oracle_ids: list[UUID],
        output_path: str | Path,
    ) -> Path:
        path = Path(output_path)
        path.write_text("[metadata]\nName=Mock\n", encoding="utf-8")
        return path

    monkeypatch.setattr("deckbuilder.experiment.orchestrator.run_sim", fake_forge)
    monkeypatch.setattr("deckbuilder.experiment.orchestrator.to_dck_format", fake_to_dck_format)
    candidates = [
        CandidateDeck(
            seed=1,
            card_oracle_ids=[uuid4()],
            predicted_win_rate=1.0,
            selection_score=0.80,
        ),
        CandidateDeck(
            seed=2,
            card_oracle_ids=[uuid4()],
            predicted_win_rate=0.95,
            selection_score=0.70,
        ),
        CandidateDeck(
            seed=3,
            card_oracle_ids=[uuid4()],
            predicted_win_rate=0.20,
            selection_score=0.20,
        ),
    ]

    outcome = _simulation_rerank_candidates(
        candidates=candidates,
        commander_oracle_id=uuid4(),
        opponent_path=opponent_path,
        tmp_root=tmp_path,
        shortlist_size=2,
        matches=2,
        prior_weight=2.0,
    )

    by_seed = {candidate.seed: candidate for candidate in outcome.candidates}
    assert by_seed[1].pre_rerank_selection_score == pytest.approx(0.80)
    assert by_seed[1].selection_score == pytest.approx(0.40)
    assert by_seed[1].rerank_sim_win_rate == pytest.approx(0.0)
    assert by_seed[2].selection_score == pytest.approx(0.85)
    assert by_seed[2].rerank_wins == 2
    assert by_seed[3].selection_score == pytest.approx(0.20)
    assert by_seed[3].rerank_matches_played == 0
    assert outcome.retry_count == 0
    assert len(outcome.manifest_rows) == 2
    assert [row["seed"] for row in outcome.manifest_rows] == [1, 2]
    assert [call[2] for call in fake_forge.calls] == [1, 1, 1, 1]


def test_selection_pool_after_optional_rerank_uses_attempted_shortlist() -> None:
    reranked = CandidateDeck(
        seed=1,
        card_oracle_ids=[uuid4()],
        predicted_win_rate=1.0,
        selection_score=0.55,
        pre_rerank_selection_score=0.80,
        rerank_score=0.55,
    )
    not_reranked = CandidateDeck(
        seed=2,
        card_oracle_ids=[uuid4()],
        predicted_win_rate=0.9,
        selection_score=0.75,
    )

    assert _selection_pool_after_optional_rerank([not_reranked, reranked]) == [reranked]
    assert _selection_pool_after_optional_rerank([not_reranked]) == [not_reranked]


def test_score_band_manifests_include_preset_metadata(tmp_path: Path) -> None:
    report_path = tmp_path / "report.md"
    selection_path = _write_score_band_manifest(
        report_path,
        [
            {
                "generated_deck_id": "deck-1",
                "seed": 1,
                "score_band": 0,
                "band_min_score": 0.5,
                "band_max_score": 0.6,
                "predicted_win_rate": 0.9,
                "selection_score": 0.55,
                "structure_penalty": 0.2,
                "preset_lane": "poison",
                "theme_tags": "poison|toxic|infect|proliferate",
                "theme_boost": 0.02,
                "pre_rerank_selection_score": 0.6,
                "rerank_matches_played": 20,
                "rerank_wins": 11,
                "rerank_losses": 9,
                "rerank_draws": 0,
                "rerank_sim_win_rate": 0.55,
                "rerank_score": 0.58,
            },
        ],
    )
    rerank_path = _write_sim_rerank_manifest(
        report_path,
        [
            {
                "seed": 1,
                "pre_rerank_rank": 1,
                "predicted_win_rate": 0.9,
                "model_selection_score": 0.6,
                "structure_penalty": 0.2,
                "preset_lane": "poison",
                "theme_tags": "poison|toxic|infect|proliferate",
                "theme_boost": 0.02,
                "rerank_matches_requested": 20,
                "rerank_matches_played": 20,
                "rerank_wins": 11,
                "rerank_losses": 9,
                "rerank_draws": 0,
                "rerank_sim_win_rate": 0.55,
                "rerank_prior_weight": 30,
                "rerank_score": 0.58,
                "rerank_failures": 0,
                "selected": True,
                "generated_deck_id": "deck-1",
                "score_band": 0,
                "forge_ai_profile": "forge-daily-snapshot",
                "forge_build_id": "2.0.13-SNAPSHOT-06.11-2026-06-11T19:12:03Z",
            },
        ],
    )

    assert "preset_lane,theme_tags,theme_boost" in selection_path.read_text(encoding="utf-8")
    assert rerank_path is not None
    assert "preset_lane,theme_tags,theme_boost" in rerank_path.read_text(encoding="utf-8")


class _FakeForgeRunner:
    def __init__(self, outcomes: list[SimResult]) -> None:
        self._outcomes = iter(outcomes)
        self.calls: list[tuple[Path, Path, int, int]] = []

    def __call__(
        self,
        deck_path: str | Path,
        opponent_path: str | Path,
        n_matches: int,
        seed: int,
    ) -> SimResult:
        self.calls.append((Path(deck_path), Path(opponent_path), n_matches, seed))
        return next(self._outcomes)
