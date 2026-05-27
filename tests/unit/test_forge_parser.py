from pathlib import Path

import pytest

from deckbuilder.forge.parser import parse_sim_output, parse_smoke_test_output

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_parse_smoke_test_output_extracts_winner() -> None:
    fixture = FIXTURES / "sample_sim_output.txt"
    result = parse_smoke_test_output(fixture.read_text(encoding="utf-8"))
    assert result.winner == "Ai(2)-alela"
    assert result.is_draw is False


def test_parse_sim_output_aggregates_five_match_fixture() -> None:
    fixture = FIXTURES / "forge_v2_B5_output.txt"
    result = parse_sim_output(fixture.read_text(encoding="utf-8"))
    assert result.wins == 2
    assert result.losses == 3
    assert result.draws == 0
    assert result.matches_played == 5
    assert result.game_durations_ms == (13014, 3057, 120000, 6684, 72436)
    assert result.game_winners[0] == "Ai(2)-Atraxa"
    assert result.game_winners[-1] == "Ai(2)-Atraxa"


def test_parse_sim_output_rejects_missing_result_lines() -> None:
    with pytest.raises(ValueError, match="Could not parse any Forge game results"):
        parse_sim_output("Simulation mode\n")
