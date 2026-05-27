"""Helpers for parsing Forge simulation output."""

from __future__ import annotations

import re
from dataclasses import dataclass

GAME_WIN_RE = re.compile(
    r"^Game Result: Game (?P<game>\d+) ended in (?P<duration_ms>\d+) ms\. "
    r"(?P<winner>.+?) has won!$",
    re.MULTILINE,
)
GAME_DRAW_RE = re.compile(
    r"^Game Result: Game (?P<game>\d+) ended in "
    r"(?:(?P<duration_ms>\d+) ms\. )?"
    r"(?:a Draw!|A Draw!|draw!|Draw!)$",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class SimResult:
    """Aggregated Forge simulation results from the candidate deck perspective."""

    wins: int
    losses: int
    draws: int
    game_durations_ms: tuple[int | None, ...]
    game_winners: tuple[str | None, ...]
    raw_output: str

    @property
    def matches_played(self) -> int:
        """Total number of games reported by Forge."""
        return self.wins + self.losses + self.draws


@dataclass(frozen=True, slots=True)
class SmokeTestParseResult:
    """Parsed result for a one-match Forge smoke test."""

    winner: str | None
    is_draw: bool
    raw_output: str


def parse_sim_output(text: str) -> SimResult:
    """Parse Forge CLI sim output into aggregate results."""
    per_game: dict[int, tuple[str | None, int | None]] = {}

    for match in GAME_WIN_RE.finditer(text):
        game_number = int(match.group("game"))
        per_game[game_number] = (match.group("winner"), int(match.group("duration_ms")))

    for match in GAME_DRAW_RE.finditer(text):
        game_number = int(match.group("game"))
        duration_raw = match.group("duration_ms")
        per_game[game_number] = (None, int(duration_raw) if duration_raw is not None else None)

    if not per_game:
        msg = "Could not parse any Forge game results from output"
        raise ValueError(msg)

    wins = 0
    losses = 0
    draws = 0
    winners: list[str | None] = []
    durations: list[int | None] = []
    for game_number in sorted(per_game):
        winner, duration_ms = per_game[game_number]
        winners.append(winner)
        durations.append(duration_ms)
        if winner is None:
            draws += 1
            continue
        if winner.startswith("Ai(1)-"):
            wins += 1
            continue
        if winner.startswith("Ai(2)-"):
            losses += 1
            continue
        msg = f"Unexpected Forge winner label: {winner!r}"
        raise ValueError(msg)

    return SimResult(
        wins=wins,
        losses=losses,
        draws=draws,
        game_durations_ms=tuple(durations),
        game_winners=tuple(winners),
        raw_output=text,
    )


def parse_smoke_test_output(text: str) -> SmokeTestParseResult:
    """Extract the winner from a one-match Forge smoke test output."""
    result = parse_sim_output(text)
    if result.matches_played != 1:
        msg = f"Expected exactly one game in smoke output, found {result.matches_played}"
        raise ValueError(msg)
    winner = result.game_winners[0]
    return SmokeTestParseResult(winner=winner, is_draw=winner is None, raw_output=text)
