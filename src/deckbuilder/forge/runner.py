"""Native Forge runner for Task T11."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from deckbuilder.config import get_settings
from deckbuilder.forge.parser import SimResult, parse_sim_output

CANDIDATE_DECK_NAME = "candidate.dck"
OPPONENT_DECK_NAME = "opponent.dck"
SECONDS_PER_MATCH = 60


def _forge_root() -> Path:
    return get_settings().forge_root


def _forge_decks_dir() -> Path:
    return get_settings().forge_decks_dir


def run_sim(
    deck_path: str | Path,
    opponent_path: str | Path,
    n_matches: int,
    seed: int,
) -> SimResult:
    """Run native Forge commander simulations and return parsed aggregate results.

    The ``seed`` parameter is accepted for API stability, but current native Forge CLI
    does not expose a seed flag for deterministic match simulation.
    """
    if n_matches <= 0:
        msg = f"n_matches must be positive, got {n_matches}"
        raise ValueError(msg)
    del seed

    source_candidate = Path(deck_path)
    source_opponent = Path(opponent_path)
    if not source_candidate.is_file():
        msg = f"Candidate deck file not found: {source_candidate}"
        raise FileNotFoundError(msg)
    if not source_opponent.is_file():
        msg = f"Opponent deck file not found: {source_opponent}"
        raise FileNotFoundError(msg)

    forge_root = _forge_root()
    forge_bin = forge_root / "forge.sh"
    if not forge_bin.is_file():
        msg = (
            f"Forge runtime not found at {forge_bin}. "
            "Set DECKBUILDER_FORGE_ROOT to the Forge install directory."
        )
        raise RuntimeError(msg)
    if shutil.which("xvfb-run") is None:
        msg = "xvfb-run is required to launch Forge simulations in headless mode."
        raise RuntimeError(msg)

    forge_decks_dir = _forge_decks_dir()
    forge_decks_dir.mkdir(parents=True, exist_ok=True)
    candidate_dest = forge_decks_dir / CANDIDATE_DECK_NAME
    opponent_dest = forge_decks_dir / OPPONENT_DECK_NAME

    shutil.copy2(source_candidate, candidate_dest)
    shutil.copy2(source_opponent, opponent_dest)

    try:
        completed = subprocess.run(
            [
                "xvfb-run",
                "-a",
                "./forge.sh",
                "sim",
                "-d",
                CANDIDATE_DECK_NAME,
                OPPONENT_DECK_NAME,
                "-n",
                str(n_matches),
                "-f",
                "commander",
            ],
            cwd=forge_root,
            capture_output=True,
            text=True,
            timeout=max(SECONDS_PER_MATCH, n_matches * SECONDS_PER_MATCH),
            check=False,
        )
    finally:
        candidate_dest.unlink(missing_ok=True)
        opponent_dest.unlink(missing_ok=True)

    output = completed.stdout
    if completed.stderr:
        output = f"{output}\n{completed.stderr}" if output else completed.stderr

    if completed.returncode != 0:
        tail = "\n".join(output.splitlines()[-50:])
        msg = f"Forge sim exited with code {completed.returncode}. Last output lines:\n{tail}"
        raise RuntimeError(msg)

    try:
        result = parse_sim_output(output)
    except ValueError as exc:
        tail = "\n".join(output.splitlines()[-200:])
        msg = f"{exc}. Last output lines:\n{tail}"
        raise RuntimeError(msg) from exc
    if result.matches_played != n_matches:
        msg = f"Forge reported {result.matches_played} results, expected {n_matches}"
        raise RuntimeError(msg)
    return result
