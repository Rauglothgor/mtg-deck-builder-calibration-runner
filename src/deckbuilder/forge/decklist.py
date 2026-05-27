"""Forge decklist serialization for Task T11."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from sqlalchemy import select

from deckbuilder.db.models import Card
from deckbuilder.db.session import get_session


def _load_card_names(oracle_ids: list[UUID]) -> dict[UUID, str]:
    with get_session() as session:
        rows = session.execute(
            select(Card.oracle_id, Card.name).where(Card.oracle_id.in_(oracle_ids))
        ).all()
    typed_rows = [(oracle_id, name) for oracle_id, name in rows]
    return dict(typed_rows)


def _deck_display_name(commander_name: str) -> str:
    return commander_name.split(",", maxsplit=1)[0].strip()


def to_dck_format(
    commander_oracle_id: UUID,
    card_oracle_ids: list[UUID],
    output_path: str | Path,
) -> Path:
    """Write a Forge commander decklist to ``output_path`` and return the path."""
    requested_ids = [commander_oracle_id, *card_oracle_ids]
    names_by_id = _load_card_names(requested_ids)
    missing_ids = [oracle_id for oracle_id in requested_ids if oracle_id not in names_by_id]
    if missing_ids:
        msg = f"Could not resolve card names for oracle ids: {missing_ids}"
        raise RuntimeError(msg)

    commander_name = names_by_id[commander_oracle_id]
    lines = [
        "[metadata]",
        f"Name={_deck_display_name(commander_name)}",
        "[Commander]",
        f"1 {commander_name}",
        "[Main]",
    ]
    lines.extend(f"1 {names_by_id[oracle_id]}" for oracle_id in card_oracle_ids)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination
