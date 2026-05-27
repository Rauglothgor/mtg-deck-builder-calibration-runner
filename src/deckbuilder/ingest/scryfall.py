"""Scryfall bulk-data ingestion for the cards table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from urllib.request import Request, urlopen

from sqlalchemy.dialects.postgresql import insert

from deckbuilder.db.models import Card
from deckbuilder.db.session import get_engine

SCRYFALL_BULK_DATA_URL = "https://api.scryfall.com/bulk-data"
ORACLE_CARDS_TYPE = "oracle_cards"
BATCH_SIZE = 1000
REQUEST_HEADERS = {
    "User-Agent": "deckbuilder-v0.5/0.5.0",
    "Accept": "application/json",
}


def project_root() -> Path:
    """Return the repository root path."""
    return Path(__file__).resolve().parents[3]


def raw_data_path() -> Path:
    """Return the local cache path for the oracle cards bulk file."""
    return project_root() / "data" / "raw" / "scryfall_oracle_cards.json"


def fetch_bulk_metadata() -> dict[str, Any]:
    """Fetch the Scryfall bulk-data catalog and return the oracle cards entry."""
    request = Request(SCRYFALL_BULK_DATA_URL, headers=REQUEST_HEADERS)
    with urlopen(request) as response:
        payload = cast(dict[str, Any], json.load(response))
    for entry in payload["data"]:
        if entry["type"] == ORACLE_CARDS_TYPE:
            return cast(dict[str, Any], entry)
    msg = "Scryfall oracle_cards bulk entry not found"
    raise RuntimeError(msg)


def download_oracle_cards() -> Path:
    """Download the oracle cards bulk file into the raw data directory."""
    metadata = fetch_bulk_metadata()
    destination = raw_data_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(metadata["download_uri"], headers=REQUEST_HEADERS)
    with urlopen(request) as response, destination.open("wb") as handle:
        handle.write(response.read())
    return destination


def load_oracle_cards(path: Path) -> list[dict[str, Any]]:
    """Load the downloaded bulk file from disk."""
    with path.open(encoding="utf-8") as handle:
        return cast(list[dict[str, Any]], json.load(handle))


def normalize_card(card: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one oracle card record to the database schema."""
    legalities = card.get("legalities", {})
    legality_commander = legalities.get("commander")
    if legality_commander is None:
        return None

    oracle_text = card.get("oracle_text") or ""
    type_line = card.get("type_line") or ""
    is_commander_legal = (
        "Legendary Creature" in type_line or "can be your commander" in oracle_text.lower()
    )

    return {
        "oracle_id": card["oracle_id"],
        "name": card["name"],
        "mana_cost": card.get("mana_cost"),
        "cmc": float(card.get("cmc", 0.0)) if card.get("cmc") is not None else None,
        "type_line": type_line,
        "oracle_text": card.get("oracle_text"),
        "color_identity": card.get("color_identity", []),
        "legality_commander": legality_commander,
        "is_commander_legal_as_commander": is_commander_legal,
        "scryfall_uri": card.get("scryfall_uri"),
    }


def chunked(rows: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    """Split rows into fixed-size batches."""
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def upsert_cards(rows: list[dict[str, Any]]) -> int:
    """Upsert normalized cards by oracle_id."""
    engine = get_engine()
    inserted = 0
    with engine.begin() as connection:
        for batch in chunked(rows, BATCH_SIZE):
            statement = insert(Card).values(batch)
            update_columns = {
                column: getattr(statement.excluded, column)
                for column in [
                    "name",
                    "mana_cost",
                    "cmc",
                    "type_line",
                    "oracle_text",
                    "color_identity",
                    "legality_commander",
                    "is_commander_legal_as_commander",
                    "scryfall_uri",
                ]
            }
            upsert_statement = statement.on_conflict_do_update(
                index_elements=[Card.oracle_id],
                set_=update_columns,
            )
            connection.execute(upsert_statement)
            inserted += len(batch)
    return inserted


def ingest_scryfall() -> tuple[Path, int]:
    """Download, normalize, and upsert Scryfall oracle cards."""
    path = download_oracle_cards()
    raw_cards = load_oracle_cards(path)
    normalized = [card for item in raw_cards if (card := normalize_card(item)) is not None]
    total = upsert_cards(normalized)
    return path, total
