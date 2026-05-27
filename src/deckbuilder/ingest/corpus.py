"""Corpus ingestion for Task T6."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert

from deckbuilder.db.models import Card, TrainingDeck
from deckbuilder.db.session import get_engine, get_session


@dataclass(slots=True)
class CorpusIngestReport:
    """Summary metrics from one corpus ingest run."""

    csv_path: Path
    commander_name: str
    csv_row_count: int
    inserted_row_count: int
    total_card_refs: int
    resolved_card_refs: int
    full_resolution_deck_count: int
    unresolved_name_counts: Counter[str]

    @property
    def overall_resolution_rate(self) -> float:
        """Return resolved / total card references."""
        if self.total_card_refs == 0:
            return 0.0
        return self.resolved_card_refs / self.total_card_refs


@dataclass(slots=True)
class _DeckRow:
    source: str
    commander_name: str
    card_names: list[str]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_report_path() -> Path:
    return _project_root() / "progress" / "T06_corpus_ingest_report.md"


def _load_csv_rows(csv_path: Path) -> list[_DeckRow]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw in reader:
            source = raw.get("source_url")
            commander_name = raw.get("commander_name")
            card_names = raw.get("card_names")
            if (
                not isinstance(source, str)
                or not isinstance(commander_name, str)
                or not isinstance(card_names, str)
            ):
                msg = f"Malformed CSV row in {csv_path}: {raw!r}"
                raise RuntimeError(msg)
            rows.append(
                _DeckRow(
                    source=source,
                    commander_name=commander_name,
                    card_names=[name for name in card_names.split(";") if name],
                )
            )
        return rows


def _load_card_lookup() -> tuple[dict[str, UUID], dict[str, list[UUID]]]:
    with get_session() as session:
        rows = session.execute(select(Card.name, Card.oracle_id)).all()

    exact: dict[str, UUID] = {}
    lower_map: dict[str, list[UUID]] = defaultdict(list)
    for name, oracle_id in rows:
        exact[name] = oracle_id
        lower_map[name.casefold()].append(oracle_id)
    return exact, dict(lower_map)


def _resolve_card_name(
    card_name: str,
    exact_lookup: dict[str, UUID],
    lower_lookup: dict[str, list[UUID]],
) -> UUID | None:
    oracle_id = exact_lookup.get(card_name)
    if oracle_id is not None:
        return oracle_id
    candidates = lower_lookup.get(card_name.casefold())
    if candidates is None or len(candidates) != 1:
        return None
    return candidates[0]


def ingest_corpus(csv_path: Path) -> CorpusIngestReport:
    """Load a collected deck CSV into training_decks, keeping partial card resolution."""
    rows = _load_csv_rows(csv_path)
    if not rows:
        msg = f"CSV contains no rows: {csv_path}"
        raise RuntimeError(msg)

    commander_names = {row.commander_name for row in rows}
    if len(commander_names) != 1:
        msg = f"Expected one commander in CSV, found {sorted(commander_names)}"
        raise RuntimeError(msg)
    commander_name = next(iter(commander_names))

    exact_lookup, lower_lookup = _load_card_lookup()
    commander_oracle_id = _resolve_card_name(commander_name, exact_lookup, lower_lookup)
    if commander_oracle_id is None:
        msg = f"Commander could not be resolved in cards table: {commander_name}"
        raise RuntimeError(msg)

    insert_rows: list[dict[str, object]] = []
    total_card_refs = 0
    resolved_card_refs = 0
    full_resolution_deck_count = 0
    unresolved_name_counts: Counter[str] = Counter()

    for row in rows:
        resolved_ids: list[UUID] = []
        unresolved_for_deck = 0
        total_card_refs += len(row.card_names)
        for card_name in row.card_names:
            oracle_id = _resolve_card_name(card_name, exact_lookup, lower_lookup)
            if oracle_id is None:
                unresolved_name_counts[card_name] += 1
                unresolved_for_deck += 1
                continue
            resolved_ids.append(oracle_id)
            resolved_card_refs += 1
        if unresolved_for_deck == 0:
            full_resolution_deck_count += 1
        insert_rows.append(
            {
                "commander_oracle_id": commander_oracle_id,
                "source": row.source,
                "card_oracle_ids": resolved_ids,
            }
        )

    engine = get_engine()
    with engine.begin() as connection:
        sources = [cast(str, row["source"]) for row in insert_rows]
        connection.execute(delete(TrainingDeck).where(TrainingDeck.source.in_(sources)))
        if insert_rows:
            connection.execute(insert(TrainingDeck).values(insert_rows))

    return CorpusIngestReport(
        csv_path=csv_path,
        commander_name=commander_name,
        csv_row_count=len(rows),
        inserted_row_count=len(insert_rows),
        total_card_refs=total_card_refs,
        resolved_card_refs=resolved_card_refs,
        full_resolution_deck_count=full_resolution_deck_count,
        unresolved_name_counts=unresolved_name_counts,
    )


def write_corpus_ingest_report(
    report: CorpusIngestReport,
    training_deck_row_count: int,
    report_path: Path | None = None,
) -> Path:
    """Write the requested T6 corpus-ingest report."""
    target_path = report_path or _default_report_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    top_unresolved = report.unresolved_name_counts.most_common(10)
    lines = [
        "# T06 - Corpus ingest report",
        "",
        f"- CSV path: `{report.csv_path}`",
        f"- Commander: `{report.commander_name}`",
        f"- CSV row count: {report.csv_row_count}",
        f"- training_decks row count: {training_deck_row_count}",
        (
            "- Overall resolution rate: "
            f"{report.resolved_card_refs}/{report.total_card_refs} "
            f"({report.overall_resolution_rate:.2%})"
        ),
        f"- Per-deck full-resolution count: {report.full_resolution_deck_count}",
        "",
        "## Top 10 unresolved card names",
    ]
    if top_unresolved:
        for name, count in top_unresolved:
            lines.append(f"- `{name}`: {count}")
    else:
        lines.append("- None")

    meets_rows = training_deck_row_count >= 300
    meets_resolution = report.overall_resolution_rate >= 0.95
    lines.extend(
        [
            "",
            "## T6 acceptance",
            f"- training_decks rows acceptable: {'yes' if meets_rows else 'no'}",
            f"- resolution rate >=95%: {'yes' if meets_resolution else 'no'}",
            (f"- Overall acceptance met: {'yes' if meets_rows and meets_resolution else 'no'}"),
            "",
            "## Notes",
            "- Decks with unresolved cards were kept with partial `card_oracle_ids` data.",
            "- Existing `training_decks` rows matching the same source URLs "
            "were replaced before insert.",
        ]
    )
    target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target_path


def unresolved_names(report: CorpusIngestReport) -> Iterable[tuple[str, int]]:
    """Expose unresolved names and counts for callers that need them."""
    return report.unresolved_name_counts.most_common()
