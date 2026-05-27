"""Archidekt deck helpers and collection flow."""

from __future__ import annotations

import csv
import json
import random
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

JsonMapping = Mapping[str, Any]

COMMANDER_CATEGORY = "Commander"
DEFAULT_USER_AGENT = "deck-builder-research/0.1"
INITIAL_QUERY_URL = "https://archidekt.com/api/decks/v3/?cardName={query}"
DETAIL_URL = "https://archidekt.com/api/decks/{deck_id}/"
DECK_URL = "https://archidekt.com/decks/{deck_id}"
MIN_REQUEST_GAP_SECONDS = 1.0
MAX_REQUESTS = 1500
MAX_WALL_CLOCK_SECONDS = 90 * 60
MIN_MAINDECK_SIZE = 97
MAX_MAINDECK_SIZE = 101
STATE_FILENAME = ".archidekt_collection_state.json"
COLLECTION_REPORT_FILENAME = "T06_phase2b_collection_report.md"


@dataclass(slots=True)
class CollectionArtifacts:
    """Paths written by the Archidekt collection command."""

    csv_path: Path
    state_path: Path
    report_path: Path


class ArchidektCollectionError(RuntimeError):
    """Raised when the Archidekt collection flow hits a stop condition."""


class ArchidektRequester:
    """HTTP requester with required rate limiting and backoff."""

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT) -> None:
        self.user_agent = user_agent
        self.last_request_at = 0.0
        self.requests_made = 0

    def fetch_json(self, url: str) -> dict[str, Any]:
        """Fetch one JSON payload, respecting rate limit and retry policy."""
        backoff_seconds = 1.0
        while True:
            self._sleep_for_rate_limit()
            request = Request(
                url,
                headers={"User-Agent": self.user_agent, "Accept": "application/json"},
            )
            try:
                with urlopen(request, timeout=30) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                self.last_request_at = time.monotonic()
                self.requests_made += 1
                if not isinstance(payload, dict):
                    msg = f"Unexpected non-object JSON payload from {url}"
                    raise ArchidektCollectionError(msg)
                return cast(dict[str, Any], payload)
            except HTTPError as exc:
                self.last_request_at = time.monotonic()
                self.requests_made += 1
                if exc.code == 429 or 500 <= exc.code <= 599:
                    retry_after = exc.headers.get("Retry-After")
                    sleep_seconds = _coerce_retry_after(retry_after) or backoff_seconds
                    time.sleep(sleep_seconds)
                    backoff_seconds = min(backoff_seconds * 2, 60.0)
                    continue
                msg = f"Stopping on HTTP {exc.code} for {url}"
                raise ArchidektCollectionError(msg) from exc
            except URLError as exc:
                time.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, 60.0)
                if backoff_seconds >= 60.0:
                    msg = f"Stopping after repeated URL errors for {url}: {exc.reason}"
                    raise ArchidektCollectionError(msg) from exc

    def _sleep_for_rate_limit(self) -> None:
        elapsed = time.monotonic() - self.last_request_at
        if elapsed < MIN_REQUEST_GAP_SECONDS:
            time.sleep(MIN_REQUEST_GAP_SECONDS - elapsed)


def project_root() -> Path:
    """Return the repository root path."""
    return Path(__file__).resolve().parents[3]


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def _coerce_retry_after(value: str | None) -> float | None:
    """Convert a Retry-After header value to seconds when possible."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _card_name(card_row: JsonMapping) -> str | None:
    """Return the Oracle card name from an Archidekt card row when present."""
    card = card_row.get("card")
    if not isinstance(card, Mapping):
        return None
    oracle_card = card.get("oracleCard")
    if not isinstance(oracle_card, Mapping):
        return None
    name = oracle_card.get("name")
    return name if isinstance(name, str) else None


def _card_categories(card_row: JsonMapping) -> list[str]:
    """Return normalized category names for one Archidekt card row."""
    raw_categories = card_row.get("categories")
    if not isinstance(raw_categories, list):
        return []
    return [category for category in raw_categories if isinstance(category, str)]


def is_commander(deck_json: JsonMapping, card_name: str) -> bool:
    """Return True when the named card is tagged as the deck's commander.

    Phase 2A validation showed the canonical signal is the per-card category label
    ``"Commander"`` on the row whose ``card.oracleCard.name`` matches the target.
    """
    cards = deck_json.get("cards")
    if not isinstance(cards, list):
        return False

    for card_row in cards:
        if not isinstance(card_row, Mapping):
            continue
        if _card_name(card_row) != card_name:
            continue
        if COMMANDER_CATEGORY in _card_categories(card_row):
            return True
    return False


def commander_names(deck_json: JsonMapping) -> list[str]:
    """Return all cards explicitly tagged as commanders in deck order."""
    cards = deck_json.get("cards")
    if not isinstance(cards, list):
        msg = "Detail payload missing cards[]"
        raise ArchidektCollectionError(msg)

    names: list[str] = []
    for card_row in cards:
        if not isinstance(card_row, Mapping):
            continue
        if COMMANDER_CATEGORY not in _card_categories(card_row):
            continue
        name = _card_name(card_row)
        if name is None:
            msg = "Commander-tagged card row missing card.oracleCard.name"
            raise ArchidektCollectionError(msg)
        names.append(name)
    return names


def category_inclusion_map(deck_json: JsonMapping) -> dict[str, bool]:
    """Build a category -> includedInDeck lookup from top-level deck categories."""
    categories = deck_json.get("categories")
    if not isinstance(categories, list):
        msg = "Detail payload missing top-level categories[]"
        raise ArchidektCollectionError(msg)

    inclusion: dict[str, bool] = {}
    for category in categories:
        if not isinstance(category, Mapping):
            continue
        name = category.get("name")
        included = category.get("includedInDeck")
        if not isinstance(name, str) or not isinstance(included, bool):
            continue
        inclusion[name] = included
    return inclusion


def is_included_deck_row(card_row: JsonMapping, inclusion_map: Mapping[str, bool]) -> bool:
    """Return True when a card row belongs to the playable deck sections."""
    categories = _card_categories(card_row)
    if not categories:
        return True
    for category in categories:
        included = inclusion_map.get(category)
        if included is False:
            return False
    return True


def extract_non_commander_card_names(deck_json: JsonMapping) -> list[str]:
    """Return the included non-commander card list, expanded by quantity."""
    cards = deck_json.get("cards")
    if not isinstance(cards, list):
        msg = "Detail payload missing cards[]"
        raise ArchidektCollectionError(msg)

    inclusion_map = category_inclusion_map(deck_json)
    names: list[str] = []
    for card_row in cards:
        if not isinstance(card_row, Mapping):
            continue
        if not is_included_deck_row(card_row, inclusion_map):
            continue
        if COMMANDER_CATEGORY in _card_categories(card_row):
            continue
        name = _card_name(card_row)
        quantity = card_row.get("quantity")
        if name is None or not isinstance(quantity, int):
            msg = "Included deck row missing card name or integer quantity"
            raise ArchidektCollectionError(msg)
        names.extend([name] * quantity)
    return names


def build_state_path(output_path: Path) -> Path:
    """Return the state file path for a collection output CSV."""
    return output_path.parent / STATE_FILENAME


def build_report_path() -> Path:
    """Return the standard Phase 2B report path."""
    return project_root() / "progress" / COLLECTION_REPORT_FILENAME


def _default_state(commander_name: str, target: int, output_path: Path) -> dict[str, Any]:
    encoded_query = urlencode({"cardName": commander_name})
    return {
        "commander_name": commander_name,
        "target": target,
        "output_path": str(output_path),
        "query_url": INITIAL_QUERY_URL.format(query=encoded_query.split("=", maxsplit=1)[1]),
        "user_agent": DEFAULT_USER_AGENT,
        "visited_ids": [],
        "accepted_ids": [],
        "rejections": {},
        "reject_reason_counts": {},
        "accept_count": 0,
        "pages_fetched": 0,
        "details_fetched": 0,
        "requests_made": 0,
        "last_page": 0,
        "next_url": INITIAL_QUERY_URL.format(query=encoded_query.split("=", maxsplit=1)[1]),
        "started_at": _now_iso(),
        "updated_at": _now_iso(),
        "stopped_reason": None,
        "created_at_min": None,
        "created_at_max": None,
        "updated_at_min": None,
        "updated_at_max": None,
    }


def load_state(
    state_path: Path,
    commander_name: str,
    target: int,
    output_path: Path,
) -> dict[str, Any]:
    """Load or initialize resumable collection state."""
    if not state_path.exists():
        return _default_state(commander_name, target, output_path)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        msg = f"State file is not a JSON object: {state_path}"
        raise ArchidektCollectionError(msg)
    if state.get("commander_name") != commander_name:
        msg = "State commander_name does not match requested commander"
        raise ArchidektCollectionError(msg)
    if Path(str(state.get("output_path"))) != output_path:
        msg = "State output_path does not match requested output path"
        raise ArchidektCollectionError(msg)
    return cast(dict[str, Any], state)


def save_state(state_path: Path, state: Mapping[str, Any]) -> None:
    """Persist collection state atomically."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(state_path)


def _ensure_csv_header(output_path: Path, state: Mapping[str, Any]) -> None:
    """Create the CSV with its required header, or validate resume state."""
    accepted_ids = state.get("accepted_ids")
    accepted_count = len(accepted_ids) if isinstance(accepted_ids, list) else 0
    if output_path.exists():
        return
    if accepted_count > 0:
        msg = "State indicates prior accepted rows but output CSV is missing"
        raise ArchidektCollectionError(msg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_url", "commander_name", "card_names"])


def _append_csv_row(
    output_path: Path,
    source_url: str,
    commander_name: str,
    card_names: list[str],
) -> None:
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([source_url, commander_name, ";".join(card_names)])


def _parse_page_number(url: str) -> int:
    parsed = urlparse(url)
    pages = parse_qs(parsed.query).get("page")
    if not pages:
        return 1
    try:
        return int(pages[0])
    except ValueError as exc:
        msg = f"Unexpected page value in next URL: {url}"
        raise ArchidektCollectionError(msg) from exc


def _normalize_next_url(next_url: object) -> str | None:
    if next_url is None:
        return None
    if not isinstance(next_url, str) or not next_url:
        msg = f"Unexpected next URL value: {next_url!r}"
        raise ArchidektCollectionError(msg)
    return next_url.replace("http://archidekt.com", "https://archidekt.com", 1)


def _update_date_span(state: dict[str, Any], key_min: str, key_max: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        return
    current_min = state.get(key_min)
    current_max = state.get(key_max)
    if not isinstance(current_min, str) or value < current_min:
        state[key_min] = value
    if not isinstance(current_max, str) or value > current_max:
        state[key_max] = value


def _reject(state: dict[str, Any], deck_id: int, reason: str) -> None:
    rejections = state.setdefault("rejections", {})
    if not isinstance(rejections, dict):
        msg = "State rejections field is not a dictionary"
        raise ArchidektCollectionError(msg)
    rejections[str(deck_id)] = reason

    histogram = state.setdefault("reject_reason_counts", {})
    if not isinstance(histogram, dict):
        msg = "State reject_reason_counts field is not a dictionary"
        raise ArchidektCollectionError(msg)
    histogram[reason] = int(histogram.get(reason, 0)) + 1


def _accept(state: dict[str, Any], deck_id: int) -> None:
    accepted_ids = state.setdefault("accepted_ids", [])
    if not isinstance(accepted_ids, list):
        msg = "State accepted_ids field is not a list"
        raise ArchidektCollectionError(msg)
    accepted_ids.append(deck_id)
    state["accept_count"] = int(state.get("accept_count", 0)) + 1


def _mark_visited(state: dict[str, Any], deck_id: int) -> None:
    visited_ids = state.setdefault("visited_ids", [])
    if not isinstance(visited_ids, list):
        msg = "State visited_ids field is not a list"
        raise ArchidektCollectionError(msg)
    visited_ids.append(deck_id)


def _load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_collection_report(state: Mapping[str, Any], csv_path: Path, report_path: Path) -> None:
    """Write the Phase 2B collection summary report."""
    rows = _load_csv_rows(csv_path)
    reject_histogram: Counter[str] = Counter()
    raw_histogram = state.get("reject_reason_counts")
    if isinstance(raw_histogram, Mapping):
        for reason, count in raw_histogram.items():
            if isinstance(reason, str) and isinstance(count, int):
                reject_histogram[reason] = count

    rng = random.Random(0)
    sample_size = min(5, len(rows))
    spot_checks = rng.sample(rows, sample_size) if sample_size else []

    lines = [
        "# T06 - Phase 2B collection report",
        "",
        "## Run summary",
        f"- Pages fetched: {state.get('pages_fetched', 0)}",
        "- Unique deck IDs visited: "
        f"{len(state.get('visited_ids', [])) if isinstance(state.get('visited_ids'), list) else 0}",
        f"- Detail payloads fetched: {state.get('details_fetched', 0)}",
        f"- Total HTTP requests: {state.get('requests_made', 0)}",
        f"- Accept count: {state.get('accept_count', 0)}",
        f"- CSV row count: {len(rows)}",
        f"- Stop reason: {state.get('stopped_reason')}",
        "",
        "## Reject reason histogram",
    ]
    if reject_histogram:
        for reason, count in sorted(reject_histogram.items()):
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Deck date span",
            f"- Earliest createdAt: {state.get('created_at_min')}",
            f"- Latest createdAt: {state.get('created_at_max')}",
            f"- Earliest updatedAt: {state.get('updated_at_min')}",
            f"- Latest updatedAt: {state.get('updated_at_max')}",
            "",
            "## 5 random spot-checks",
        ]
    )
    if spot_checks:
        for row in spot_checks:
            card_count = len([name for name in row["card_names"].split(";") if name])
            lines.append(f"- {row['source_url']} - {card_count} cards")
    else:
        lines.append("- No accepted rows to sample")

    lines.extend(
        [
            "",
            "## T6 acceptance",
            "- Requirement: >=200 rows and deck size 99+/-2",
            f"- Row target met: {'yes' if len(rows) >= 200 else 'no'}",
            "- Deck size filter enforced during collection: yes",
            "",
            "## Output artifacts",
            f"- CSV: `{csv_path}`",
            f"- State: `{build_state_path(csv_path)}`",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_archidekt_corpus(
    commander_name: str,
    target: int,
    output_path: Path,
) -> CollectionArtifacts:
    """Collect a bounded Commander corpus from Archidekt v3 + detail endpoints."""
    state_path = build_state_path(output_path)
    report_path = build_report_path()
    state = load_state(state_path, commander_name, target, output_path)
    _ensure_csv_header(output_path, state)

    requester = ArchidektRequester(user_agent=DEFAULT_USER_AGENT)
    visited_existing = state.get("visited_ids")
    visited_ids = set(visited_existing) if isinstance(visited_existing, list) else set()
    started_at = time.monotonic()

    try:
        next_url = state.get("next_url")
        if not isinstance(next_url, str) or not next_url:
            msg = "State next_url is missing or invalid"
            raise ArchidektCollectionError(msg)

        while int(state.get("accept_count", 0)) < target:
            if requester.requests_made >= MAX_REQUESTS:
                state["stopped_reason"] = f"request_cap_reached:{MAX_REQUESTS}"
                break
            if time.monotonic() - started_at >= MAX_WALL_CLOCK_SECONDS:
                state["stopped_reason"] = f"wall_clock_cap_reached:{MAX_WALL_CLOCK_SECONDS}"
                break

            page_payload = requester.fetch_json(next_url)
            state["requests_made"] = requester.requests_made
            results = page_payload.get("results")
            if not isinstance(results, list):
                msg = "v3 page payload missing results[]"
                raise ArchidektCollectionError(msg)
            state["pages_fetched"] = int(state.get("pages_fetched", 0)) + 1
            state["last_page"] = _parse_page_number(next_url)
            state["next_url"] = _normalize_next_url(page_payload.get("next"))
            state["updated_at"] = _now_iso()
            save_state(state_path, state)

            for summary_row in results:
                if int(state.get("accept_count", 0)) >= target:
                    break
                if requester.requests_made >= MAX_REQUESTS:
                    state["stopped_reason"] = f"request_cap_reached:{MAX_REQUESTS}"
                    break
                if time.monotonic() - started_at >= MAX_WALL_CLOCK_SECONDS:
                    state["stopped_reason"] = f"wall_clock_cap_reached:{MAX_WALL_CLOCK_SECONDS}"
                    break
                if not isinstance(summary_row, Mapping):
                    msg = "v3 results[] contained a non-object row"
                    raise ArchidektCollectionError(msg)
                deck_id = summary_row.get("id")
                if not isinstance(deck_id, int):
                    msg = f"v3 result missing integer id: {summary_row!r}"
                    raise ArchidektCollectionError(msg)
                if deck_id in visited_ids:
                    continue

                detail_payload = requester.fetch_json(DETAIL_URL.format(deck_id=deck_id))
                state["requests_made"] = requester.requests_made
                state["details_fetched"] = int(state.get("details_fetched", 0)) + 1
                _mark_visited(state, deck_id)
                visited_ids.add(deck_id)
                _update_date_span(
                    state,
                    "created_at_min",
                    "created_at_max",
                    detail_payload.get("createdAt"),
                )
                _update_date_span(
                    state,
                    "updated_at_min",
                    "updated_at_max",
                    detail_payload.get("updatedAt"),
                )

                if not is_commander(detail_payload, commander_name):
                    _reject(state, deck_id, "target_not_commander")
                else:
                    main_deck_cards = extract_non_commander_card_names(detail_payload)
                    if not MIN_MAINDECK_SIZE <= len(main_deck_cards) <= MAX_MAINDECK_SIZE:
                        _reject(
                            state,
                            deck_id,
                            f"main_deck_count_out_of_range:{len(main_deck_cards)}",
                        )
                    else:
                        _append_csv_row(
                            output_path,
                            DECK_URL.format(deck_id=deck_id),
                            commander_name,
                            main_deck_cards,
                        )
                        _accept(state, deck_id)

                state["updated_at"] = _now_iso()
                save_state(state_path, state)

            if state.get("stopped_reason") is not None:
                break
            next_url = state.get("next_url")
            if next_url is None:
                state["stopped_reason"] = "no_more_pages"
                break
            if not isinstance(next_url, str) or not next_url:
                msg = "State next_url became invalid during pagination"
                raise ArchidektCollectionError(msg)

        if state.get("stopped_reason") is None:
            state["stopped_reason"] = (
                f"target_reached:{state.get('accept_count', 0)}"
                if int(state.get("accept_count", 0)) >= target
                else "stopped"
            )
    except ArchidektCollectionError as exc:
        state["stopped_reason"] = f"anomaly:{exc}"
        state["updated_at"] = _now_iso()
        save_state(state_path, state)
        raise

    state["updated_at"] = _now_iso()
    save_state(state_path, state)
    write_collection_report(state, output_path, report_path)
    return CollectionArtifacts(csv_path=output_path, state_path=state_path, report_path=report_path)
