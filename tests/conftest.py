"""Pytest configuration for deckbuilder."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import psycopg
import pytest
from sqlalchemy.engine import make_url

from deckbuilder.config import get_settings


@lru_cache(maxsize=1)
def _postgres_available() -> bool:
    url = make_url(get_settings().database_url)
    try:
        with psycopg.connect(
            dbname=url.database,
            host=url.host,
            password=url.password,
            port=url.port,
            user=url.username,
            connect_timeout=2,
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
        return True
    except psycopg.Error:
        return False


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip integration tests quickly when Postgres is unavailable."""
    if _postgres_available():
        return

    skip_integration = pytest.mark.skip(
        reason=(
            "Postgres not available; start `docker compose up -d postgres` for integration tests."
        )
    )
    for item in items:
        if "integration" in Path(str(item.fspath)).parts:
            item.add_marker(skip_integration)
