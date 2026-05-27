"""Database engine and session helpers."""

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from deckbuilder.config import get_settings


def get_engine() -> Engine:
    """Create the shared synchronous SQLAlchemy engine."""
    settings = get_settings()
    return create_engine(settings.database_url, future=True)


SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, class_=Session)


def get_session() -> Session:
    """Return a new database session."""
    return SessionLocal()


def reset_database() -> None:
    """Drop and recreate the public schema for local development."""
    engine = get_engine()
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
        connection.execute(text("GRANT ALL ON SCHEMA public TO deckbuilder"))
        connection.execute(text("GRANT ALL ON SCHEMA public TO public"))
