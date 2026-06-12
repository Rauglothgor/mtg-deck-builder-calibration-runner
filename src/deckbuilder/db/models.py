"""SQLAlchemy models for the v0.5 experiment schema."""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class Card(Base):
    """Commander-legal card data ingested from Scryfall."""

    __tablename__ = "cards"

    oracle_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    name: Mapped[str] = mapped_column(Text, index=True)
    mana_cost: Mapped[str | None] = mapped_column(Text, nullable=True)
    cmc: Mapped[float | None] = mapped_column(Float, nullable=True)
    type_line: Mapped[str] = mapped_column(Text)
    oracle_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    color_identity: Mapped[list[str]] = mapped_column(ARRAY(String()), default=list)
    legality_commander: Mapped[str] = mapped_column(Text, index=True)
    is_commander_legal_as_commander: Mapped[bool] = mapped_column(Boolean)
    scryfall_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now()
    )


class CardEmbedding(Base):
    """Vector embeddings for ingested cards."""

    __tablename__ = "card_embeddings"

    oracle_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cards.oracle_id"), primary_key=True
    )
    model_name: Mapped[str] = mapped_column(Text, primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(384))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now()
    )


class TrainingDeck(Base):
    """Curated commander decklists used for AWR fitting."""

    __tablename__ = "training_decks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    commander_oracle_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cards.oracle_id"), index=True
    )
    source: Mapped[str] = mapped_column(Text)
    card_oracle_ids: Mapped[list[uuid.UUID]] = mapped_column(ARRAY(UUID(as_uuid=True)))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now()
    )


class AwrCoefficient(Base):
    """Per-card AWR coefficients for one commander fit run."""

    __tablename__ = "awr_coefficients"

    commander_oracle_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cards.oracle_id"), primary_key=True
    )
    oracle_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cards.oracle_id"), primary_key=True
    )
    fit_run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    strength_intercept: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now()
    )


class AwrSynergy(Base):
    """Pairwise synergy coefficients for one commander fit run."""

    __tablename__ = "awr_synergy"

    commander_oracle_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cards.oracle_id"), primary_key=True
    )
    card_a_oracle_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cards.oracle_id"), primary_key=True
    )
    card_b_oracle_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cards.oracle_id"), primary_key=True
    )
    fit_run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    synergy: Mapped[float] = mapped_column(Float)


class GeneratedDeck(Base):
    """Surrogate-generated decks for a given experiment run."""

    __tablename__ = "generated_decks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    commander_oracle_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cards.oracle_id")
    )
    card_oracle_ids: Mapped[list[uuid.UUID]] = mapped_column(ARRAY(UUID(as_uuid=True)))
    predicted_win_rate: Mapped[float] = mapped_column(Float)
    experiment_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("experiment_runs.id")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now()
    )


class SimResult(Base):
    """Forge simulation results for a generated deck."""

    __tablename__ = "sim_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    generated_deck_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("generated_decks.id")
    )
    opponent_deck_name: Mapped[str] = mapped_column(Text)
    matches_played: Mapped[int] = mapped_column(Integer)
    wins: Mapped[int] = mapped_column(Integer)
    losses: Mapped[int] = mapped_column(Integer)
    draws: Mapped[int] = mapped_column(Integer)
    actual_win_rate: Mapped[float] = mapped_column(Float)
    forge_log_path: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now()
    )


class ExperimentRun(Base):
    """Metadata and summary metrics for an experiment execution."""

    __tablename__ = "experiment_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    commander_oracle_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cards.oracle_id")
    )
    n_decks: Mapped[int] = mapped_column(Integer)
    matches_per_deck: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(Text, default="pending")
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    forge_ai_profile: Mapped[str] = mapped_column(Text, default="forge-baseline")
    forge_build_id: Mapped[str] = mapped_column(Text, default="unknown")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    mean_absolute_deviation: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_deviation: Mapped[float | None] = mapped_column(Float, nullable=True)
    adversarial_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_path: Mapped[str | None] = mapped_column(Text, nullable=True)
