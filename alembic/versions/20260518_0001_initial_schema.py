"""Initial schema.

Revision ID: 20260518_0001
Revises: None
Create Date: 2026-05-18 17:20:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260518_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "cards",
        sa.Column("oracle_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("mana_cost", sa.Text(), nullable=True),
        sa.Column("cmc", sa.Float(), nullable=True),
        sa.Column("type_line", sa.Text(), nullable=False),
        sa.Column("oracle_text", sa.Text(), nullable=True),
        sa.Column("color_identity", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("legality_commander", sa.Text(), nullable=False),
        sa.Column("is_commander_legal_as_commander", sa.Boolean(), nullable=False),
        sa.Column("scryfall_uri", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_cards_name", "cards", ["name"])
    op.create_index("ix_cards_legality_commander", "cards", ["legality_commander"])
    op.create_index("ix_cards_color_identity", "cards", ["color_identity"], postgresql_using="gin")

    op.create_table(
        "card_embeddings",
        sa.Column(
            "oracle_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("cards.oracle_id"),
            primary_key=True,
        ),
        sa.Column("model_name", sa.Text(), primary_key=True),
        sa.Column("embedding", Vector(dim=384), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index(
        "ix_card_embeddings_embedding_hnsw",
        "card_embeddings",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_l2_ops"},
    )

    op.create_table(
        "training_decks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "commander_oracle_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("cards.oracle_id"),
            nullable=False,
        ),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column(
            "card_oracle_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index(
        "ix_training_decks_commander_oracle_id", "training_decks", ["commander_oracle_id"]
    )

    op.create_table(
        "experiment_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "commander_oracle_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("cards.oracle_id"),
            nullable=False,
        ),
        sa.Column("n_decks", sa.Integer(), nullable=False),
        sa.Column("matches_per_deck", sa.Integer(), nullable=False),
        sa.Column(
            "started_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("completed_at", sa.DateTime(timezone=False), nullable=True),
        sa.Column("mean_absolute_deviation", sa.Float(), nullable=True),
        sa.Column("max_deviation", sa.Float(), nullable=True),
        sa.Column("adversarial_rate", sa.Float(), nullable=True),
        sa.Column("decision", sa.Text(), nullable=True),
        sa.Column("report_path", sa.Text(), nullable=True),
    )

    op.create_table(
        "awr_coefficients",
        sa.Column(
            "commander_oracle_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("cards.oracle_id"),
            primary_key=True,
        ),
        sa.Column(
            "oracle_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("cards.oracle_id"),
            primary_key=True,
        ),
        sa.Column("fit_run_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("strength_intercept", sa.Float(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False
        ),
    )

    op.create_table(
        "awr_synergy",
        sa.Column(
            "commander_oracle_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("cards.oracle_id"),
            primary_key=True,
        ),
        sa.Column(
            "card_a_oracle_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("cards.oracle_id"),
            primary_key=True,
        ),
        sa.Column(
            "card_b_oracle_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("cards.oracle_id"),
            primary_key=True,
        ),
        sa.Column("fit_run_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("synergy", sa.Float(), nullable=False),
    )

    op.create_table(
        "generated_decks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "commander_oracle_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("cards.oracle_id"),
            nullable=False,
        ),
        sa.Column(
            "card_oracle_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False
        ),
        sa.Column("predicted_win_rate", sa.Float(), nullable=False),
        sa.Column(
            "experiment_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("experiment_runs.id"),
            nullable=False,
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False
        ),
    )

    op.create_table(
        "sim_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "generated_deck_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("generated_decks.id"),
            nullable=False,
        ),
        sa.Column("opponent_deck_name", sa.Text(), nullable=False),
        sa.Column("matches_played", sa.Integer(), nullable=False),
        sa.Column("wins", sa.Integer(), nullable=False),
        sa.Column("losses", sa.Integer(), nullable=False),
        sa.Column("draws", sa.Integer(), nullable=False),
        sa.Column("actual_win_rate", sa.Float(), nullable=False),
        sa.Column("forge_log_path", sa.Text(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=False), server_default=sa.func.now(), nullable=False
        ),
    )


def downgrade() -> None:
    op.drop_table("sim_results")
    op.drop_table("generated_decks")
    op.drop_table("awr_synergy")
    op.drop_table("awr_coefficients")
    op.drop_table("experiment_runs")
    op.drop_index("ix_training_decks_commander_oracle_id", table_name="training_decks")
    op.drop_table("training_decks")
    op.drop_index("ix_card_embeddings_embedding_hnsw", table_name="card_embeddings")
    op.drop_table("card_embeddings")
    op.drop_index("ix_cards_color_identity", table_name="cards")
    op.drop_index("ix_cards_legality_commander", table_name="cards")
    op.drop_index("ix_cards_name", table_name="cards")
    op.drop_table("cards")
