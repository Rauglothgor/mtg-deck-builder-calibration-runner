"""Add Forge AI metadata to experiment runs.

Revision ID: 20260610_0003
Revises: 20260519_0002
Create Date: 2026-06-10 02:20:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260610_0003"
down_revision = "20260519_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "experiment_runs",
        sa.Column("forge_ai_profile", sa.Text(), nullable=False, server_default="forge-baseline"),
    )
    op.add_column(
        "experiment_runs",
        sa.Column("forge_build_id", sa.Text(), nullable=False, server_default="unknown"),
    )
    op.alter_column("experiment_runs", "forge_ai_profile", server_default=None)
    op.alter_column("experiment_runs", "forge_build_id", server_default=None)


def downgrade() -> None:
    op.drop_column("experiment_runs", "forge_build_id")
    op.drop_column("experiment_runs", "forge_ai_profile")
