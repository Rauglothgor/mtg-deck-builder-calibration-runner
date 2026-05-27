"""Add experiment run status and retry count.

Revision ID: 20260519_0002
Revises: 20260518_0001
Create Date: 2026-05-19 19:20:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260519_0002"
down_revision = "20260518_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "experiment_runs",
        sa.Column("status", sa.Text(), nullable=False, server_default="pending"),
    )
    op.add_column(
        "experiment_runs",
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.alter_column("experiment_runs", "status", server_default=None)
    op.alter_column("experiment_runs", "retry_count", server_default=None)


def downgrade() -> None:
    op.drop_column("experiment_runs", "retry_count")
    op.drop_column("experiment_runs", "status")
