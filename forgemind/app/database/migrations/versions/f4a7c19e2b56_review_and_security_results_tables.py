"""review_results + security_results tables (Phase 9)

Revision ID: f4a7c19e2b56
Revises: 9e3f2c81d4a7
Create Date: 2026-08-17

The Reviewer Agent's decision (review_results) and the Security Agent's
checklist verdict (security_results) — Section G tables, Phase 9.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "f4a7c19e2b56"
down_revision = "9e3f2c81d4a7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "review_results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("task_id", sa.Uuid(), nullable=False),
        sa.Column("commit_sha", sa.String(length=64), nullable=False),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column('issues', postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), 'sqlite'), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_review_results_task_id", "review_results", ["task_id"], unique=False
    )
    op.create_table(
        "security_results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("task_id", sa.Uuid(), nullable=False),
        sa.Column("commit_sha", sa.String(length=64), nullable=False),
        sa.Column("decision", sa.String(length=16), nullable=False),
        sa.Column('findings', postgresql.JSONB(astext_type=sa.Text()).with_variant(sa.JSON(), 'sqlite'), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_security_results_task_id", "security_results", ["task_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_security_results_task_id", table_name="security_results")
    op.drop_table("security_results")
    op.drop_index("ix_review_results_task_id", table_name="review_results")
    op.drop_table("review_results")
