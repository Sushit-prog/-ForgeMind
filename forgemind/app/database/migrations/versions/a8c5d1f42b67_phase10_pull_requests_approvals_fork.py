"""Phase 10: pull_requests + approvals tables, repositories.fork_url, tasks.issue_number

Revision ID: a8c5d1f42b67
Revises: f4a7c19e2b56
Create Date: 2026-08-19

Phase 10 — GitHub Agent + PR runtime:

- ``repositories.fork_url`` (nullable): the fork ForgeMind pushes to and
  opens PRs against; ``repositories.url`` stays the upstream reference.
  Unset -> git.push / github.create_pr fail closed, never fall back.
- ``tasks.issue_number`` (nullable): the GitHub issue a task originated from
  (target of github.get_issue reads + the PR-comment link).
- ``pull_requests``: the draft PR created on the fork (Section-G table).
- ``approvals``: the HUMAN approve/reject checkpoint record (Section-G table).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "a8c5d1f42b67"
down_revision = "f4a7c19e2b56"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "repositories",
        sa.Column("fork_url", sa.String(length=2048), nullable=True),
    )
    op.add_column(
        "tasks",
        sa.Column("issue_number", sa.Integer(), nullable=True),
    )
    op.create_table(
        "pull_requests",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("task_id", sa.Uuid(), nullable=False),
        sa.Column("repo", sa.String(length=2048), nullable=False),
        sa.Column("branch", sa.String(length=255), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
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
        "ix_pull_requests_task_id", "pull_requests", ["task_id"], unique=False
    )
    op.create_index(
        "ix_pull_requests_repo_branch",
        "pull_requests",
        ["repo", "branch"],
        unique=False,
    )
    op.create_table(
        "approvals",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("task_id", sa.Uuid(), nullable=False),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_approvals_task_id", "approvals", ["task_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_approvals_task_id", table_name="approvals")
    op.drop_table("approvals")
    op.drop_index("ix_pull_requests_repo_branch", table_name="pull_requests")
    op.drop_index("ix_pull_requests_task_id", table_name="pull_requests")
    op.drop_table("pull_requests")
    op.drop_column("tasks", "issue_number")
    op.drop_column("repositories", "fork_url")
