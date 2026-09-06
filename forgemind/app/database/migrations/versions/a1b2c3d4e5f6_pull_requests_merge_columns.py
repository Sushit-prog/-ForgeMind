"""Phase 12: pull_requests merge columns (base_sha, merged_at, merge_commit_sha)

Revision ID: a1b2c3d4e5f6
Revises: c4a91f7b2d30
Create Date: 2026-09-06 00:00:00.000000

Phase 12 — Merge capability + minimal action UI:

- ``pull_requests.base_sha`` (nullable String(64)): the base branch SHA
  captured when the PR was created. The pre-merge staleness check compares
  it against the PR's current base SHA (mirroring the VERIFICATION pattern).
- ``pull_requests.merged_at`` (nullable DateTime(tz)): set atomically by a
  successful squash merge via ``github.merge_pr``. A task can be COMPLETED
  with this unset (approved, never merged) or set (approved, then merged).
- ``pull_requests.merge_commit_sha`` (nullable String(64)): the merge commit
  SHA returned by GitHub's merge API response.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: str = "c4a91f7b2d30"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "pull_requests",
        sa.Column("base_sha", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "pull_requests",
        sa.Column("merged_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "pull_requests",
        sa.Column("merge_commit_sha", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("pull_requests", "merge_commit_sha")
    op.drop_column("pull_requests", "merged_at")
    op.drop_column("pull_requests", "base_sha")
