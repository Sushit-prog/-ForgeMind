"""Phase 13: repositories.install_command (dependency provisioning)

Revision ID: d4e5f6a7b8c9
Revises: a1b2c3d4e5f6
Create Date: 2026-09-24 00:00:00.000000

Phase 13 — per-task venv + dependency provisioning:

- ``repositories.install_command`` (nullable Text): the dependency-install
  command detected + validated at discovery time (Phase 13). ``shell.install_deps``
  runs it inside the task's own venv; when NULL provisioning cleanly skips
  (the suite runs on the worker environment). Nothing else changes: venv
  paths are derived from repository/task ids, never persisted, and install
  output rides the existing ``tool_calls`` audit rows.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "d4e5f6a7b8c9"
down_revision: str = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "repositories",
        sa.Column("install_command", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("repositories", "install_command")