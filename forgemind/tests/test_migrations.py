"""Verify the migration chain runs cleanly on a fresh database.

Uses a fresh file-based SQLite DB so no Postgres is required. The migration
itself is the same one used for Postgres (JSONB falls back to JSON).

The URL is passed to Alembic DIRECTLY via ``Config.set_main_option`` —
``migrations/env.py`` now honors an explicitly-set ``sqlalchemy.url`` instead
of clobbering it from ``get_settings()``. This test therefore NEVER mutates
the process environment or clears the ``get_settings`` cache, so it cannot
leak a swapped ``DATABASE_URL`` into the rest of the suite (previously the
suite's only process-global env mutation — the suspected cross-test leak).
"""

from __future__ import annotations

import os
import tempfile

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

EXPECTED_TABLES = {
    # (unchanged from before)
    "tasks",
    "plans",
    "plan_steps",
    "task_steps",
    "capabilities",
    "policies",
    "audit_logs",
    "repositories",
    "worktrees",
    "research_artifacts",
    "implementation_summaries",
    "test_runs",
    "failures",
    "failure_classifications",
    "review_results",
    "security_results",
    "pull_requests",
    "approvals",
}


def test_migrations_upgrade_fresh_db() -> None:
    fd, path = tempfile.mkstemp(suffix=".db")
    url = f"sqlite:///{path.replace(os.sep, '/')}"
    try:
        cfg = Config("alembic.ini")
        cfg.set_main_option("sqlalchemy.url", url)
        command.upgrade(cfg, "head")

        engine = create_engine(url)
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        engine.dispose()

        assert EXPECTED_TABLES <= tables, f"missing tables: {EXPECTED_TABLES - tables}"
        assert "alembic_version" in tables

        # Alembic runs migrations once: upgrading again is a no-op.
        command.upgrade(cfg, "head")
    finally:
        os.close(fd)
        os.unlink(path)
