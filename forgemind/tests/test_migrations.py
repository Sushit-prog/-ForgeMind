"""Verify the migration chain runs cleanly on a fresh database.

Uses a fresh file-based SQLite DB so no Postgres is required. The migration
itself is the same one used for Postgres (JSONB falls back to JSON).
"""

from __future__ import annotations

import os
import tempfile

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from app.config import get_settings

EXPECTED_TABLES = {
    "tasks",
    "plans",
    "plan_steps",
    "task_steps",
    "capabilities",
    "policies",
    "audit_logs",
    "repositories",
    "worktrees",
}


def test_migrations_upgrade_fresh_db() -> None:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    url = f"sqlite:///{path.replace(os.sep, '/')}"
    original_url = get_settings().database_url
    try:
        # Point Alembic at the fresh DB (env.py reads settings at import).
        os.environ["DATABASE_URL"] = url
        get_settings.cache_clear()

        cfg = Config("alembic.ini")
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
        os.environ["DATABASE_URL"] = original_url
        get_settings.cache_clear()
        os.unlink(path)
