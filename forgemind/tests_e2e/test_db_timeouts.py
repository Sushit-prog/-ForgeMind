"""Fix 2 proof: DB-side lock_timeout bounds a stuck FOR UPDATE.

The deadlock's second ingredient: a sync `FOR UPDATE` blocked on another
session's row lock froze the worker's event loop with NO ceiling. These
tests prove Postgres itself now cancels the wait (lock timeout ->
QueryCanceled) within the configured ceiling — on the real Postgres stack.

Subtlety that makes this test honest: the probe ROW must be COMMITTED
before the holder takes its lock. An uncommitted insert is invisible to
the waiter under READ COMMITTED, so its FOR UPDATE would match nothing
and return instantly — proving nothing.
"""

from __future__ import annotations

import time
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from app.config import get_settings
from app.database.session import SessionLocal, engine


def _interval_seconds(conn, setting: str) -> float:
    raw = conn.execute(text(f"SHOW {setting}")).scalar()
    # psycopg already coerces '::interval' to datetime.timedelta.
    if hasattr(raw, "total_seconds"):
        return float(raw.total_seconds())
    return float(
        conn.execute(text(f"SELECT '{raw}'::interval")).scalar().total_seconds()
    )


def test_lock_timeouts_applied_per_connection() -> None:
    """Every pooled connection carries the configured ceilings."""
    settings = get_settings()
    with engine.connect() as conn:
        assert (
            _interval_seconds(conn, "lock_timeout") == settings.db_lock_timeout_seconds
        )
        assert (
            _interval_seconds(conn, "statement_timeout")
            == settings.db_statement_timeout_seconds
        )


def test_conflicting_for_update_cancels_within_lock_timeout() -> None:
    """Holder + conflicting FOR UPDATE -> QueryCanceled, not a hang."""
    setup = SessionLocal()
    obj_id = uuid.uuid4()
    try:
        # Committed probe row: visible to BOTH sessions.
        setup.execute(
            text("CREATE TABLE IF NOT EXISTS _fm_lock_probe (id UUID PRIMARY KEY)")
        )
        setup.execute(
            text("INSERT INTO _fm_lock_probe (id) VALUES (:i) ON CONFLICT DO NOTHING"),
            {"i": obj_id},
        )
        setup.commit()

        holder = SessionLocal()
        waiter = SessionLocal()
        try:
            # Holder takes the row lock in its OWN open transaction.
            holder.execute(
                text("SELECT id FROM _fm_lock_probe WHERE id = :i FOR UPDATE"),
                {"i": obj_id},
            )
            assert holder.in_transaction()

            started = time.monotonic()
            with pytest.raises(DBAPIError) as excinfo:
                waiter.execute(
                    text("SELECT id FROM _fm_lock_probe WHERE id = :i FOR UPDATE"),
                    {"i": obj_id},
                )
            elapsed = time.monotonic() - started

            assert "lock timeout" in str(excinfo.value.orig).lower()
            # 10s ceiling + generous CI slack — the old code hung FOREVER here.
            assert elapsed < 30, f"lock wait took {elapsed:.1f}s"
            assert holder.in_transaction(), "holder must still own the lock"
        finally:
            holder.rollback()
            holder.close()
            waiter.rollback()
            waiter.close()
    finally:
        setup.rollback()
        setup.close()
        cleanup = SessionLocal()
        try:
            cleanup.execute(text("DROP TABLE IF EXISTS _fm_lock_probe"))
            cleanup.commit()
        finally:
            cleanup.close()
