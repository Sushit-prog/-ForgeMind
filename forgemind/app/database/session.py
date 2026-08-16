"""SQLAlchemy engine and session management.

- Engine is built from ``settings.database_url`` (local dev: docker-compose
  Postgres; prod: Supabase Postgres).
- ``check_database_connection`` pings the DB and raises on failure so the
  app fails fast at startup with a clear log message instead of hanging or
  erroring mid-request.
- ``get_db`` is the FastAPI dependency yielding a request-scoped session.
"""

from __future__ import annotations

import logging
from collections.abc import Generator

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings
from app.logging import redact_url

logger = logging.getLogger(__name__)


def build_engine(settings: Settings | None = None) -> Engine:
    """Create the SQLAlchemy engine for the given (or default) settings."""
    settings = settings or get_settings()
    connect_args: dict[str, object] = {}
    if settings.database_url.startswith("sqlite"):
        # SQLite (tests) doesn't support the connect timeout kwarg.
        connect_args = {"check_same_thread": False}
    else:
        # Bound the connect attempt so a dead/unreachable DB fails fast
        # instead of hanging on the OS-level TCP timeout.
        connect_args = {"connect_timeout": settings.db_connect_timeout_seconds}
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        connect_args=connect_args,
    )


engine = build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def check_database_connection(settings: Settings | None = None) -> None:
    """Ping the database; raise if unreachable (fail fast, never hang silently).

    Raises:
        RuntimeError: if the database cannot be reached within the configured
            timeout. The URL is logged redacted — credentials never appear.
    """
    settings = settings or get_settings()
    eng = build_engine(settings)
    try:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection OK (%s)", redact_url(settings.database_url))
    except Exception as exc:  # noqa: BLE001 — any DB error means fail fast.
        logger.error(
            "Database connection FAILED (%s): %s",
            redact_url(settings.database_url),
            exc,
        )
        raise RuntimeError(
            "Cannot connect to the database at startup — refusing to start. "
            "Check DATABASE_URL and that the database is running."
        ) from exc
    finally:
        eng.dispose()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yield a session, always close it afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
