"""Test fixtures.

The app is pointed at a throwaway file-based SQLite DB *before* any app
module is imported, so the module-level engine binds to it. JSONB columns
fall back to plain JSON on SQLite, so no Postgres is needed for tests.
"""

from __future__ import annotations

import os
import tempfile

_TMP_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_TMP_DB.close()

# Forward slashes: SQLAlchemy on Windows chokes on backslashes in sqlite URLs.
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB.name.replace(os.sep, '/')}"
os.environ["ENVIRONMENT"] = "test"
os.environ["LOG_LEVEL"] = "WARNING"
# Hermetic unit tests: the API must not touch Redis. Tasks stay CREATED and
# the worker's startup sweep would collect them if a queue existed.
os.environ["QUEUE_ENABLED"] = "false"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.database.session import SessionLocal, engine  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models import Base  # noqa: E402


@pytest.fixture()
def client():
    """TestClient with a fresh schema per test; lifespan runs the DB check."""
    Base.metadata.create_all(engine)
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    Base.metadata.drop_all(engine)


@pytest.fixture()
def db_session():
    """Direct session for asserting on persisted rows (e.g. audit log).

    Creates the schema itself so it works standalone (state-machine/lifecycle
    tests never touch the API client). Idempotent when paired with ``client``.
    """
    Base.metadata.create_all(engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
