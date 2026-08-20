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
# Phase 10.5: a known bearer token for the mutating routes. The ``client``
# fixture injects it by default so positive-path tests stay unchanged; the
# auth test module exercises the 401 paths via a bare client / per-request
# override.
os.environ["FORGEMIND_API_TOKEN"] = "test-token"
# Hermetic unit tests: the API must not touch Redis. Tasks stay CREATED and
# the worker's startup sweep would collect them if a queue existed.
os.environ["QUEUE_ENABLED"] = "false"
# Phase 4: keep clones/worktrees out of the repo tree — a throwaway dir.
os.environ["REPO_CACHE_DIR"] = tempfile.mkdtemp(prefix="forgemind-test-cache-")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.database.session import SessionLocal, engine  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models import Base  # noqa: E402


@pytest.fixture()
def client():
    """TestClient with a fresh schema per test; lifespan runs the DB check.

    Defaults the ``Authorization: Bearer`` header from the configured token
    so every existing mutating-route test keeps working unchanged. Override
    per-request (e.g. ``headers={"Authorization": "Bearer wrong"}``) to
    exercise the 401 paths.
    """
    Base.metadata.create_all(engine)
    app = create_app()
    token = get_settings().api_token
    with TestClient(app, headers={"Authorization": f"Bearer {token}"}) as test_client:
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


@pytest.fixture()
def source_repo(tmp_path):
    """A real throwaway git repo (Phase 4 git-runtime tests)."""
    from pathlib import Path

    from app.git.runner import run_git

    repo = tmp_path / "source"
    repo.mkdir()
    run_git(repo, "init", "-b", "main")
    (repo / "README.md").write_text("# Fixture Repo\n")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("VALUE = 1\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_app.py").write_text("def test_v():\n    assert True\n")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-m", "initial commit")
    return repo


@pytest.fixture()
def repo_task(db_session, source_repo):
    """A Repository + Task row pointing at ``source_repo``."""
    from app.models import Repository, Task

    repo = Repository(url=str(source_repo), default_branch="main")
    db_session.add(repo)
    db_session.flush()
    task = Task(objective="fix a bug", repository_id=repo.id)
    db_session.add(task)
    db_session.commit()
    db_session.refresh(repo)
    db_session.refresh(task)
    return repo, task
