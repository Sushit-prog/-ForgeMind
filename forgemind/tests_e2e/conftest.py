"""End-to-end test fixtures: real Postgres + Redis + worker subprocesses.

This directory is intentionally OUTSIDE ``tests/``: its conftest must set
DATABASE_URL/REDIS_URL *before* any ``app`` module is imported, while the
hermetic suite (``tests/``) binds ``app`` to SQLite. Run them separately:

    pytest tests/          # hermetic, no services needed
    pytest tests_e2e/      # needs docker-compose up (Postgres + Redis)

If the services are unreachable every test here skips with a clear message.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

os.environ["DATABASE_URL"] = "postgresql+psycopg://forgemind:forgemind@localhost:5433/forgemind"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["ENVIRONMENT"] = "test"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["QUEUE_ENABLED"] = "true"
os.environ["WORKER_SWEEP_ENABLED"] = "true"
os.environ["REPO_CACHE_DIR"] = tempfile.mkdtemp(prefix="forgemind-e2e-cache-")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.database.session import SessionLocal, engine  # noqa: E402
from app.main import create_app  # noqa: E402


def _services_reachable() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        with socket.create_connection(("localhost", 6379), timeout=2):
            pass
        return True
    except Exception:  # noqa: BLE001 — any failure means skip
        return False


_SERVICES_UP = _services_reachable()

pytestmark = pytest.mark.skipif(
    not _SERVICES_UP,
    reason="Postgres (5433) / Redis (6379) not reachable — run `docker compose up -d --wait` first",
)


@pytest.fixture(scope="session", autouse=True)
def _apply_migrations():
    """Bring the e2e schema to head once per session (driven by env.py)."""
    if not _SERVICES_UP:
        yield
        return
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    yield


@pytest.fixture()
def client():
    """TestClient bound to Postgres, with a clean schema + queue per test.

    Redis is flushed too: failed jobs (agent errors schedule arq retries with
    backoff) would otherwise linger across tests, and the startup sweep would
    re-enqueue them — a crash-flag worker could then die processing a STALE
    job instead of the test's own task."""
    _truncate_all()
    _flush_redis()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    _truncate_all()
    _flush_redis()


@pytest.fixture()
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def source_repo(tmp_path):
    """A real throwaway git repo (Phase 4 git-runtime tests).

    Since Phase 8, TESTING runs the repo's REAL configured test command
    against the worktree, so the fixture carries a genuine pytest suite:
    a test that FAILS on the initial state (VALUE = 1) and PASSES after the
    stub developer writes VALUE = 2 — the full pipeline can then reach
    COMPLETED through a real subprocess test run. ``pyproject.toml`` is the
    detection marker that makes discovery store ``test_command = pytest``.
    """
    from app.git.runner import run_git

    repo = tmp_path / "source"
    repo.mkdir()
    run_git(repo, "init", "-b", "main")
    (repo / "README.md").write_text("# Fixture Repo\n")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("VALUE = 1\n")
    (repo / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = [\"tests\"]\n"
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_app.py").write_text(
        "from pathlib import Path\n\n\n"
        "def test_value_is_two() -> None:\n"
        "    content = Path(\"src/app.py\").read_text()\n"
        "    assert \"VALUE = 2\" in content\n"
    )
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-m", "initial commit")
    return repo


@pytest.fixture()
def repo_task(db_session, source_repo):
    """A Repository + Task row pointing at ``source_repo`` (Postgres)."""
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


def _truncate_all() -> None:
    """Wipe the task-domain tables (CASCADE handles all FKs)."""
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE repositories, tasks, execution_events, audit_logs, "
                "tool_calls, worktrees RESTART IDENTITY CASCADE"
            )
        )


def _flush_redis() -> None:
    """Drop every arq job key (queued + retry state) so tests start with a
    clean queue. The worker subprocesses connect via the same Redis."""
    import asyncio

    from app.worker.queue import get_redis_settings

    async def _flush() -> None:
        import redis.asyncio as aioredis

        settings = get_redis_settings()
        client = aioredis.Redis(
            host=settings.host, port=settings.port, db=settings.database
        )
        try:
            await client.flushdb()
        finally:
            await client.aclose()

    asyncio.run(_flush())


def wait_for(predicate, timeout: float = 30.0, interval: float = 0.1) -> bool:
    """Poll ``predicate`` until truthy or ``timeout`` seconds elapse."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except Exception:  # noqa: BLE001 — transient errors during startup
            pass
        time.sleep(interval)
    return False


def spawn_worker(env_extra: dict[str, str] | None = None) -> subprocess.Popen:
    """Start a worker subprocess against the e2e Postgres/Redis.

    Defaults to the stub LLM provider (no API key needed); pass
    ``FORGEMIND_MOCK_LLM_FLAKY=1`` to exercise the planner retry path.
    """
    env = os.environ.copy()
    env.setdefault("FORGEMIND_MOCK_LLM", "1")
    if env_extra:
        env.update(env_extra)
    return subprocess.Popen(
        [sys.executable, "-m", "app.worker.worker"],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
