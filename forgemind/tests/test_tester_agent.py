"""Test Agent integration tests (Phase 8, architecture doc sections 9/41/E).

The Test Agent is the ONE agent with no LLM call at all: one real
``shell.run_test`` subprocess against the repository's configured test
command, parsed deterministically (exit code + structured parser), persisted
as a ``TestRun``. These tests run REAL pytest subprocesses against real
fixture repos — the ``error`` status (timeout / no tests / no command) is
verified distinct from a clean ``failed`` exit code.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from sqlalchemy import select

from app.agents.tester.agent import TestAgent
from app.agents.tester.schema import TestResult, parse_test_run
from app.git.runner import run_git
from app.git.worktree_manager import WorktreeManager
from app.models import Failure, Repository, Task, TestRun
from app.tools.base import ExecutionContext


def run(coro):
    return asyncio.run(coro)


def make_test_repo(tmp_path, *, tests: str, config: str | None = None) -> Path:
    """A real git repo with the given pytest test file content."""
    repo = tmp_path / "testrepo"
    repo.mkdir()
    run_git(repo, "init", "-b", "main")
    if config is not None:
        (repo / "pyproject.toml").write_text(config)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_suite.py").write_text(tests)
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-m", "initial")
    return repo


def create_worktree(db_session, repo: Repository, task: Task):
    """Clone + create the task's worktree (runs discovery, which detects the
    test command from pyproject.toml)."""
    return WorktreeManager(db_session).create(task.id, repo)


def repo_and_task(db_session, repo_path: Path) -> tuple[Repository, Task]:
    repo = Repository(url=str(repo_path), default_branch="main")
    db_session.add(repo)
    db_session.flush()
    task = Task(objective="run tests", repository_id=repo.id)
    db_session.add(task)
    db_session.commit()
    db_session.refresh(repo)
    db_session.refresh(task)
    return repo, task


def ctx_for(task: Task, db_session) -> ExecutionContext:
    return ExecutionContext(task_id=task.id, agent_type="tester", db=db_session)


def test_passing_suite_reports_passed(db_session, tmp_path) -> None:
    repo_path = make_test_repo(
        tmp_path,
        config="[tool.pytest.ini_options]\ntestpaths = ['tests']\n",
        tests="def test_ok():\n    assert 1 + 1 == 2\n",
    )
    repo, task = repo_and_task(db_session, repo_path)
    worktree = create_worktree(db_session, repo, task)

    result = run(TestAgent().run(task, worktree, ctx_for(task, db_session)))

    assert isinstance(result, TestResult)
    assert result.status == "passed"
    assert result.exit_code == 0
    assert result.passed == 1
    assert result.failed == 0
    assert result.failures == []

    # Persisted: one TestRun row, no Failure rows.
    runs = db_session.scalars(
        select(TestRun).where(TestRun.task_id == task.id)
    ).all()
    assert len(runs) == 1
    assert runs[0].status == "passed"
    assert runs[0].exit_code == 0
    assert runs[0].worktree_id == worktree.id


def test_failing_suite_reports_failed_with_failures(db_session, tmp_path) -> None:
    repo_path = make_test_repo(
        tmp_path,
        config="[tool.pytest.ini_options]\ntestpaths = ['tests']\n",
        tests=(
            "def test_ok():\n    assert True\n"
            "def test_bad():\n    assert 1 == 2, 'boom'\n"
        ),
    )
    repo, task = repo_and_task(db_session, repo_path)
    worktree = create_worktree(db_session, repo, task)

    result = run(TestAgent().run(task, worktree, ctx_for(task, db_session)))

    assert result.status == "failed"
    assert result.exit_code != 0
    assert result.passed == 1
    assert result.failed == 1
    assert any("test_bad" in f.test for f in result.failures)

    # Persisted: TestRun + the parsed Failure row.
    runs = db_session.scalars(
        select(TestRun).where(TestRun.task_id == task.id)
    ).all()
    assert len(runs) == 1
    assert runs[0].status == "failed"
    failures = db_session.scalars(
        select(Failure).where(Failure.test_run_id == runs[0].id)
    ).all()
    assert len(failures) == 1
    assert "test_bad" in failures[0].test


def test_no_test_command_is_error_not_failed(db_session, tmp_path) -> None:
    """A repo with no pytest setup: discovery stores no test_command, the
    runner refuses clearly, and the status is ``error`` — distinct from a
    clean failing exit code, so the Debugger can tell them apart."""
    repo_path = tmp_path / "norepo"
    repo_path.mkdir()
    run_git(repo_path, "init", "-b", "main")
    (repo_path / "README.md").write_text("no tests here\n")
    run_git(repo_path, "add", "-A")
    run_git(repo_path, "commit", "-m", "initial")
    repo, task = repo_and_task(db_session, repo_path)
    worktree = create_worktree(db_session, repo, task)

    result = run(TestAgent().run(task, worktree, ctx_for(task, db_session)))

    assert result.status == "error"
    assert result.exit_code is None
    runs = db_session.scalars(
        select(TestRun).where(TestRun.task_id == task.id)
    ).all()
    assert runs[0].status == "error"


def test_timeout_is_error_not_failed(db_session, tmp_path) -> None:
    """A hung suite times out into ``error`` — never ``failed`` (Section 41:
    the raw signal must distinguish a hang from a clean failing exit code)."""
    repo_path = make_test_repo(
        tmp_path,
        config="[tool.pytest.ini_options]\ntestpaths = ['tests']\n",
        tests="import time\n\ndef test_hang():\n    time.sleep(30)\n",
    )
    repo, task = repo_and_task(db_session, repo_path)
    worktree = create_worktree(db_session, repo, task)

    # Small timeout so the test actually trips it. ``shell_tools`` imports
    # get_settings lazily, so patch the source in ``app.config``.
    from unittest.mock import patch

    with patch("app.config.get_settings") as mock_settings:
        mock_settings.return_value.test_timeout_seconds = 1
        result = run(TestAgent().run(task, worktree, ctx_for(task, db_session)))

    assert result.status == "error"
    assert result.exit_code is None
    runs = db_session.scalars(
        select(TestRun).where(TestRun.task_id == task.id)
    ).all()
    assert runs[0].status == "error"
    assert runs[0].timed_out is True


def test_parse_test_run_deterministic() -> None:
    """The parser decides from exit code + output — the Section-41 principle,
    tested directly."""
    result = parse_test_run(
        exit_code=0, output="=== 3 passed in 0.5s ===", timed_out=False
    )
    assert result.status == "passed"
    failed_output = (
        "=== 1 failed, 2 passed in 0.6s ===\n"
        "FAILED tests/test_s.py::test_x - AssertionError: nope"
    )
    failed = parse_test_run(
        exit_code=1, output=failed_output, timed_out=False
    )
    assert failed.status == "failed"
    assert failed.failed == 1
    assert failed.passed == 2
    assert failed.failures[0].test == "tests/test_s.py::test_x"
    # No tests collected (exit 5) is an error, not a failure.
    empty = parse_test_run(exit_code=5, output="no tests ran", timed_out=False)
    assert empty.status == "error"
    # A hang is an error with no exit code.
    hung = parse_test_run(exit_code=None, output="", timed_out=True)
    assert hung.status == "error"
    assert hung.exit_code is None


def test_tester_requires_db() -> None:
    from app.agents.tester.agent import TestError

    task = Task(objective="x")
    try:
        run(TestAgent().run(task, None, ExecutionContext(task_id=uuid.uuid4())))
    except TestError:
        return
    raise AssertionError("TestAgent without a DB must raise TestError")
