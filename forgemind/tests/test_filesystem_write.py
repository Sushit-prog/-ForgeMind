"""``filesystem.write_file`` through the Phase 3 pipeline (Phase 7).

The write path is NEW attack surface (write, not just read): a traversal
attempt must be rejected by the exact same containment check as reads, and
nothing may be written outside the worktree root. The tool reuses
``FileAccess._resolve`` — it does not reimplement containment with a subtly
different check — and this adversarial test proves the write side is blocked
independently of the Phase 4 read-side test.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select

from app.execution import ToolPipeline, make_execution_context
from app.git.worktree_manager import WorktreeManager
from app.models import ToolCall
from app.tools import build_runtime_registry


def run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def worktree_env(db_session, repo_task, tmp_path):
    """A real worktree for the fixture repo, ready for tool calls."""
    repo, task = repo_task
    manager = WorktreeManager(db_session, cache_dir=tmp_path / "cache")
    wt = manager.create(task.id, repo)
    ctx = make_execution_context(task_id=task.id, agent_type="developer", db=db_session)
    pipeline = ToolPipeline(db=db_session, registry=build_runtime_registry())
    return {
        "pipeline": pipeline,
        "ctx": ctx,
        "worktree_id": wt.id,
        "wt_path": Path(wt.path),
        "task_id": task.id,
    }


def rows_for(db_session, tool_name: str) -> list[ToolCall]:
    return list(
        db_session.scalars(select(ToolCall).where(ToolCall.tool_name == tool_name))
    )


# --- capability gating -------------------------------------------------------

def test_write_file_denied_without_repo_write(worktree_env, db_session) -> None:
    result = run(
        worktree_env["pipeline"].invoke(
            "filesystem.write_file",
            {"worktree_id": str(worktree_env["worktree_id"]), "path": "src/app.py", "content": "x"},
            set(),
            worktree_env["ctx"],
        )
    )
    assert result.status == "DENIED"
    assert "repo.write" in result.denial_reason
    rows = rows_for(db_session, "filesystem.write_file")
    assert len(rows) == 1
    assert rows[0].status == "DENIED"


def test_write_file_denied_without_git_write_is_irrelevant(worktree_env, db_session) -> None:
    """write_file needs repo.write only; git.write is not required."""
    result = run(
        worktree_env["pipeline"].invoke(
            "filesystem.write_file",
            {"worktree_id": str(worktree_env["worktree_id"]), "path": "src/app.py", "content": "x"},
            {"repo.write"},
            worktree_env["ctx"],
        )
    )
    assert result.status == "EXECUTED"


# --- happy path: modify existing + create new ---------------------------------

def test_write_file_modifies_existing(worktree_env, db_session) -> None:
    result = run(
        worktree_env["pipeline"].invoke(
            "filesystem.write_file",
            {
                "worktree_id": str(worktree_env["worktree_id"]),
                "path": "src/app.py",
                "content": "VALUE = 2\n",
            },
            {"repo.write"},
            worktree_env["ctx"],
        )
    )
    assert result.status == "EXECUTED"
    assert result.output == {"path": "src/app.py", "existed": True}
    assert (worktree_env["wt_path"] / "src" / "app.py").read_text() == "VALUE = 2\n"
    rows = rows_for(db_session, "filesystem.write_file")
    assert len(rows) == 1
    assert rows[0].status == "EXECUTED"
    assert rows[0].risk == "MEDIUM"


def test_write_file_creates_new_file_with_parents(worktree_env, db_session) -> None:
    result = run(
        worktree_env["pipeline"].invoke(
            "filesystem.write_file",
            {
                "worktree_id": str(worktree_env["worktree_id"]),
                "path": "src/utils/helper.py",
                "content": "def h():\n    pass\n",
            },
            {"repo.write"},
            worktree_env["ctx"],
        )
    )
    assert result.status == "EXECUTED"
    assert result.output == {"path": "src/utils/helper.py", "existed": False}
    assert (worktree_env["wt_path"] / "src" / "utils" / "helper.py").is_file()


# --- security: the write-path traversal defense ------------------------------

def test_write_path_traversal_rejected_and_nothing_written(
    worktree_env, db_session, tmp_path
) -> None:
    """The write-side attack: ../../evil.py must be rejected BEFORE anything
    is written, proven independently of the read-side traversal test."""
    result = run(
        worktree_env["pipeline"].invoke(
            "filesystem.write_file",
            {
                "worktree_id": str(worktree_env["worktree_id"]),
                "path": "../../evil.py",
                "content": "import os; os.system('rm -rf /')\n",
            },
            {"repo.write"},
            worktree_env["ctx"],
        )
    )
    assert result.status == "FAILED"
    assert "escape" in result.error.lower() or "worktree root" in result.error.lower()
    # Nothing named evil.py exists anywhere under the test cache — the write
    # never touched the filesystem outside the worktree root.
    assert list(tmp_path.rglob("evil.py")) == []
    rows = rows_for(db_session, "filesystem.write_file")
    assert len(rows) == 1
    assert rows[0].status == "FAILED"
    assert rows[0].output["error"] == result.error


def test_write_absolute_path_rejected(worktree_env, db_session) -> None:
    result = run(
        worktree_env["pipeline"].invoke(
            "filesystem.write_file",
            {
                "worktree_id": str(worktree_env["worktree_id"]),
                "path": "/etc/passwd",
                "content": "pwned",
            },
            {"repo.write"},
            worktree_env["ctx"],
        )
    )
    assert result.status == "FAILED"
    assert "escape" in result.error.lower() or "worktree root" in result.error.lower()


def test_write_unknown_worktree_failed(worktree_env, db_session) -> None:
    result = run(
        worktree_env["pipeline"].invoke(
            "filesystem.write_file",
            {"worktree_id": str(uuid.uuid4()), "path": "src/app.py", "content": "x"},
            {"repo.write"},
            worktree_env["ctx"],
        )
    )
    assert result.status == "FAILED"
    assert "worktree" in result.error.lower()


def test_write_file_survives_commit_and_shows_in_diff(worktree_env, db_session) -> None:
    """End-to-end shape: write -> git.status shows it -> git.commit captures it."""
    run(
        worktree_env["pipeline"].invoke(
            "filesystem.write_file",
            {
                "worktree_id": str(worktree_env["worktree_id"]),
                "path": "src/app.py",
                "content": "VALUE = 42\n",
            },
            {"repo.write"},
            worktree_env["ctx"],
        )
    )
    status = run(
        worktree_env["pipeline"].invoke(
            "git.status",
            {"worktree_id": str(worktree_env["worktree_id"])},
            {"git.read"},
            worktree_env["ctx"],
        )
    )
    assert status.status == "EXECUTED"
    assert status.output["status"]["unstaged"] == ["src/app.py"]

    commit = run(
        worktree_env["pipeline"].invoke(
            "git.commit",
            {"worktree_id": str(worktree_env["worktree_id"]), "message": "fix: 42"},
            {"git.write"},
            worktree_env["ctx"],
        )
    )
    assert commit.status == "EXECUTED"
    assert len(commit.output["sha"]) == 40
