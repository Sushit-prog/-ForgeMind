"""Phase 4 tools through the Phase 3 pipeline.

Proves capability gating (repo.read / git.read / git.write), correct audit
rows, and that a path-traversal attempt surfaces as a FAILED tool call —
never a read — with the denial/error recorded.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.execution import ToolPipeline, make_execution_context
from app.git.worktree_manager import WorktreeManager
from app.models import Repository, Task, ToolCall
from app.tools import build_runtime_registry


def run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def worktree_env(db_session, repo_task, tmp_path):
    """A real worktree for the fixture repo, ready for tool calls."""
    repo, task = repo_task
    manager = WorktreeManager(db_session, cache_dir=tmp_path / "cache")
    wt = manager.create(task.id, repo)
    ctx = make_execution_context(
        task_id=task.id, agent_type="developer", db=db_session
    )
    pipeline = ToolPipeline(db=db_session, registry=build_runtime_registry())
    return {
        "pipeline": pipeline,
        "ctx": ctx,
        "worktree_id": wt.id,
        "wt_path": Path(wt.path),
        "task_id": task.id,
    }


def count_rows(db_session) -> int:
    return db_session.scalar(select(func.count()).select_from(ToolCall))


def rows_for(db_session, tool_name: str) -> list[ToolCall]:
    return list(
        db_session.scalars(select(ToolCall).where(ToolCall.tool_name == tool_name))
    )


# --- repository.read_file ---------------------------------------------------

def test_read_file_denied_without_repo_read(worktree_env, db_session) -> None:
    result = run(
        worktree_env["pipeline"].invoke(
            "repository.read_file",
            {"worktree_id": str(worktree_env["worktree_id"]), "path": "README.md"},
            set(),
            worktree_env["ctx"],
        )
    )
    assert result.status == "DENIED"
    assert "repo.read" in result.denial_reason
    rows = rows_for(db_session, "repository.read_file")
    assert len(rows) == 1
    assert rows[0].status == "DENIED"


def test_read_file_executes_with_repo_read(worktree_env, db_session) -> None:
    result = run(
        worktree_env["pipeline"].invoke(
            "repository.read_file",
            {"worktree_id": str(worktree_env["worktree_id"]), "path": "src/app.py"},
            {"repo.read"},
            worktree_env["ctx"],
        )
    )
    assert result.status == "EXECUTED"
    assert "VALUE = 1" in result.output["content"]
    rows = rows_for(db_session, "repository.read_file")
    assert len(rows) == 1
    assert rows[0].status == "EXECUTED"
    assert rows[0].task_id == worktree_env["task_id"]


def test_read_file_traversal_is_failed_call_not_a_read(worktree_env, db_session) -> None:
    """../escaping input -> FAILED row; outside content never leaks."""
    result = run(
        worktree_env["pipeline"].invoke(
            "repository.read_file",
            {"worktree_id": str(worktree_env["worktree_id"]), "path": "../../secrets.env"},
            {"repo.read"},
            worktree_env["ctx"],
        )
    )
    assert result.status == "FAILED"
    assert "escape" in result.error.lower() or "worktree root" in result.error.lower()
    assert "TOP-SECRET" not in (result.error or "")
    rows = rows_for(db_session, "repository.read_file")
    assert len(rows) == 1
    assert rows[0].status == "FAILED"
    assert rows[0].output["error"] == result.error


def test_read_file_absolute_path_failed(worktree_env, db_session) -> None:
    result = run(
        worktree_env["pipeline"].invoke(
            "repository.read_file",
            {"worktree_id": str(worktree_env["worktree_id"]), "path": "/etc/passwd"},
            {"repo.read"},
            worktree_env["ctx"],
        )
    )
    assert result.status == "FAILED"


def test_read_file_unknown_worktree_failed(worktree_env, db_session) -> None:
    import uuid

    result = run(
        worktree_env["pipeline"].invoke(
            "repository.read_file",
            {"worktree_id": str(uuid.uuid4()), "path": "README.md"},
            {"repo.read"},
            worktree_env["ctx"],
        )
    )
    assert result.status == "FAILED"
    assert "worktree" in result.error.lower()


# --- repository.search / list_files -----------------------------------------

def test_search_executes(worktree_env, db_session) -> None:
    result = run(
        worktree_env["pipeline"].invoke(
            "repository.search",
            {"worktree_id": str(worktree_env["worktree_id"]), "query": "assert True"},
            {"repo.read"},
            worktree_env["ctx"],
        )
    )
    assert result.status == "EXECUTED"
    assert [m["path"] for m in result.output["matches"]] == ["tests/test_app.py"]


def test_list_files_executes(worktree_env, db_session) -> None:
    result = run(
        worktree_env["pipeline"].invoke(
            "repository.list_files",
            {"worktree_id": str(worktree_env["worktree_id"])},
            {"repo.read"},
            worktree_env["ctx"],
        )
    )
    assert result.status == "EXECUTED"
    assert result.output["files"] == ["README.md", "src/app.py", "tests/test_app.py"]


# --- git.* tools ------------------------------------------------------------

def test_git_status_requires_git_read(worktree_env, db_session) -> None:
    result = run(
        worktree_env["pipeline"].invoke(
            "git.status",
            {"worktree_id": str(worktree_env["worktree_id"])},
            set(),
            worktree_env["ctx"],
        )
    )
    assert result.status == "DENIED"
    assert "git.read" in result.denial_reason
    assert len(rows_for(db_session, "git.status")) == 1


def test_git_status_executes(worktree_env, db_session) -> None:
    result = run(
        worktree_env["pipeline"].invoke(
            "git.status",
            {"worktree_id": str(worktree_env["worktree_id"])},
            {"git.read"},
            worktree_env["ctx"],
        )
    )
    assert result.status == "EXECUTED"
    assert result.output["status"]["branch"] == f"agent/task-{worktree_env['task_id']}"
    assert result.output["status"]["clean"] is True


def test_git_diff_after_modification(worktree_env, db_session) -> None:
    (worktree_env["wt_path"] / "src" / "app.py").write_text("VALUE = 2\n")
    result = run(
        worktree_env["pipeline"].invoke(
            "git.diff",
            {"worktree_id": str(worktree_env["worktree_id"])},
            {"git.read"},
            worktree_env["ctx"],
        )
    )
    assert result.status == "EXECUTED"
    assert "VALUE = 2" in result.output["diff"]


def test_git_log_executes(worktree_env, db_session) -> None:
    result = run(
        worktree_env["pipeline"].invoke(
            "git.log",
            {"worktree_id": str(worktree_env["worktree_id"]), "limit": 5},
            {"git.read"},
            worktree_env["ctx"],
        )
    )
    assert result.status == "EXECUTED"
    assert result.output["commits"][0]["summary"] == "initial commit"


def test_git_create_branch_requires_git_write(worktree_env, db_session) -> None:
    result = run(
        worktree_env["pipeline"].invoke(
            "git.create_branch",
            {"worktree_id": str(worktree_env["worktree_id"]), "name": "feature/x"},
            {"git.read"},
            worktree_env["ctx"],
        )
    )
    assert result.status == "DENIED"
    assert "git.write" in result.denial_reason


def test_git_create_branch_executes(worktree_env, db_session) -> None:
    result = run(
        worktree_env["pipeline"].invoke(
            "git.create_branch",
            {"worktree_id": str(worktree_env["worktree_id"]), "name": "feature/abc"},
            {"git.write"},
            worktree_env["ctx"],
        )
    )
    assert result.status == "EXECUTED"
    assert result.output["branch"] == "feature/abc"


def test_git_commit_executes_and_audits(worktree_env, db_session) -> None:
    (worktree_env["wt_path"] / "src" / "app.py").write_text("VALUE = 42\n")
    result = run(
        worktree_env["pipeline"].invoke(
            "git.commit",
            {"worktree_id": str(worktree_env["worktree_id"]), "message": "fix: 42"},
            {"git.write"},
            worktree_env["ctx"],
        )
    )
    assert result.status == "EXECUTED"
    assert len(result.output["sha"]) == 40

    rows = rows_for(db_session, "git.commit")
    assert len(rows) == 1
    assert rows[0].status == "EXECUTED"
    assert rows[0].risk == "MEDIUM"


def test_git_commit_empty_worktree_is_failed_call(worktree_env, db_session) -> None:
    result = run(
        worktree_env["pipeline"].invoke(
            "git.commit",
            {"worktree_id": str(worktree_env["worktree_id"]), "message": "nothing to do"},
            {"git.write"},
            worktree_env["ctx"],
        )
    )
    assert result.status == "FAILED"
    assert "nothing to commit" in result.error
    assert rows_for(db_session, "git.commit")[0].status == "FAILED"


# --- audit guarantees -------------------------------------------------------

def test_every_git_call_writes_exactly_one_row(worktree_env, db_session) -> None:
    invocations = [
        ("git.status", {"git.read"}),
        ("git.log", {"git.read"}),
        ("repository.list_files", {"repo.read"}),
        ("git.status", set()),  # denied
    ]
    for tool, caps in invocations:
        result = run(
            worktree_env["pipeline"].invoke(
                tool, {"worktree_id": str(worktree_env["worktree_id"])}, caps, worktree_env["ctx"]
            )
        )
        assert result.status in ("EXECUTED", "DENIED")
    assert count_rows(db_session) == len(invocations)
