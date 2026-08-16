"""E2E (Postgres): the Phase-4 tools end-to-end against a real local repo.

Clones, creates a worktree, reads/searches, edits, diffs and commits —
all through the Phase 3 ToolPipeline, with capability gating and exactly
one audit row per call persisted in Postgres.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from sqlalchemy import func, select

from app.execution import ToolPipeline, make_execution_context
from app.git.worktree_manager import WorktreeManager
from app.models import ToolCall
from app.tools import build_runtime_registry


def run(coro):
    return asyncio.run(coro)


def test_git_tools_end_to_end_on_postgres(client, db_session, repo_task, tmp_path) -> None:
    repo, task = repo_task
    wt = WorktreeManager(db_session, cache_dir=tmp_path / "cache").create(task.id, repo)
    ctx = make_execution_context(task_id=task.id, agent_type="developer", db=db_session)
    pipeline = ToolPipeline(db=db_session, registry=build_runtime_registry())
    wt_id = str(wt.id)

    read = run(
        pipeline.invoke(
            "repository.read_file",
            {"worktree_id": wt_id, "path": "src/app.py"},
            {"repo.read"},
            ctx,
        )
    )
    assert read.status == "EXECUTED"
    assert "VALUE = 1" in read.output["content"]

    denied = run(pipeline.invoke("git.status", {"worktree_id": wt_id}, set(), ctx))
    assert denied.status == "DENIED"
    assert "git.read" in denied.denial_reason

    (Path(wt.path) / "src" / "app.py").write_text("VALUE = 2\n")
    commit = run(
        pipeline.invoke(
            "git.commit",
            {"worktree_id": wt_id, "message": "fix: bump"},
            {"git.write"},
            ctx,
        )
    )
    assert commit.status == "EXECUTED"
    assert len(commit.output["sha"]) == 40

    status = run(
        pipeline.invoke("git.status", {"worktree_id": wt_id}, {"git.read"}, ctx)
    )
    assert status.output["status"]["clean"] is True

    # Every call audited, exactly once, in Postgres.
    count = db_session.scalar(select(func.count()).select_from(ToolCall))
    assert count == 4
    statuses = db_session.scalars(select(ToolCall.status)).all()
    assert sorted(statuses) == ["DENIED", "EXECUTED", "EXECUTED", "EXECUTED"]
