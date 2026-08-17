"""E2E (Postgres + real worker): IMPLEMENTING runs the real Developer Agent.

The task is submitted through the API against a real local git repo. The
worker (stub LLM provider, default per-schema script) drives PLANNING ->
RESEARCHING -> IMPLEMENTING, where the Developer Agent runs a real, bounded,
audited tool-use loop against the worktree: it writes src/app.py, commits it
ONCE on the task's branch (never main), and persists a grounded
ImplementationSummary before the task proceeds to TESTING and beyond.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy import select

from app.git.runner import run_git
from app.models import (
    ExecutionEvent,
    ImplementationSummary,
    Plan,
    ResearchArtifact,
    ToolCall,
)
from tests_e2e.conftest import spawn_worker, wait_for


def _file_url(repo: Path) -> str:
    """file:// URL for a local repo path (validator accepts file://)."""
    return "file:///" + str(repo).replace("\\", "/")


def test_worker_developer_loop_produces_commit_and_summary(
    client, db_session, source_repo
) -> None:
    main_head_before = run_git(source_repo, "rev-parse", "HEAD").stdout.strip()
    proc = spawn_worker()
    try:
        payload = {
            "objective": "Fix the VALUE bug",
            "repository_url": _file_url(source_repo),
        }
        created = client.post("/tasks", json=payload).json()
        task_id = uuid.UUID(created["id"])

        def completed() -> bool:
            return client.get(f"/tasks/{task_id}").json()["status"] == "COMPLETED"

        assert wait_for(completed, timeout=90)

        # The developer persisted a COMPLETE summary with a real commit sha.
        summaries = db_session.scalars(
            select(ImplementationSummary).where(ImplementationSummary.task_id == task_id)
        ).all()
        assert len(summaries) == 1
        summary = summaries[0]
        assert summary.status == "COMPLETE"
        assert summary.commit_sha and len(summary.commit_sha) == 40
        assert summary.files_changed == ["src/app.py"]

        # The tool-use loop ran REAL write + commit calls, audited.
        calls = db_session.scalars(
            select(ToolCall)
            .where(ToolCall.task_id == task_id, ToolCall.agent_type == "developer")
            .order_by(ToolCall.created_at)
        ).all()
        assert [c.tool_name for c in calls] == ["filesystem.write_file", "git.commit"]
        assert all(c.status == "EXECUTED" for c in calls)

        # IMPLEMENTING -> TESTING fired only after persistence.
        events = db_session.scalars(
            select(ExecutionEvent).where(ExecutionEvent.task_id == task_id)
        ).all()
        impl_events = [e for e in events if e.from_status == "IMPLEMENTING"]
        assert impl_events
        assert impl_events[0].to_status == "TESTING"
        assert impl_events[0].reason == "implementation_persisted"

        # Research and plan were persisted too (the pipeline ran in order).
        artifacts = db_session.scalars(
            select(ResearchArtifact).where(ResearchArtifact.task_id == task_id)
        ).all()
        assert len(artifacts) == 1
        plans = db_session.scalars(select(Plan).where(Plan.task_id == task_id)).all()
        assert len(plans) == 1
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    # The developer committed on the task branch, never on main.
    assert run_git(source_repo, "rev-parse", "HEAD").stdout.strip() == main_head_before
