"""E2E (Postgres + real worker): RESEARCHING runs the real Research Agent.

The task is submitted through the API against a real local git repo
(file:// URL). The worker (stub LLM provider, default per-schema script)
drives PLANNING -> RESEARCHING, where the Research Agent runs a real,
bounded, audited tool-use loop against the cloned worktree and persists a
content-validated ResearchArtifact before the task proceeds.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy import select

from app.models import (
    ExecutionEvent,
    Plan,
    ResearchArtifact,
    ToolCall,
)
from tests_e2e.conftest import approve_task, spawn_worker


def _file_url(repo: Path) -> str:
    """file:// URL for a local repo path (validator accepts file://)."""
    return "file:///" + str(repo).replace("\\", "/")


def test_worker_research_loop_produces_artifact(
    client, db_session, source_repo
) -> None:
    proc = spawn_worker()
    try:
        payload = {
            "objective": "Fix the VALUE bug",
            "repository_url": _file_url(source_repo),
            "fork_url": "https://github.com/fork-owner/forgemind-e2e-fork",
        }
        created = client.post("/tasks", json=payload).json()
        task_id = uuid.UUID(created["id"])

        approve_task(client, task_id, timeout=120)

        # A research artifact was persisted by the real agent.
        artifacts = db_session.scalars(
            select(ResearchArtifact).where(ResearchArtifact.task_id == task_id)
        ).all()
        assert len(artifacts) == 1
        artifact = artifacts[0]
        assert artifact.relevant_files == ["src/app.py"]  # grounded in the loop
        assert 0.0 <= float(artifact.confidence) <= 1.0

        # The tool-use loop ran REAL tool calls through the pipeline: the
        # repository.search was EXECUTED and audited as one tool_calls row.
        calls = db_session.scalars(
            select(ToolCall)
            .where(ToolCall.task_id == task_id, ToolCall.agent_type == "researcher")
            .order_by(ToolCall.created_at)
        ).all()
        assert [c.tool_name for c in calls] == ["repository.search"]
        assert calls[0].status == "EXECUTED"

        # RESEARCHING -> IMPLEMENTING fired only after persistence.
        events = db_session.scalars(
            select(ExecutionEvent).where(ExecutionEvent.task_id == task_id)
        ).all()
        research_events = [e for e in events if e.from_status == "RESEARCHING"]
        assert research_events
        assert research_events[0].to_status == "IMPLEMENTING"
        assert research_events[0].reason == "artifact_persisted"

        # And the plan was persisted too (planner ran first).
        plans = db_session.scalars(select(Plan).where(Plan.task_id == task_id)).all()
        assert len(plans) == 1
    finally:
        proc.terminate()
        proc.wait(timeout=10)
