"""E2E (Postgres + real worker): PLANNING produces a real persisted plan.

The worker runs with the stub LLM provider (no API key needed); the flaky
variant exercises the retry-once path end to end.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select

from app.models import ExecutionEvent, Plan, PlanStep
from tests_e2e.conftest import approve_task, spawn_worker


def valid_payload(source_repo) -> dict:
    """A real clonable repo: RESEARCHING now runs the real agent (Phase 6),
    which needs a real worktree — a fake github URL would fail at clone."""
    return {
        "objective": "Fix the login bug",
        "repository_url": "file:///" + str(source_repo).replace("\\", "/"),
        "fork_url": "https://github.com/fork-owner/forgemind-e2e-fork",
    }


def test_worker_persists_real_plan(client, db_session, source_repo) -> None:
    proc = spawn_worker()
    try:
        created = client.post("/tasks", json=valid_payload(source_repo)).json()
        task_id = uuid.UUID(created["id"])

        approve_task(client, task_id)

        # A REAL plan was persisted (not the stub's hardcoded happy path).
        plans = db_session.scalars(select(Plan).where(Plan.task_id == task_id)).all()
        assert len(plans) == 1
        assert plans[0].status == "ACTIVE"
        assert "research" in plans[0].raw_llm_output
        steps = db_session.scalars(
            select(PlanStep).where(PlanStep.plan_id == plans[0].id)
        ).all()
        assert len(steps) >= 3
        assert [s.step_type for s in steps][0] == "research"

        # The PLANNING -> RESEARCHING transition recorded its reason.
        events = db_session.scalars(
            select(ExecutionEvent).where(ExecutionEvent.task_id == task_id)
        ).all()
        planning_events = [e for e in events if e.from_status == "PLANNING"]
        assert planning_events
        assert planning_events[0].to_status == "RESEARCHING"
        assert planning_events[0].reason == "plan_persisted"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_worker_retries_flaky_planner_output(client, db_session, source_repo) -> None:
    """Malformed first attempt, valid second — the worker's retry heals it."""
    proc = spawn_worker({"FORGEMIND_MOCK_LLM_FLAKY": "1"})
    try:
        created = client.post("/tasks", json=valid_payload(source_repo)).json()
        task_id = uuid.UUID(created["id"])

        approve_task(client, task_id)
        plans = db_session.scalars(select(Plan).where(Plan.task_id == task_id)).all()
        # Retry succeeded: exactly one ACTIVE plan (no INVALID leftover row
        # from a persisted failure — the first attempt never persisted).
        assert len(plans) == 1
        assert plans[0].status == "ACTIVE"
    finally:
        proc.terminate()
        proc.wait(timeout=10)
