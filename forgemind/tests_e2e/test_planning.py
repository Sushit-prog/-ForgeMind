"""E2E (Postgres + real worker): PLANNING produces a real persisted plan.

The worker runs with the stub LLM provider (no API key needed); the flaky
variant exercises the retry-once path end to end.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select

from app.models import ExecutionEvent, Plan, PlanStep
from tests_e2e.conftest import spawn_worker, wait_for

VALID_PAYLOAD = {
    "objective": "Fix the login bug",
    "repository_url": "https://github.com/org/repo.git",
}


def test_worker_persists_real_plan(client, db_session) -> None:
    proc = spawn_worker()
    try:
        created = client.post("/tasks", json=VALID_PAYLOAD).json()
        task_id = uuid.UUID(created["id"])

        def completed() -> bool:
            return client.get(f"/tasks/{task_id}").json()["status"] == "COMPLETED"

        assert wait_for(completed, timeout=60)

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


def test_worker_retries_flaky_planner_output(client, db_session) -> None:
    """Malformed first attempt, valid second — the worker's retry heals it."""
    proc = spawn_worker({"FORGEMIND_MOCK_LLM_FLAKY": "1"})
    try:
        created = client.post("/tasks", json=VALID_PAYLOAD).json()
        task_id = uuid.UUID(created["id"])

        def completed() -> bool:
            return client.get(f"/tasks/{task_id}").json()["status"] == "COMPLETED"

        assert wait_for(completed, timeout=60)
        plans = db_session.scalars(select(Plan).where(Plan.task_id == task_id)).all()
        # Retry succeeded: exactly one ACTIVE plan (no INVALID leftover row
        # from a persisted failure — the first attempt never persisted).
        assert len(plans) == 1
        assert plans[0].status == "ACTIVE"
    finally:
        proc.terminate()
        proc.wait(timeout=10)
