"""Phase 11 — read-only execution trace viewer (GET /tasks/{id}/trace).

A human-readable HTML timeline over the task's execution history. Covered:
200 + expected rendered content for a task with events, a rendered (not JSON)
404 page for an unknown id, the zero-events empty state, the prominent PR
link at AWAITING_APPROVAL, the failure-reason banner, and the
auto-refresh meta only for non-terminal tasks. The trace route is open /
unaunthenticated (read-only, same as GET /tasks/{id}/events).
"""

from __future__ import annotations

import uuid

from sqlalchemy import select

from app.models import (
    Approval,
    ExecutionEvent,
    Plan,
    PlanStep,
    PullRequest,
    Repository,
    ReviewResult,
    SecurityResult,
    Task,
    TestRun,
    ToolCall,
)

VALID_PAYLOAD = {
    "objective": "Fix the flaky test in auth",
    "repository_url": "https://github.com/org/repo.git",
}


def _seed_repo_task(db_session) -> Task:
    repo = Repository(url="https://github.com/org/repo.git", default_branch="main")
    db_session.add(repo)
    db_session.flush()
    task = Task(objective=VALID_PAYLOAD["objective"], repository_id=repo.id)
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)
    return task


def _seed_events(db_session, task: Task, status: str = "AWAITING_APPROVAL") -> None:
    """A realistic journey: transitions + plan + tool call + test + review +
    security + PR + human approval. Events get explicit, distinct timestamps
    so the created_at ordering is deterministic regardless of sibling insert
    microsecond ties."""
    from datetime import datetime, timedelta, timezone

    task.status = status
    db_session.add(task)
    t0 = datetime.now(timezone.utc) - timedelta(seconds=30)

    def ev(from_s: str, to_s: str, reason: str | None = None, i: int = 0) -> None:
        db_session.add(
            ExecutionEvent(
                task_id=task.id,
                from_status=from_s,
                to_status=to_s,
                reason=reason,
                created_at=t0 + timedelta(seconds=i),
            )
        )

    ev("CREATED", "PLANNING", i=0)
    ev("PLANNING", "RESEARCHING", i=1)
    ev("RESEARCHING", "IMPLEMENTING", "artifact persisted", i=2)

    plan = Plan(task_id=task.id, status="ACTIVE")
    db_session.add(plan)
    db_session.flush()
    db_session.add(
        PlanStep(plan_id=plan.id, step_type="research", sequence=0, depends_on=None)
    )
    db_session.add(
        PlanStep(plan_id=plan.id, step_type="implement", sequence=1, depends_on=plan.id)
    )

    db_session.add(
        ToolCall(
            task_id=task.id,
            agent_type="researcher",
            tool_name="repository.search",
            input={"query": "VALUE"},
            output={},
            status="EXECUTED",
            risk="LOW",
        )
    )
    db_session.add(
        TestRun(task_id=task.id, status="passed", passed=2, failed=0, duration_ms=12)
    )
    db_session.add(
        ReviewResult(
            task_id=task.id,
            commit_sha="abc123",
            decision="APPROVE",
            severity="low",
            issues=[],
        )
    )
    db_session.add(
        SecurityResult(
            task_id=task.id,
            commit_sha="abc123",
            decision="PASS",
            findings=[],
        )
    )
    db_session.add(
        PullRequest(
            task_id=task.id,
            repo="you/org-fork",
            branch="agent/task-1",
            number=7,
            url="https://github.com/you/org-fork/pull/7",
            status="awaiting_approval",
        )
    )
    db_session.add(Approval(task_id=task.id, action="approve", reason="looks good"))
    db_session.commit()
    db_session.refresh(task)


# --- core rendering -------------------------------------------------------


def test_trace_renders_events_timeline(client, db_session) -> None:
    task = _seed_repo_task(db_session)
    _seed_events(db_session, task)

    resp = client.get(f"/tasks/{task.id}/trace")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    body = resp.text
    assert "Fix the flaky test in auth" in body
    assert "status: AWAITING_APPROVAL" in body
    assert "CREATED → PLANNING" in body
    assert "PLANNING → RESEARCHING" in body
    assert "artifact persisted" in body
    assert "researcher: repository.search" in body
    assert "Test run: passed" in body
    assert "Review: APPROVE" in body
    assert "Security: PASS" in body
    assert "Human approve" in body
    assert "github.com/you/org-fork/pull/7" in body
    assert "Plan (ACTIVE)" in body
    assert "implement" in body


def test_trace_surfaces_pr_link_at_approval(client, db_session) -> None:
    task = _seed_repo_task(db_session)
    _seed_events(db_session, task)

    body = client.get(f"/tasks/{task.id}/trace").text

    # A plain anchor to the draft PR, not just a raw detail line.
    assert 'href="https://github.com/you/org-fork/pull/7"' in body
    assert "#7 (awaiting_approval)" in body


def test_trace_surfaces_failure_reason(client, db_session) -> None:
    task = _seed_repo_task(db_session)
    task.status = "FAILED"
    db_session.add(
        ExecutionEvent(
            task_id=task.id,
            from_status="IMPLEMENTING",
            to_status="FAILED",
            reason="test suite failed: VALUE != 3",
        )
    )
    db_session.commit()
    db_session.refresh(task)

    body = client.get(f"/tasks/{task.id}/trace").text

    assert "FAILED" in body
    assert "test suite failed: VALUE != 3" in body
    # Terminal task: no auto-refresh.
    assert 'http-equiv="refresh"' not in body


def test_trace_zero_events_is_not_an_error(client, db_session) -> None:
    task = _seed_repo_task(db_session)

    resp = client.get(f"/tasks/{task.id}/trace")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    body = resp.text
    assert "No events yet" in body
    assert "status: CREATED" in body
    # Still in flight: auto-refresh.
    assert 'http-equiv="refresh"' in body


def test_trace_unknown_task_renders_404_page(client) -> None:
    resp = client.get(f"/tasks/{uuid.uuid4()}/trace")

    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("text/html")
    assert "Task not found" in resp.text


def test_trace_is_open_unauthenticated(db_session) -> None:
    """Trace needs no bearer token — consistent with the read-only routes."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    task = _seed_repo_task(db_session)

    with TestClient(create_app()) as bare:
        resp = bare.get(f"/tasks/{task.id}/trace")

    assert resp.status_code == 200
    assert "status: CREATED" in resp.text


# --- shared events query still intact (refactor guard) --------------------


def test_events_endpoint_still_returns_events(client, db_session) -> None:
    task = _seed_repo_task(db_session)
    _seed_events(db_session, task)

    resp = client.get(f"/tasks/{task.id}/events")

    assert resp.status_code == 200
    rows = resp.json()
    # Oldest first: the three seeded transitions in order.
    assert [r["to_status"] for r in rows] == ["PLANNING", "RESEARCHING", "IMPLEMENTING"]
    assert rows[0]["from_status"] == "CREATED"
