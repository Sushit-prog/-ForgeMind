"""Phase 10 API: fork_url + issue_number on task creation, and the human
approve/reject checkpoint endpoints."""

from __future__ import annotations

import uuid

from sqlalchemy import select

from app.models import Approval, PullRequest, Task, Worktree


def _create(client, *, fork_url=None, issue_number=None) -> uuid.UUID:
    payload = {"objective": "fix a bug", "repository_url": "https://github.com/o/r.git"}
    if fork_url:
        payload["fork_url"] = fork_url
    if issue_number:
        payload["issue_number"] = issue_number
    resp = client.post("/tasks", json=payload)
    assert resp.status_code == 201, resp.text
    return uuid.UUID(resp.json()["id"])


def _park(client, db_session, task_id: uuid.UUID) -> None:
    """Set the task to AWAITING_APPROVAL (where the human decides)."""
    task = db_session.get(Task, task_id)
    assert task is not None
    task.status = "AWAITING_APPROVAL"
    db_session.commit()


def test_create_task_stores_fork_url_and_issue_number(client, db_session) -> None:
    task_id = _create(
        client, fork_url="https://github.com/you/fork.git", issue_number=12
    )
    task = db_session.get(Task, task_id)
    assert task.issue_number == 12
    assert task.repository.fork_url == "https://github.com/you/fork.git"
    # No fork_url on a NEW repo -> stays None (PR_CREATION fails closed later).
    other = client.post(
        "/tasks",
        json={
            "objective": "fix a bug",
            "repository_url": "https://github.com/o/other-repo.git",
        },
    ).json()
    other_task = db_session.get(Task, uuid.UUID(other["id"]))
    assert other_task.repository.fork_url is None


def test_create_task_rejects_malformed_fork_url(client) -> None:
    resp = client.post(
        "/tasks",
        json={
            "objective": "x",
            "repository_url": "https://github.com/o/r.git",
            "fork_url": "not a url",
        },
    )
    assert resp.status_code == 422


def test_approve_moves_to_completed_and_records_decision(client, db_session) -> None:
    task_id = _create(client, fork_url="https://github.com/you/fork.git")
    _park(client, db_session, task_id)
    db_session.add(
        PullRequest(
            task_id=task_id,
            repo="you/fork",
            branch="agent/task",
            number=1,
            url="https://github.com/you/fork/pull/1",
            status="draft",
        )
    )
    db_session.commit()

    resp = client.post(f"/tasks/{task_id}/approve", json={"reason": "looks good"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "COMPLETED"
    approval = db_session.scalar(select(Approval).where(Approval.task_id == task_id))
    assert approval is not None
    assert approval.action == "approve"
    assert approval.reason == "looks good"
    # The PR row mirrors the human decision — never a merge claim.
    pr = db_session.scalar(select(PullRequest).where(PullRequest.task_id == task_id))
    assert pr.status == "approved"
    # Worktree untouched: task done does not mean anything was git-merged.
    wt = db_session.scalar(select(Worktree).where(Worktree.task_id == task_id))
    assert wt is None or wt.status == "active"


def test_reject_moves_to_failed_with_reason(client, db_session) -> None:
    task_id = _create(client)
    _park(client, db_session, task_id)

    resp = client.post(f"/tasks/{task_id}/reject", json={"reason": "wrong approach"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "FAILED"
    approval = db_session.scalar(select(Approval).where(Approval.task_id == task_id))
    assert approval is not None and approval.action == "reject"
    task = db_session.get(Task, task_id)
    assert task.status == "FAILED"


def test_approve_reject_conflict_when_not_awaiting(client, db_session) -> None:
    """A decision on a task that is NOT waiting for a human is a 409, never a
    silent no-op (the Phase-2 cancel-on-completed pattern)."""
    task_id = _create(client)  # CREATED, not AWAITING_APPROVAL

    assert client.post(f"/tasks/{task_id}/approve").status_code == 409
    assert client.post(f"/tasks/{task_id}/reject").status_code == 409
    task = db_session.get(Task, task_id)
    assert task.status == "CREATED"  # untouched


def test_approve_unknown_task_is_404(client) -> None:
    assert client.post(f"/tasks/{uuid.uuid4()}/approve").status_code == 404
    assert client.post(f"/tasks/{uuid.uuid4()}/reject").status_code == 404
