"""Phase 2 API tests: cancel + execution-event trail.

The queue is disabled in the test environment (see conftest), so POST /tasks
never touches Redis — tasks simply stay CREATED until the worker's sweep.
"""

import uuid

from sqlalchemy import select

from app.models import ExecutionEvent, Task, TaskStatus
from app.runtime.state_machine import LEGAL_TRANSITIONS
from app.runtime.task_lifecycle import AUTO_PIPELINE, USER_CANCELLED, transition_task

VALID_PAYLOAD = {
    "objective": "Fix the flaky test in auth",
    "repository_url": "https://github.com/org/repo.git",
}


def test_cancel_transitions_to_failed_with_reason(client, db_session) -> None:
    created = client.post("/tasks", json=VALID_PAYLOAD).json()
    resp = client.post(f"/tasks/{created['id']}/cancel")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "FAILED"

    events = db_session.scalars(
        select(ExecutionEvent).where(ExecutionEvent.task_id == uuid.UUID(created["id"]))
    ).all()
    assert len(events) == 1
    assert events[0].from_status == "CREATED"
    assert events[0].to_status == "FAILED"
    assert events[0].reason == USER_CANCELLED


def test_cancel_on_completed_returns_409(client, db_session) -> None:
    created = client.post("/tasks", json=VALID_PAYLOAD).json()
    task = db_session.get(Task, uuid.UUID(created["id"]))
    task.status = "COMPLETED"
    db_session.commit()

    resp = client.post(f"/tasks/{created['id']}/cancel")
    assert resp.status_code == 409
    assert "COMPLETED" in resp.json()["detail"]


def test_cancel_on_escalated_returns_409(client, db_session) -> None:
    created = client.post("/tasks", json=VALID_PAYLOAD).json()
    task = db_session.get(Task, uuid.UUID(created["id"]))
    task.status = "ESCALATED"
    db_session.commit()

    resp = client.post(f"/tasks/{created['id']}/cancel")
    assert resp.status_code == 409


def test_cancel_on_already_failed_returns_409(client, db_session) -> None:
    created = client.post("/tasks", json=VALID_PAYLOAD).json()
    task = db_session.get(Task, uuid.UUID(created["id"]))
    task.status = "FAILED"
    db_session.commit()

    resp = client.post(f"/tasks/{created['id']}/cancel")
    assert resp.status_code == 409


def test_cancel_unknown_task_returns_404(client) -> None:
    resp = client.post(f"/tasks/{uuid.uuid4()}/cancel")
    assert resp.status_code == 404


def test_cancel_malformed_id_returns_422(client) -> None:
    resp = client.post("/tasks/not-a-uuid/cancel")
    assert resp.status_code == 422


def test_events_endpoint_returns_ordered_trail(client, db_session) -> None:
    created = client.post("/tasks", json=VALID_PAYLOAD).json()
    task_id = uuid.UUID(created["id"])

    # Drive the task through the whole pipeline directly (worker not running).
    task = db_session.get(Task, task_id)
    for target in AUTO_PIPELINE[1:]:
        transition_task(db_session, task, target)
    db_session.commit()

    resp = client.get(f"/tasks/{task_id}/events")
    assert resp.status_code == 200
    events = resp.json()

    assert [e["to_status"] for e in events] == [s.value for s in AUTO_PIPELINE[1:]]
    # Oldest first, each consecutive pair a legal transition.
    for e in events:
        assert TaskStatus(e["to_status"]) in LEGAL_TRANSITIONS[TaskStatus(e["from_status"])]
    assert events[-1]["to_status"] == "COMPLETED"


def test_events_unknown_task_returns_404(client) -> None:
    resp = client.get(f"/tasks/{uuid.uuid4()}/events")
    assert resp.status_code == 404


def test_events_empty_for_fresh_task(client) -> None:
    created = client.post("/tasks", json=VALID_PAYLOAD).json()
    resp = client.get(f"/tasks/{created['id']}/events")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_task_stays_created_when_queue_disabled(client, db_session) -> None:
    """With the queue disabled, POST must not advance the task itself."""
    created = client.post("/tasks", json=VALID_PAYLOAD).json()
    assert created["status"] == "CREATED"
    resp = client.get(f"/tasks/{created['id']}")
    assert resp.json()["status"] == "CREATED"
