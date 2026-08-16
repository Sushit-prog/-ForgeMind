"""Integration tests for the Task API (POST then GET round-trips)."""

import uuid

from sqlalchemy import func, select

from app.models import AuditLog, Repository, Task

VALID_PAYLOAD = {
    "objective": "Fix the flaky test in auth",
    "repository_url": "https://github.com/org/repo.git",
}


def test_health(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_create_task_returns_created(client) -> None:
    resp = client.post("/tasks", json=VALID_PAYLOAD)
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "CREATED"
    assert body["objective"] == VALID_PAYLOAD["objective"]
    uuid.UUID(body["id"])  # must be a valid UUID


def test_get_task_round_trip(client) -> None:
    created = client.post("/tasks", json=VALID_PAYLOAD).json()
    resp = client.get(f"/tasks/{created['id']}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == created["id"]
    assert body["objective"] == VALID_PAYLOAD["objective"]
    assert body["status"] == "CREATED"
    assert body["created_at"] == created["created_at"]


def test_list_tasks(client) -> None:
    client.post("/tasks", json=VALID_PAYLOAD)
    client.post(
        "/tasks",
        json={
            "objective": "Add pagination to the API",
            "repository_url": "https://github.com/org/repo.git",
        },
    )
    resp = client.get("/tasks")
    assert resp.status_code == 200
    tasks = resp.json()
    assert len(tasks) == 2
    # Most recent first.
    assert tasks[0]["objective"] == "Add pagination to the API"


def test_invalid_repository_url_returns_422(client) -> None:
    resp = client.post("/tasks", json={"objective": "x", "repository_url": "not-a-url"})
    assert resp.status_code == 422


def test_missing_repository_url_returns_422(client) -> None:
    resp = client.post("/tasks", json={"objective": "x"})
    assert resp.status_code == 422


def test_missing_objective_returns_422(client) -> None:
    resp = client.post(
        "/tasks", json={"repository_url": "https://github.com/org/repo.git"}
    )
    assert resp.status_code == 422


def test_get_unknown_task_returns_404(client) -> None:
    resp = client.get(f"/tasks/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_get_malformed_task_id_returns_422(client) -> None:
    resp = client.get("/tasks/not-a-uuid")
    assert resp.status_code == 422


def test_task_creation_writes_audit_log(client, db_session) -> None:
    client.post("/tasks", json=VALID_PAYLOAD)
    entries = db_session.scalars(select(AuditLog)).all()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.action == "task.created"
    assert entry.actor == "api"
    assert entry.entity_type == "task"
    assert entry.task_id is not None


def test_same_repository_url_reuses_repository_row(client, db_session) -> None:
    client.post("/tasks", json=VALID_PAYLOAD)
    client.post("/tasks", json={**VALID_PAYLOAD, "objective": "Second task"})
    repo_count = db_session.scalar(select(func.count()).select_from(Repository))
    task_count = db_session.scalar(select(func.count()).select_from(Task))
    assert repo_count == 1
    assert task_count == 2
