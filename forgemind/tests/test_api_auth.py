"""Phase 10.5 — bearer-token auth on the mutating Task API routes.

Covers the fail-closed matrix (missing / wrong / empty / non-bearer scheme),
the happy path WITH a valid token (proving ``secrets.compare_digest`` is
genuinely exercised, not short-circuited by a ``==``), the audit identity on
approve/reject/cancel, read routes staying open, and the config-level dev
default / production fail-closed behavior.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import select

from app.config import DEV_API_TOKEN, Settings
from app.database.session import SessionLocal, engine
from app.main import create_app
from app.models import AuditLog, Approval, Base, Task

VALID_PAYLOAD = {
    "objective": "fix a bug",
    "repository_url": "https://github.com/org/repo.git",
}


@pytest.fixture()
def bare_client():
    """TestClient WITHOUT the default Authorization header — for 401 paths."""
    Base.metadata.create_all(engine)
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    Base.metadata.drop_all(engine)


def _park(db_session, task_id: uuid.UUID) -> None:
    """Set the task to AWAITING_APPROVAL (where the human decides)."""
    task = db_session.get(Task, task_id)
    assert task is not None
    task.status = "AWAITING_APPROVAL"
    db_session.commit()


# --- 401 matrix (bare client, no token) -----------------------------------


def test_create_task_requires_token(bare_client) -> None:
    resp = bare_client.post("/tasks", json=VALID_PAYLOAD)
    assert resp.status_code == 401
    assert resp.headers.get("WWW-Authenticate") == "Bearer"


def test_approve_requires_token(bare_client) -> None:
    resp = bare_client.post(f"/tasks/{uuid.uuid4()}/approve")
    assert resp.status_code == 401
    assert resp.headers.get("WWW-Authenticate") == "Bearer"


def test_reject_requires_token(bare_client) -> None:
    resp = bare_client.post(f"/tasks/{uuid.uuid4()}/reject")
    assert resp.status_code == 401


def test_cancel_requires_token(bare_client, client, db_session) -> None:
    created = client.post("/tasks", json=VALID_PAYLOAD)
    assert created.status_code == 201
    task_id = uuid.UUID(created.json()["id"])

    resp = bare_client.post(f"/tasks/{task_id}/cancel")
    assert resp.status_code == 401
    assert db_session.get(Task, task_id).status == "CREATED"  # untouched


def test_wrong_token_is_401(client) -> None:
    resp = client.post(
        "/tasks",
        json=VALID_PAYLOAD,
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401
    assert resp.headers.get("WWW-Authenticate") == "Bearer"


def test_empty_credential_is_401_not_500(client) -> None:
    """A present-but-empty token must 401, never crash on encode/compare."""
    resp = client.post(
        "/tasks",
        json=VALID_PAYLOAD,
        headers={"Authorization": "Bearer "},
    )
    assert resp.status_code == 401


def test_non_bearer_scheme_is_401(client) -> None:
    resp = client.post(
        "/tasks",
        json=VALID_PAYLOAD,
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
    )
    assert resp.status_code == 401
    assert resp.headers.get("WWW-Authenticate") == "Bearer"


# --- happy path with a valid token (compare_digest exercised) --------------


def test_create_task_with_valid_token(client) -> None:
    resp = client.post("/tasks", json=VALID_PAYLOAD)
    assert resp.status_code == 201
    assert resp.json()["status"] == "CREATED"


def test_approve_with_valid_token_records_authenticated_actor(
    client, db_session
) -> None:
    created = client.post("/tasks", json=VALID_PAYLOAD)
    task_id = uuid.UUID(created.json()["id"])
    _park(db_session, task_id)

    resp = client.post(f"/tasks/{task_id}/approve", json={"reason": "looks good"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "COMPLETED"
    approval = db_session.scalar(select(Approval).where(Approval.task_id == task_id))
    assert approval is not None and approval.action == "approve"
    entry = db_session.scalar(
        select(AuditLog).where(
            AuditLog.task_id == task_id, AuditLog.action == "task.approve"
        )
    )
    assert entry is not None
    assert entry.actor == "token-holder"


def test_cancel_with_valid_token_audited_as_token_holder(client, db_session) -> None:
    created = client.post("/tasks", json=VALID_PAYLOAD)
    task_id = uuid.UUID(created.json()["id"])

    resp = client.post(f"/tasks/{task_id}/cancel")

    assert resp.status_code == 200
    assert resp.json()["status"] == "FAILED"
    entry = db_session.scalar(
        select(AuditLog).where(
            AuditLog.task_id == task_id, AuditLog.action == "task.cancelled"
        )
    )
    assert entry is not None
    assert entry.actor == "token-holder"
    assert entry.details == {"reason": "user_cancelled"}


# --- read routes stay open ------------------------------------------------


def test_read_routes_open_without_token(bare_client) -> None:
    assert bare_client.get("/tasks").status_code == 200
    assert bare_client.get("/health").json() == {"status": "ok"}


# --- config-level: dev default + production fail-closed --------------------


def test_unset_token_defaults_to_dev_token(monkeypatch) -> None:
    monkeypatch.delenv("FORGEMIND_API_TOKEN", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "development")
    assert Settings(_env_file=None).api_token == DEV_API_TOKEN


def test_production_without_token_fails_closed(monkeypatch) -> None:
    monkeypatch.delenv("FORGEMIND_API_TOKEN", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_production_with_empty_token_fails_closed(monkeypatch) -> None:
    """An empty ``FORGEMIND_API_TOKEN=`` is treated as unset — never a
    silently-always-rejecting or silently-open server."""
    monkeypatch.setenv("FORGEMIND_API_TOKEN", "")
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_empty_token_in_development_falls_back_to_dev_default(monkeypatch) -> None:
    monkeypatch.setenv("FORGEMIND_API_TOKEN", "")
    monkeypatch.setenv("ENVIRONMENT", "development")
    assert Settings(_env_file=None).api_token == DEV_API_TOKEN


def test_production_with_token_configures(monkeypatch) -> None:
    monkeypatch.setenv("FORGEMIND_API_TOKEN", "prod-secret")
    monkeypatch.setenv("ENVIRONMENT", "production")
    assert Settings(_env_file=None).api_token == "prod-secret"


def test_explicit_token_beats_dev_default(monkeypatch) -> None:
    monkeypatch.setenv("FORGEMIND_API_TOKEN", "custom")
    monkeypatch.setenv("ENVIRONMENT", "development")
    assert Settings(_env_file=None).api_token == "custom"
