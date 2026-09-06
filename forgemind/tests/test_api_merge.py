"""Phase 12 API: POST /tasks/{id}/merge (and 409/400 gates).

The gated squash-merge endpoint — bearer-auth'd, requires an approval record,
and returns the tool's structured outcome (merged / denial reason) as JSON.
All GitHub calls go through the in-memory stub, so the API route exercises
the full tool pipeline hermetically.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import get_settings
from app.database.session import engine
from app.github.stub import StubGitHubClient
from app.main import create_app
from app.models import AuditLog, Base, PullRequest, Task


def run(coro):
    return asyncio.run(coro)


def stub_client(monkeypatch) -> StubGitHubClient:
    stub = StubGitHubClient()
    monkeypatch.setattr("app.tools.github_tools._client", lambda: stub)
    return stub


@pytest.fixture()
def bare_client():
    """TestClient WITHOUT the default Authorization header — for 401 paths."""
    Base.metadata.create_all(engine)
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    Base.metadata.drop_all(engine)


def _create(client, *, fork_url="https://github.com/fork/repo.git") -> uuid.UUID:
    resp = client.post(
        "/tasks",
        json={
            "objective": "fix a bug",
            "repository_url": "https://github.com/upstream/some-repo.git",
            "fork_url": fork_url,
        },
    )
    assert resp.status_code == 201, resp.text
    return uuid.UUID(resp.json()["id"])


def _park(client, db_session, task_id: uuid.UUID) -> None:
    task = db_session.get(Task, task_id)
    assert task is not None
    task.status = "AWAITING_APPROVAL"
    db_session.commit()


def _seed_pr(db_session, task_id: uuid.UUID, stub: StubGitHubClient) -> None:
    """Create the stub PR (number 1 by default) + the persisted PR row to match."""
    pr_data = run(
        stub.create_pr(
            "fork",
            "repo",
            head=f"agent/task-{task_id}",
            base="main",
            title="Fix it",
            body="body",
        )
    )
    db_session.add(
        PullRequest(
            task_id=task_id,
            repo=pr_data.repo,
            branch=pr_data.branch,
            number=pr_data.number,
            url=pr_data.url,
            status="draft",
            base_sha=pr_data.base_sha,
        )
    )
    db_session.commit()


@pytest.fixture()
def allow_merge(monkeypatch):
    """MERGE_ALLOWED_REPOS = the fork under test; settings cache restored after."""
    monkeypatch.setenv("MERGE_ALLOWED_REPOS", "fork/repo")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_merge_requires_bearer_token(bare_client) -> None:
    resp = bare_client.post(f"/tasks/{uuid.uuid4()}/merge")
    assert resp.status_code == 401
    assert resp.headers.get("WWW-Authenticate") == "Bearer"


def test_merge_unknown_task_is_404(client) -> None:
    assert client.post(f"/tasks/{uuid.uuid4()}/merge").status_code == 404


def test_merge_409_when_not_approved(client, db_session, monkeypatch) -> None:
    """Precondition: a task with no approval record is a 409, never a no-op."""
    stub_client(monkeypatch)
    task_id = _create(client)
    _park(client, db_session, task_id)
    _seed_pr(db_session, task_id, stub_client(monkeypatch))

    resp = client.post(f"/tasks/{task_id}/merge")
    assert resp.status_code == 409
    assert "approve" in resp.json()["detail"]


def test_merge_denial_when_repo_not_allowlisted(
    client, db_session, monkeypatch
) -> None:
    """MERGE_ALLOWED_REPOS empty/unset -> 200 with merged=False + reason."""
    get_settings.cache_clear()  # ensure the disabled (empty) default is re-read
    stub_client(monkeypatch)
    task_id = _create(client)
    _park(client, db_session, task_id)
    _seed_pr(db_session, task_id, stub_client(monkeypatch))

    assert (
        client.post(f"/tasks/{task_id}/approve", json={"reason": "ok"}).status_code
        == 200
    )

    resp = client.post(f"/tasks/{task_id}/merge")
    assert resp.status_code == 200
    body = resp.json()
    assert body["merged"] is False
    assert "MERGE_ALLOWED_REPOS" in (body["denial_reason"] or "")
    pr = db_session.scalar(select(PullRequest).where(PullRequest.task_id == task_id))
    assert pr is not None and pr.merged_at is None


def test_merge_success_returns_sha_and_records_audit(
    client, db_session, monkeypatch, allow_merge
) -> None:
    stub = stub_client(monkeypatch)
    task_id = _create(client)
    _park(client, db_session, task_id)
    _seed_pr(db_session, task_id, stub)

    assert (
        client.post(f"/tasks/{task_id}/approve", json={"reason": "ok"}).status_code
        == 200
    )

    resp = client.post(f"/tasks/{task_id}/merge")

    assert resp.status_code == 200
    body = resp.json()
    assert body["merged"] is True
    assert body["merge_commit_sha"]

    pr = db_session.scalar(select(PullRequest).where(PullRequest.task_id == task_id))
    assert pr is not None
    assert pr.merged_at is not None
    assert pr.merge_commit_sha == body["merge_commit_sha"]

    # Audit trail: a human-facing task.merged row, actor = token holder, with
    # the merge SHA — the approve + merge decision pair is fully recorded.
    merged_rows = list(
        db_session.scalars(
            select(AuditLog).where(
                AuditLog.task_id == task_id, AuditLog.action == "task.merged"
            )
        )
    )
    assert len(merged_rows) == 1
    assert merged_rows[0].actor == "token-holder"
    assert merged_rows[0].details["merge_commit_sha"] == body["merge_commit_sha"]
    assert merged_rows[0].details["repo"] == "fork/repo"


def test_merge_approve_then_get_task_shows_recommendation(
    client, db_session, monkeypatch, allow_merge
) -> None:
    """GET /tasks/{id} exposes the derived recommendation after approval."""
    stub = stub_client(monkeypatch)
    task_id = _create(client)
    _park(client, db_session, task_id)
    _seed_pr(db_session, task_id, stub)
    assert client.post(f"/tasks/{task_id}/approve").status_code == 200

    body = client.get(f"/tasks/{task_id}").json()
    # No review/security verdicts were seeded, so the recommendation holds.
    assert body["recommended_action"] == "hold"
    assert "no review verdict" in (body["recommendation_reason"] or "")
