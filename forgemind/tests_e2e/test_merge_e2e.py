"""E2E — Phase 12 real-GitHub merge: draft PR -> human review -> approve -> merge.

Same env gate as ``test_github_e2e.py``; additionally exercises the gated
``POST /tasks/{id}/merge`` path end-to-end against the REAL GitHub API:

    FORGEMIND_GITHUB_E2E=1
    GITHUB_TOKEN=<PAT with write access to the fork>
    FORGEMIND_E2E_UPSTREAM_URL=https://github.com/<you>/<public-cloneable-repo>
    FORGEMIND_E2E_FORK_URL=https://github.com/<you>/<your-fork-or-repo>

Flow: the worker runs the FULL pipeline (stub LLM, REAL GitHub client) and
opens a real DRAFT PR on the fork. A draft PR cannot be merged (GitHub
422), so the HOST test plays the human reviewer: it un-drafts the PR via the
normal REST API (``PATCH /pulls/{n}`` with ``{"draft": false}``), waits for
GitHub to compute mergeability, approves via the API, then calls
``POST /tasks/{id}/merge``. The merge tool re-verifies every gate (approval
evidence, MERGE_ALLOWED_REPOS allowlist, fresh staleness) inside the API
process — which must therefore run the REAL GitHub client and carry the
allowlist, so both are configured in-process before the merge call
(``get_settings.cache_clear()`` after setting MERGE_ALLOWED_REPOS).

The test skips unless fully configured (it writes to a REAL fork).
"""

from __future__ import annotations

import asyncio
import os
import uuid

import httpx
import pytest
from sqlalchemy import select

from app.config import get_settings
from app.github.client import GitHubClient
from app.models import AuditLog, PullRequest, Task
from tests_e2e.conftest import spawn_worker, wait_for

_CACHED_SETTINGS = (
    get_settings.__wrapped__ if hasattr(get_settings, "__wrapped__") else None
)

_CONFIGURED = (
    os.environ.get("FORGEMIND_GITHUB_E2E") == "1"
    and bool(os.environ.get("GITHUB_TOKEN"))
    and bool(os.environ.get("FORGEMIND_E2E_UPSTREAM_URL"))
    and bool(os.environ.get("FORGEMIND_E2E_FORK_URL"))
)

pytestmark = pytest.mark.skipif(
    not _CONFIGURED,
    reason="set FORGEMIND_GITHUB_E2E=1 + GITHUB_TOKEN + E2E upstream/fork URLs (see module docstring)",
)


def _real_client() -> GitHubClient:
    return GitHubClient(os.environ["GITHUB_TOKEN"], max_retries=3)


def _mark_ready(owner: str, repo: str, number: int) -> None:
    """Human review step: flip the draft PR to ready-for-review via the real
    REST API (PATCH), exactly as an operator would in the GitHub UI."""
    resp = httpx.patch(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}",
        headers={
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={"draft": False},
        timeout=30.0,
    )
    assert resp.status_code == 200, (
        f"un-draft PATCH failed: {resp.status_code} {resp.text[:300]}"
    )


def _configure_in_process(fork_slug: str) -> None:
    """Point the API process at the REAL GitHub client + the merge allowlist.

    ``build_github_client`` reads FORGEMIND_MOCK_GITHUB live from the env,
    but MERGE_ALLOWED_REPOS (and github_token) come from the lru-cached
    Settings — so the env is updated and the cache cleared before any merge
    call so the route's ToolPipeline picks them up at request time.
    """
    os.environ["FORGEMIND_MOCK_GITHUB"] = "0"
    os.environ["MERGE_ALLOWED_REPOS"] = fork_slug
    get_settings.cache_clear()


def _restore_in_process(
    original_mock: str | None, original_allowlist: str | None
) -> None:
    if original_allowlist is None:
        os.environ.pop("MERGE_ALLOWED_REPOS", None)
    else:
        os.environ["MERGE_ALLOWED_REPOS"] = original_allowlist
    if original_mock is None:
        os.environ.pop("FORGEMIND_MOCK_GITHUB", None)
    else:
        os.environ["FORGEMIND_MOCK_GITHUB"] = original_mock
    get_settings.cache_clear()


def test_real_merge_after_reviewer_un_drafts(client, db_session) -> None:
    upstream = os.environ["FORGEMIND_E2E_UPSTREAM_URL"]
    fork = os.environ["FORGEMIND_E2E_FORK_URL"]
    owner, repo_name = fork.split("/")[-2:]
    fork_slug = f"{owner}/{repo_name}"
    original_mock = os.environ.get("FORGEMIND_MOCK_GITHUB")
    original_allowlist = os.environ.get("MERGE_ALLOWED_REPOS")
    _configure_in_process(fork_slug)
    proc = spawn_worker(
        {"FORGEMIND_MOCK_GITHUB": "0", "MERGE_ALLOWED_REPOS": fork_slug}
    )
    try:
        created = client.post(
            "/tasks",
            json={
                "objective": "Add a trivial README note documenting ForgeMind e2e merge",
                "repository_url": upstream,
                "fork_url": fork,
            },
        )
        assert created.status_code == 201, created.text
        task_id = uuid.UUID(created.json()["id"])

        def awaiting() -> bool:
            return (
                client.get(f"/tasks/{task_id}").json()["status"] == "AWAITING_APPROVAL"
            )

        assert wait_for(awaiting, timeout=300), "task never reached AWAITING_APPROVAL"

        row = db_session.scalar(
            select(PullRequest).where(PullRequest.task_id == task_id)
        )
        assert row is not None, "no PullRequest row persisted"
        assert row.repo == fork_slug
        assert row.base_sha, "base_sha not persisted at PR creation"
        assert row.merged_at is None

        # Human review step: un-draft so GitHub will accept a merge, then
        # wait for mergeability to be computed (async on GitHub's side).
        _mark_ready(owner, repo_name, row.number)

        def mergeable() -> bool:
            fresh = asyncio.run(_real_client().get_pr(owner, repo_name, row.number))
            return fresh.state == "open" and fresh.mergeable is True

        assert wait_for(mergeable, timeout=120), "PR never became mergeable"

        # Approve -> COMPLETED (Phase 10 contract).
        assert client.post(f"/tasks/{task_id}/approve").status_code == 200
        db_session.expire_all()
        task = db_session.get(Task, task_id)
        assert task is not None and task.status == "COMPLETED"

        # The gated merge through the API (real client, all gates re-checked).
        merged = client.post(f"/tasks/{task_id}/merge")
        assert merged.status_code == 200, merged.text
        payload = merged.json()
        assert payload["merged"] is True, merged.text
        assert payload["merge_commit_sha"], merged.text
        assert payload["denial_reason"] is None, merged.text

        # Persisted outcome on the PR row.
        db_session.expire_all()
        row = db_session.get(PullRequest, row.id)
        assert row is not None and row.merged_at is not None
        assert row.merge_commit_sha == payload["merge_commit_sha"]

        # GitHub-side truth: the PR really got squash-merged into its base.
        fresh = asyncio.run(_real_client().get_pr(owner, repo_name, row.number))
        assert fresh.state == "closed"
        assert fresh.merged is True

        # The endpoint's audit trail entry (actor = the shared token).
        audit = db_session.scalar(
            select(AuditLog)
            .where(AuditLog.task_id == task_id, AuditLog.action == "task.merged")
            .order_by(AuditLog.created_at.desc())
            .limit(1)
        )
        assert audit is not None and audit.actor == "token-holder"
        assert audit.details.get("merge_commit_sha") == payload["merge_commit_sha"]
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        _restore_in_process(original_mock, original_allowlist)


del _CACHED_SETTINGS  # keep module namespace clean (noqa: F841 guard)
