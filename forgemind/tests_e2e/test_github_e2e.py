"""E2E — Phase 10 against the REAL GitHub API: a fresh task → a real DRAFT PR
on a real fork → human approve → COMPLETED.

This is the phase's mandatory real-GitHub exercise. It is env-gated because
it needs YOUR GitHub account, a PAT, and a repo you control:

    FORGEMIND_GITHUB_E2E=1
    GITHUB_TOKEN=<PAT with write access to the fork>
    FORGEMIND_E2E_UPSTREAM_URL=https://github.com/<you>/<public-cloneable-repo>
    FORGEMIND_E2E_FORK_URL=https://github.com/<you>/<your-fork-or-repo>

The worker runs the FULL pipeline with the stub LLM (deterministic agents)
but the REAL GitHub client for PR_CREATION: it pushes the task branch to
your fork and opens a real DRAFT PR (body assembled from the persisted
artifacts, no LLM). The test waits for AWAITING_APPROVAL, proves the PR
exists on GitHub, then approves via the API → COMPLETED.

Merging is never attempted by the system — the PR stays a draft for YOU to
review and merge by hand. The test skips unless fully configured.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from sqlalchemy import select

from app.github.client import GitHubClient
from app.models import PullRequest, Task
from tests_e2e.conftest import spawn_worker, wait_for

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


def test_real_draft_pr_then_approve(client, db_session) -> None:
    upstream = os.environ["FORGEMIND_E2E_UPSTREAM_URL"]
    fork = os.environ["FORGEMIND_E2E_FORK_URL"]
    proc = spawn_worker({"FORGEMIND_MOCK_GITHUB": "0"})
    try:
        created = client.post(
            "/tasks",
            json={
                "objective": "Add a trivial README note documenting ForgeMind e2e",
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

        # The draft PR row was persisted with a real fork URL.
        row = db_session.scalar(
            select(PullRequest).where(PullRequest.task_id == task_id)
        )
        assert row is not None, "no PullRequest row persisted"
        assert row.status == "draft"
        assert "/" in row.repo
        assert row.url.startswith("https://github.com/")

        # Prove the PR really exists on GitHub and its head is the task's
        # branch on the fork — fetched through the API, not the stub.
        owner, repo_name = row.repo.split("/", 1)
        found = asyncio.run(
            _real_client().find_open_pr_for_branch(owner, repo_name, row.branch)
        )
        assert found is not None, f"PR for branch {row.branch} not found on GitHub"
        assert found.number == row.number
        assert found.url == row.url

        # The human checkpoint: approve -> COMPLETED.
        assert client.post(f"/tasks/{task_id}/approve").status_code == 200
        db_session.expire_all()
        task = db_session.get(Task, task_id)
        assert task is not None and task.status == "COMPLETED"
    finally:
        proc.terminate()
        proc.wait(timeout=10)
