"""Phase 12 ``github.merge_pr`` tool: gated squash merge.

Coverage: capability-gate denies a caller without ``github.merge``; the
repo allowlist (MERGE_ALLOWED_REPOS) denies a non-allowlisted fork; the
approval gate denies an unapproved task; the fresh staleness check denies
a mergeable=False PR and a drifted base; already-merged is a no-op denial;
and the success path sets merged_at/merge_commit_sha + records the merge.

All GitHub calls go through ``StubGitHubClient`` (patched ``_client()``),
so these are hermetic — no network, no token.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import select

from app.config import get_settings
from app.execution.tool_pipeline import (
    ToolInputValidationError,
    ToolPipeline,
    ToolResult,
)
from app.github.stub import StubGitHubClient
from app.models import Approval, PullRequest, Task, ToolCall
from app.tools.base import ExecutionContext


def run(coro):
    return asyncio.run(coro)


def ctx_for(db, task_id) -> ExecutionContext:
    return ExecutionContext(task_id=task_id, agent_type="github", db=db)


def stub_client(monkeypatch) -> StubGitHubClient:
    stub = StubGitHubClient()
    monkeypatch.setattr("app.tools.github_tools._client", lambda: stub)
    return stub


def executed_output(result: ToolResult) -> dict:
    """Narrow a ToolResult to its output, requiring an EXECUTED-status call."""
    assert result.status == "EXECUTED"
    assert result.output is not None, result.error
    return result.output


@pytest.fixture()
def allow_merge(monkeypatch):
    """MERGE_ALLOWED_REPOS = the fork under test; restore settings cache."""
    monkeypatch.setenv("MERGE_ALLOWED_REPOS", "fork/repo")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def seed_approved_mergeable(
    db_session,
    repo_task,
    stub: StubGitHubClient,
    *,
    approved: bool = True,
    status: str = "COMPLETED",
) -> tuple[PullRequest, Task]:
    """A repo+task with a stub PR + persisted PR row, parked post-approval.

    ``stub.create_pr`` seeds the stub's by-number PR so ``get_pr`` can find
    it; the persisted row copies the stub's number/base_sha so the tool's
    staleness gate sees a consistent world.
    """
    repo, task = repo_task
    repo.url = "https://github.com/upstream/some-repo"
    repo.fork_url = "https://github.com/fork/repo"
    task.status = status
    if approved:
        db_session.add(Approval(task_id=task.id, action="approve"))
    pr_data = run(
        stub.create_pr(
            "fork",
            "repo",
            head=f"agent/task-{task.id}",
            base="main",
            title="Fix it",
            body="body",
        )
    )
    pr = PullRequest(
        task_id=task.id,
        repo=pr_data.repo,
        branch=pr_data.branch,
        number=pr_data.number,
        url=pr_data.url,
        status="draft",
        base_sha=pr_data.base_sha,
    )
    db_session.add(pr)
    db_session.commit()
    db_session.refresh(pr)
    return pr, task


def invoke_merge(db_session, task_id, caps=None) -> ToolResult:
    return run(
        ToolPipeline(db_session).invoke(
            "github.merge_pr",
            {"task_id": str(task_id)},
            caps or {"github.merge"},
            ctx_for(db_session, task_id),
        )
    )


def test_merge_requires_github_merge_capability(
    db_session, repo_task, monkeypatch, allow_merge
) -> None:
    """The capability gate is the first layer: no github.merge -> DENIED."""
    stub = stub_client(monkeypatch)
    pr, task = seed_approved_mergeable(db_session, repo_task, stub)
    result = invoke_merge(db_session, task.id, caps={"github.write"})
    assert result.status == "DENIED"
    assert "missing required capability" in (result.denial_reason or "")
    assert "github.merge" in (result.denial_reason or "")
    row = db_session.scalar(
        select(ToolCall).where(ToolCall.tool_name == "github.merge_pr")
    )
    assert row is not None and row.status == "DENIED"
    assert db_session.get(PullRequest, pr.id).merged_at is None


def test_merge_denies_repo_not_in_allowlist(db_session, repo_task, monkeypatch) -> None:
    """MERGE_ALLOWED_REPOS empty/unset = merge disabled everywhere."""
    stub = stub_client(monkeypatch)
    pr, task = seed_approved_mergeable(db_session, repo_task, stub)
    out = executed_output(invoke_merge(db_session, task.id))
    assert out["merged"] is False
    assert "MERGE_ALLOWED_REPOS" in (out["denial_reason"] or "")
    assert db_session.get(PullRequest, pr.id).merged_at is None


def test_merge_denies_unapproved_task(
    db_session, repo_task, monkeypatch, allow_merge
) -> None:
    stub = stub_client(monkeypatch)
    pr, task = seed_approved_mergeable(db_session, repo_task, stub, approved=False)
    out = executed_output(invoke_merge(db_session, task.id))
    assert out["merged"] is False
    assert "not approved" in (out["denial_reason"] or "")
    assert db_session.get(PullRequest, pr.id).merged_at is None


def test_merge_denies_past_approval_with_non_completed_status(
    db_session, repo_task, monkeypatch, allow_merge
) -> None:
    """Approval-evidence exists but the task is not COMPLETED -> fail closed."""
    stub = stub_client(monkeypatch)
    pr, task = seed_approved_mergeable(
        db_session, repo_task, stub, status="AWAITING_APPROVAL"
    )
    out = executed_output(invoke_merge(db_session, task.id))
    assert out["merged"] is False
    assert "COMPLETED" in (out["denial_reason"] or "")


def test_merge_denies_stale_pr_when_not_mergeable(
    db_session, repo_task, monkeypatch, allow_merge
) -> None:
    """GitHub's own mergeable flag is the second staleness check."""
    stub = stub_client(monkeypatch)
    pr, task = seed_approved_mergeable(db_session, repo_task, stub)
    stub._override_mergeable[("fork", "repo", pr.number)] = False
    out = executed_output(invoke_merge(db_session, task.id))
    assert out["merged"] is False
    assert "mergeable=False" in (out["denial_reason"] or "")


def test_merge_denies_stale_pr_when_base_drifted(
    db_session, repo_task, monkeypatch, allow_merge
) -> None:
    """Base-branch advanced since the PR opened -> deny (VERIFICATION pattern)."""
    stub = stub_client(monkeypatch)
    pr, task = seed_approved_mergeable(db_session, repo_task, stub)
    stub._override_base_sha[("fork", "repo", pr.number)] = "newer-base-sha"
    out = executed_output(invoke_merge(db_session, task.id))
    assert out["merged"] is False
    assert "base branch drifted" in (out["denial_reason"] or "")
    assert db_session.get(PullRequest, pr.id).merged_at is None


def test_merge_already_merged_is_noop_denial(
    db_session, repo_task, monkeypatch, allow_merge
) -> None:
    from datetime import datetime, timezone

    stub = stub_client(monkeypatch)
    pr, task = seed_approved_mergeable(db_session, repo_task, stub)
    pr.merged_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pr.merge_commit_sha = "already-sha"
    db_session.commit()
    out = executed_output(invoke_merge(db_session, task.id))
    assert out["merged"] is False
    assert "already merged" in (out["denial_reason"] or "")
    assert db_session.get(PullRequest, pr.id).merge_commit_sha == "already-sha"


def test_merge_success_sets_merged_fields_and_calls_squash(
    db_session, repo_task, monkeypatch, allow_merge
) -> None:
    stub = stub_client(monkeypatch)
    pr, task = seed_approved_mergeable(db_session, repo_task, stub)

    out = executed_output(invoke_merge(db_session, task.id))

    assert out["merged"] is True
    assert out["merge_commit_sha"]
    assert out["merge_commit_sha"] != pr.base_sha

    pr = db_session.get(PullRequest, pr.id)
    assert pr.merged_at is not None
    assert pr.merge_commit_sha == out["merge_commit_sha"]

    merge_calls = [c for c in stub.calls if c["op"] == "merge_pr"]
    assert len(merge_calls) == 1
    assert merge_calls[0]["owner"] == "fork"
    assert merge_calls[0]["repo"] == "repo"
    assert merge_calls[0]["number"] == pr.number
    assert merge_calls[0]["merge_method"] == "squash"
    # Staleness probes happened before the merge call.
    assert any(c["op"] == "get_pr" for c in stub.calls)


def test_merge_pr_passes_policy_default_allow(
    db_session, repo_task, monkeypatch, allow_merge
) -> None:
    """github.merge_pr is NOT name-scoped-denied by the explicit-deny rule.

    The belt-and-suspenders ExplicitDenyRule denies the bare name
    ``github.merge``; our tool is ``github.merge_pr`` and must pass the
    policy gate (otherwise the gated tool could never run at all).
    """
    stub = stub_client(monkeypatch)
    pr, task = seed_approved_mergeable(db_session, repo_task, stub)
    # Reached execute (or was denied later by a business gate, never by
    # policy). With allow_merge+approved+mergeable it must fully succeed.
    out = executed_output(invoke_merge(db_session, task.id))
    assert out["merged"] is True


def test_merge_input_schema_forbids_extra_fields(
    db_session, repo_task, monkeypatch, allow_merge
) -> None:
    stub = stub_client(monkeypatch)
    pr, task = seed_approved_mergeable(db_session, repo_task, stub)
    # A smuggled owner/repo/number field is rejected at validation.
    with pytest.raises(ToolInputValidationError):
        run(
            ToolPipeline(db_session).invoke(
                "github.merge_pr",
                {
                    "task_id": str(task.id),
                    "owner": "evil",
                    "repo": "repo",
                    "number": 1,
                },
                {"github.merge"},
                ctx_for(db_session, task.id),
            )
        )
