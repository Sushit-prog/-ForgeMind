"""Phase 10 github.* + git.push tools: server-side resolution, the
fork/upstream split (adversarially), fail-closed guards, and the missing
``github.merge`` boundary."""

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import select

from app.execution.tool_pipeline import ToolInputValidationError, ToolPipeline
from app.github.errors import GitHubConfigError
from app.github.stub import StubGitHubClient
from app.git.runner import run_git
from app.models import Repository, Task, ToolCall, Worktree
from app.tools.base import ExecutionContext
from app.tools.github_tools import (
    CreatePrInput,
    GitHubCreatePrTool,
    GitHubGetIssueTool,
    GITHUB_TOOLS,
)
from app.tools.git_tools import PushInput, PushTool


def run(coro):
    return asyncio.run(coro)


def ctx_for(db, task_id) -> ExecutionContext:
    return ExecutionContext(task_id=task_id, agent_type="github", db=db)


def stub_client(monkeypatch):
    """Point the tools at one shared stub and return it for assertions."""
    stub = StubGitHubClient()
    monkeypatch.setattr("app.tools.github_tools._client", lambda: stub)
    return stub


# --- git.push ----------------------------------------------------------------


def make_worktree_with_commit(
    db_session, task_id: uuid.UUID, repo: Repository
) -> Worktree:
    """A real worktree with one committed change, ready to push."""
    from pathlib import Path

    from app.git.operations import GitOperations
    from app.git.worktree_manager import WorktreeManager

    wt = WorktreeManager(db_session).create(task_id, repo)
    Path(wt.path, "src", "app.py").write_text("VALUE = 2\n")
    run_git(wt.path, "add", "-A")
    run_git(wt.path, "commit", "-m", "the fix")
    return wt


def test_git_push_targets_only_the_fork(db_session, repo_task, tmp_path) -> None:
    """Real push: the branch lands on the FORK (a local bare repo here)."""
    repo, task = repo_task
    bare = tmp_path / "fork.git"
    run_git(tmp_path, "init", "--bare", str(bare))
    repo.fork_url = str(bare)
    db_session.commit()
    wt = make_worktree_with_commit(db_session, task.id, repo)

    result = run(
        ToolPipeline(db_session).invoke(
            "git.push",
            {"worktree_id": str(wt.id)},
            {"git.write"},
            ctx_for(db_session, task.id),
        )
    )

    assert result.status == "EXECUTED"
    assert result.output["branch"] == f"agent/task-{task.id}"
    # The fork holds the branch.
    assert (
        f"refs/heads/agent/task-{task.id}"
        in run_git(bare, "for-each-ref", "--format=%(refname)").stdout
    )


def test_git_push_fails_closed_without_fork(db_session, repo_task) -> None:
    repo, task = repo_task
    repo.fork_url = None
    db_session.commit()
    wt = make_worktree_with_commit(db_session, task.id, repo)

    with pytest.raises(GitHubConfigError, match="no fork"):
        run(
            PushTool().execute(
                PushInput(worktree_id=wt.id), ctx_for(db_session, task.id)
            )
        )


def test_git_push_forbids_upstream_equality(db_session, repo_task) -> None:
    """Adversarial: fork_url == upstream url is a SecurityError — the system
    can never be configured to push to the upstream reference."""
    from app.git.errors import SecurityError

    repo, task = repo_task
    repo.fork_url = repo.url  # the "trick": fork == upstream
    db_session.commit()
    wt = make_worktree_with_commit(db_session, task.id, repo)

    with pytest.raises(SecurityError, match="upstream reference"):
        run(
            PushTool().execute(
                PushInput(worktree_id=wt.id), ctx_for(db_session, task.id)
            )
        )


# --- github.create_pr --------------------------------------------------------


def test_create_pr_resolves_fork_target_server_side(
    db_session, repo_task, monkeypatch
) -> None:
    """The PR is created against the FORK slug derived from fork_url — the
    upstream reference can never be the target, and the input schema carries
    no owner/repo/head/base fields at all."""
    stub = stub_client(monkeypatch)
    repo, task = repo_task
    repo.url = "https://github.com/upstream/org-upstream"
    repo.fork_url = "https://github.com/fork-owner/fork-repo"
    repo.default_branch = "main"
    db_session.commit()
    wt = Worktree(
        task_id=task.id,
        repository_id=repo.id,
        branch_name=f"agent/task-{task.id}",
        path="/tmp/nowhere",
        base_commit="a" * 40,
        status="active",
    )
    db_session.add(wt)
    db_session.commit()
    db_session.refresh(wt)

    result = run(
        ToolPipeline(db_session).invoke(
            "github.create_pr",
            {"worktree_id": str(wt.id), "title": "Fix it", "body": "body"},
            {"github.write"},
            ctx_for(db_session, task.id),
        )
    )

    assert result.status == "EXECUTED"
    assert result.output["repo"] == "fork-owner/fork-repo"
    assert result.output["branch"] == wt.branch_name
    assert result.output["status"] == "draft"

    calls = [c for c in stub.calls if c["op"] == "create_pr"]
    assert len(calls) == 1
    assert calls[0]["owner"] == "fork-owner"
    assert calls[0]["repo"] == "fork-repo"
    assert calls[0]["head"] == wt.branch_name
    assert calls[0]["base"] == "main"
    assert calls[0]["draft"] is True


def test_create_pr_is_idempotent_for_same_branch(
    db_session, repo_task, monkeypatch
) -> None:
    """A second create_pr for the same branch reuses the existing PR."""
    stub = stub_client(monkeypatch)
    repo, task = repo_task
    repo.url = "https://github.com/upstream/org-upstream"
    repo.fork_url = "https://github.com/fork-owner/fork-repo"
    repo.default_branch = "main"
    db_session.commit()
    wt = Worktree(
        task_id=task.id,
        repository_id=repo.id,
        branch_name=f"agent/task-{task.id}",
        path="/tmp/nowhere",
        status="active",
    )
    db_session.add(wt)
    db_session.commit()
    db_session.refresh(wt)

    payload = {"worktree_id": str(wt.id), "title": "Fix it", "body": "body"}
    first = run(
        ToolPipeline(db_session).invoke(
            "github.create_pr", payload, {"github.write"}, ctx_for(db_session, task.id)
        )
    )
    second = run(
        ToolPipeline(db_session).invoke(
            "github.create_pr", payload, {"github.write"}, ctx_for(db_session, task.id)
        )
    )

    assert first.output["number"] == second.output["number"]  # reused, not duplicated
    # Both calls went through the client, but no SECOND PR number was issued.
    creates = [c for c in stub.calls if c["op"] == "create_pr"]
    assert len(creates) == 2
    assert {c["head"] for c in creates} == {wt.branch_name}


def test_create_pr_fails_closed_without_fork(db_session, repo_task) -> None:
    repo, task = repo_task
    repo.fork_url = None
    db_session.commit()
    wt = Worktree(
        task_id=task.id,
        repository_id=repo.id,
        branch_name="agent/x",
        path="/tmp/nowhere",
        status="active",
    )
    db_session.add(wt)
    db_session.commit()

    with pytest.raises(GitHubConfigError, match="no fork"):
        run(
            GitHubCreatePrTool().execute(
                CreatePrInput(worktree_id=wt.id, title="t", body="b"),
                ctx_for(db_session, task.id),
            )
        )


def test_create_pr_rejects_smuggled_target_fields(
    db_session, repo_task, monkeypatch
) -> None:
    """Adversarial: an extra 'owner'/'repo'/'base'/'head' field is rejected at
    validation — there is no agent-input path to redirect the PR target."""
    stub_client(monkeypatch)
    repo, task = repo_task
    repo.url = "https://github.com/upstream/org-upstream"
    repo.fork_url = "https://github.com/fork-owner/fork-repo"
    db_session.commit()
    wt = Worktree(
        task_id=task.id,
        repository_id=repo.id,
        branch_name="agent/x",
        path="/tmp/nowhere",
        status="active",
    )
    db_session.add(wt)
    db_session.commit()

    with pytest.raises(ToolInputValidationError):
        run(
            ToolPipeline(db_session).invoke(
                "github.create_pr",
                {
                    "worktree_id": str(wt.id),
                    "title": "t",
                    "body": "b",
                    "owner": "evil",
                    "repo": "malware",
                },
                {"github.write"},
                ctx_for(db_session, task.id),
            )
        )


# --- github.get_issue --------------------------------------------------------


def test_get_issue_reads_from_upstream(db_session, repo_task, monkeypatch) -> None:
    """Issue reads resolve owner/repo from repositories.url (upstream)."""
    stub = stub_client(monkeypatch)
    repo, task = repo_task
    repo.url = "https://github.com/upstream/org-upstream"
    db_session.commit()

    result = run(
        ToolPipeline(db_session).invoke(
            "github.get_issue",
            {"number": 42},
            {"github.read"},
            ctx_for(db_session, task.id),
        )
    )

    assert result.status == "EXECUTED"
    assert result.output["number"] == 42
    issue_call = [c for c in stub.calls if c["op"] == "get_issue"]
    assert issue_call[0]["owner"] == "upstream"
    assert issue_call[0]["repo"] == "org-upstream"


def test_comment_issue_posts_to_upstream_issue(
    db_session, repo_task, monkeypatch
) -> None:
    """The comment link resolves owner/repo from the upstream URL and reaches
    the client only when the issue exists (seeded in the stub)."""
    stub = stub_client(monkeypatch)
    repo, task = repo_task
    repo.url = "https://github.com/upstream/org-upstream"
    db_session.commit()
    stub.seed_issue("upstream", "org-upstream", 7)

    result = run(
        ToolPipeline(db_session).invoke(
            "github.comment_issue",
            {"number": 7, "body": "ForgeMind opened a draft PR: http://x/1"},
            {"github.write"},
            ctx_for(db_session, task.id),
        )
    )

    assert result.status == "EXECUTED"
    assert [c for c in stub.calls if c["op"] == "comment_on_issue"][0]["number"] == 7
    assert len(stub.comments) == 1
    assert "draft PR" in stub.comments[0]["body"]


# --- capability boundary + explicit deny -------------------------------------


def test_create_pr_denied_without_github_write(
    db_session, repo_task, monkeypatch
) -> None:
    """A caller without github.write (e.g. the Research agent, which holds
    only github.read) is DENIED at the capability gate — never executed."""
    stub = stub_client(monkeypatch)
    repo, task = repo_task
    repo.url = "https://github.com/upstream/org-upstream"
    repo.fork_url = "https://github.com/fork-owner/fork-repo"
    db_session.commit()
    wt = Worktree(
        task_id=task.id,
        repository_id=repo.id,
        branch_name="agent/x",
        path="/tmp/nowhere",
        status="active",
    )
    db_session.add(wt)
    db_session.commit()

    result = run(
        ToolPipeline(db_session).invoke(
            "github.create_pr",
            {"worktree_id": str(wt.id), "title": "t", "body": "b"},
            {"github.read"},  # research's set — no write
            ctx_for(db_session, task.id),
        )
    )

    assert result.status == "DENIED"
    assert "github.write" in (result.denial_reason or "")
    assert not any(c["op"] == "create_pr" for c in stub.calls)
    row = db_session.scalar(
        select(ToolCall).where(ToolCall.tool_name == "github.create_pr")
    )
    assert row is not None and row.status == "DENIED"


def test_github_merge_does_not_exist() -> None:
    """No github.merge tool is registered — grep-verifiable literal check."""
    names = {t.name for t in GITHUB_TOOLS}
    assert "github.merge" not in names
    assert "merge" not in " ".join(names)


def test_github_merge_denied_by_policy_even_if_registered() -> None:
    """Belt-and-suspenders: if a merge tool were ever registered, the policy
    engine's explicit-deny rule rejects it before execution."""
    from app.policies.engine import PolicyEngine

    class _MergeInput(dict):
        pass

    fake = type(
        "FakeMergeTool",
        (),
        {"name": "github.merge", "risk": "HIGH", "input": None},
    )
    decision = PolicyEngine().evaluate(fake, _MergeInput(), {"github.write"})
    assert decision.allowed is False
    assert decision.rule == "explicit-deny"
