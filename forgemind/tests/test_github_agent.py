"""GitHub Agent (Phase 10): the deterministic PR-creating agent.

Covers the full agent run through the Phase-3 pipeline:

- mock mode (``FORGEMIND_MOCK_GITHUB=1``): the real git push is skipped and
  audited; the PR is created through the stub client and persisted.
- real mode: ``git.push`` executes for REAL against a local bare "fork"
  (the stub client + slug seam simulate the GitHub API side); the PR is
  created as a draft and persisted; a comment failure on the source issue
  is tolerated (audited), never failing the task.

The REAL push + REAL GitHub API path is covered by the env-gated e2e test
(``tests_e2e/test_github_e2e.py``).
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from sqlalchemy import select

from app.agents.github_agent.agent import GitHubAgent
from app.github.stub import StubGitHubClient
from app.git.runner import run_git
from app.models import AuditLog, PullRequest as PullRequestRow
from app.tools.base import ExecutionContext


def run(coro):
    return asyncio.run(coro)


def ctx_for(db, task_id) -> ExecutionContext:
    return ExecutionContext(task_id=task_id, agent_type="github", db=db)


def stub_client(monkeypatch):
    stub = StubGitHubClient()
    monkeypatch.setattr("app.tools.github_tools._client", lambda: stub)
    return stub


def audits_for(db, task_id: uuid.UUID, action: str) -> list[AuditLog]:
    return list(
        db.scalars(
            select(AuditLog)
            .where(AuditLog.task_id == task_id, AuditLog.action == action)
            .order_by(AuditLog.created_at)
        )
    )


def test_mock_mode_skips_push_and_persists_draft_pr(
    db_session, repo_task, monkeypatch
) -> None:
    """Mock provider: push is skipped (audited), the draft PR is created via
    the stub against the fork slug, and the PullRequest row is persisted."""
    stub = stub_client(monkeypatch)
    repo, task = repo_task
    repo.fork_url = "https://github.com/fork-owner/fork-repo"
    repo.default_branch = "main"
    db_session.commit()
    monkeypatch.setenv("FORGEMIND_MOCK_GITHUB", "1")

    pr = run(GitHubAgent().run(task, ctx_for(db_session, task.id)))

    assert pr.repo == "fork-owner/fork-repo"
    assert pr.branch == f"agent/task-{task.id}"
    assert pr.status == "draft"
    assert pr.number >= 1

    row = db_session.scalar(
        select(PullRequestRow).where(PullRequestRow.task_id == task.id)
    )
    assert row is not None
    assert row.repo == "fork-owner/fork-repo"
    assert row.number == pr.number
    assert row.status == "draft"

    # push skipped + audited, exactly one create_pr call, no comment (no issue).
    assert audits_for(db_session, task.id, "github.push_skipped")
    creates = [c for c in stub.calls if c["op"] == "create_pr"]
    assert len(creates) == 1
    assert creates[0]["draft"] is True
    assert creates[0]["owner"] == "fork-owner"
    assert creates[0]["head"] == pr.branch
    assert stub.comments == []


def test_real_push_then_pr_with_tolerated_comment_failure(
    db_session, repo_task, tmp_path, monkeypatch
) -> None:
    """Real mode: git.push runs for REAL against a local bare fork; the draft
    PR is created (stub + slug seam); the source-issue comment is ATTEMPTED
    but fails (local repo has no GitHub URL) and is tolerated — audited, the
    PR still persists and the agent returns."""
    stub = stub_client(monkeypatch)
    # The slug seam: the fork is a LOCAL bare repo here (real push target),
    # so server-side slug resolution is simulated for the API side only.
    monkeypatch.setattr(
        "app.tools.github_tools._fork_slug_parts",
        lambda repository: ("fork-owner", "fork-repo"),
    )
    monkeypatch.delenv("FORGEMIND_MOCK_GITHUB", raising=False)

    repo, task = repo_task
    bare = tmp_path / "fork.git"
    run_git(tmp_path, "init", "--bare", str(bare))
    repo.fork_url = str(bare)
    repo.default_branch = "main"
    task.issue_number = 42  # exercise the comment path
    db_session.commit()

    # The task's worktree gets a real commit to push.
    from app.git.worktree_manager import WorktreeManager

    wt = WorktreeManager(db_session).create(task.id, repo)
    Path(wt.path, "src", "app.py").write_text("VALUE = 2\n")
    run_git(wt.path, "add", "-A")
    run_git(wt.path, "commit", "-m", "the fix")

    pr = run(GitHubAgent().run(task, ctx_for(db_session, task.id)))

    assert pr.repo == "fork-owner/fork-repo"
    assert pr.status == "draft"
    # The branch really landed on the fork (the real push happened).
    assert (
        f"refs/heads/{pr.branch}"
        in run_git(bare, "for-each-ref", "--format=%(refname)").stdout
    )

    row = db_session.scalar(
        select(PullRequestRow).where(PullRequestRow.task_id == task.id)
    )
    assert row is not None and row.url == pr.url

    # The comment was attempted (issue_number set) but the local repo URL is
    # not a GitHub URL, so it failed and was tolerated — not a task failure.
    comment_calls = [c for c in stub.calls if c["op"] == "comment_on_issue"]
    assert comment_calls == []  # never reached the (stubbed) API
    assert audits_for(db_session, task.id, "github.comment_failed")
    assert audits_for(db_session, task.id, "github.push_skipped") == []  # real push


def test_no_issue_number_skips_comment(db_session, repo_task, monkeypatch) -> None:
    """A task without an issue_number never attempts the comment."""
    stub = stub_client(monkeypatch)
    repo, task = repo_task
    repo.fork_url = "https://github.com/fork-owner/fork-repo"
    repo.default_branch = "main"
    db_session.commit()
    monkeypatch.setenv("FORGEMIND_MOCK_GITHUB", "1")

    run(GitHubAgent().run(task, ctx_for(db_session, task.id)))

    assert stub.comments == []
    assert audits_for(db_session, task.id, "github.comment_failed") == []
