"""Phase 10 lifecycle: PR_CREATION runs the GitHub Agent for real and
AWAITING_APPROVAL is a genuine terminal-until-human checkpoint."""

from __future__ import annotations

import asyncio
import uuid

from sqlalchemy import select

from app.agents.github_agent.agent import GitHubAgent
from app.models import ExecutionEvent, PullRequest, Task, TaskStatus
from app.runtime.task_lifecycle import advance_task_with_agents, transition_task


def run(coro):
    return asyncio.run(coro)


def events_for(db_session, task_id: uuid.UUID) -> list[ExecutionEvent]:
    return list(
        db_session.scalars(
            select(ExecutionEvent)
            .where(ExecutionEvent.task_id == task_id)
            .order_by(ExecutionEvent.created_at, ExecutionEvent.id)
        )
    )


def to_pr_creation(db_session, task: Task) -> None:
    """Walk the task through every agent state to PR_CREATION (bypassing the
    agents' semantics — this test only exercises the Phase 10 routing)."""
    for target in (
        TaskStatus.PLANNING,
        TaskStatus.RESEARCHING,
        TaskStatus.IMPLEMENTING,
        TaskStatus.TESTING,
        TaskStatus.REVIEWING,
        TaskStatus.SECURITY_REVIEW,
        TaskStatus.VERIFICATION,
        TaskStatus.PR_CREATION,
    ):
        transition_task(db_session, task, target)
    db_session.commit()
    db_session.refresh(task)


def test_pr_creation_reaches_awaiting_approval(
    db_session, repo_task, monkeypatch
) -> None:
    """A real GitHubAgent (mock provider) at PR_CREATION persists the PR row
    and the transition fires: PR_CREATION -> AWAITING_APPROVAL, reason
    ``pr_created``."""
    monkeypatch.setenv("FORGEMIND_MOCK_GITHUB", "1")
    repo, task = repo_task
    repo.fork_url = "https://github.com/fork-owner/fork-repo"
    repo.default_branch = "main"
    db_session.commit()
    to_pr_creation(db_session, task)

    status = run(advance_task_with_agents(db_session, task.id, github=GitHubAgent()))

    assert status is TaskStatus.AWAITING_APPROVAL
    pr = db_session.scalar(select(PullRequest).where(PullRequest.task_id == task.id))
    assert pr is not None and pr.status == "draft"
    trail = [(e.from_status, e.to_status) for e in events_for(db_session, task.id)]
    assert ("PR_CREATION", "AWAITING_APPROVAL") in trail, trail
    assert events_for(db_session, task.id)[-1].reason == "pr_created"


def test_pr_creation_without_client_fails_cleanly(db_session, repo_task) -> None:
    """No GitHub agent (no token / no mock) -> FAILED with ``no_github_client``,
    never a hang and never a silent skip."""
    repo, task = repo_task
    to_pr_creation(db_session, task)

    status = run(advance_task_with_agents(db_session, task.id, github=None))

    assert status is TaskStatus.FAILED
    assert events_for(db_session, task.id)[-1].reason == "no_github_client"
    db_session.expire_all()
    assert db_session.get(Task, task.id).status == "FAILED"


def test_pr_creation_failure_fails_task(db_session, repo_task) -> None:
    """create_pr failing (fork_url unset -> fail closed) is a HARD failure:
    the agent raises, the task goes FAILED with ``github_failed``."""
    repo, task = repo_task  # no fork_url configured
    to_pr_creation(db_session, task)

    status = run(advance_task_with_agents(db_session, task.id, github=GitHubAgent()))

    assert status is TaskStatus.FAILED
    assert events_for(db_session, task.id)[-1].reason == "github_failed"


def test_awaiting_approval_is_terminal_until_human(
    db_session, repo_task, monkeypatch
) -> None:
    """AWAITING_APPROVAL never auto-advances: re-running the worker unit of
    work returns None and the task stays parked."""
    monkeypatch.setenv("FORGEMIND_MOCK_GITHUB", "1")
    repo, task = repo_task
    repo.fork_url = "https://github.com/fork-owner/fork-repo"
    db_session.commit()
    to_pr_creation(db_session, task)
    assert run(advance_task_with_agents(db_session, task.id, github=GitHubAgent())) is (
        TaskStatus.AWAITING_APPROVAL
    )

    parked = run(advance_task_with_agents(db_session, task.id))
    assert parked is None
    db_session.expire_all()
    assert db_session.get(Task, task.id).status == "AWAITING_APPROVAL"
