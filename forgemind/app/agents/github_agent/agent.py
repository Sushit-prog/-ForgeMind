"""The GitHub Agent (architecture doc section E, Phase 10).

The phase's finishing agent — DETERMINISTIC, not a tool-use loop (the Test
Agent pattern): it pushes the worktree branch to the fork, opens a DRAFT PR
from the persisted-artifact template, comments on the source issue (when
the task has one), and persists the ``PullRequest`` row. There is NO LLM
call and no LLM judgment on *whether* to open a PR — that decision was
already made by the Reviewer + Security gates and the VERIFICATION staleness
check passing. The PR body is assembled from PERSISTED artifacts
(``pr_template``), never re-generated.

The three steps run through the Phase-3 pipeline, so every one is audited
exactly like any other tool call:

1. ``git.push``           — force-push the worktree branch to the fork
                            (skipped under ``FORGEMIND_MOCK_GITHUB=1``; the
                            real push is covered by its own runtime tests).
2. ``github.create_pr``   — draft PR, target resolved server-side.
3. ``github.comment_issue``— PR link on the source issue, if any; a comment
                            failure NEVER fails the task (the PR exists; the
                            comment is annotation) — audited instead.

A failure at push or create_pr raises ``GitHubAgentError`` and the task goes
FAILED at PR_CREATION. A failure to persist the PR row is likewise a hard
failure — an unreported PR is nothing.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import ClassVar

from app.agents.base import Agent
from app.agents.github_agent.pr_template import build_pr_content
from app.agents.github_agent.schema import PullRequest
from app.execution import ToolPipeline
from app.models import AuditLog, Task
from app.models import PullRequest as PullRequestRow
from app.tools.base import ExecutionContext

logger = logging.getLogger(__name__)


class GitHubAgentError(RuntimeError):
    """PR creation could not be produced at all (task FAILED at PR_CREATION)."""


class GitHubAgent(Agent):
    name: ClassVar[str] = "github"
    description: ClassVar[str] = (
        "Pushes the implementation branch to the fork and opens a draft PR "
        "whose body is assembled from the persisted artifacts."
    )
    # Write-capable along exactly one axis: the fork (push + PR) and the
    # source-issue comment. There is no github.merge capability anywhere.
    capabilities: ClassVar[list[str]] = ["github.read", "github.write", "git.write"]

    async def run(self, task: Task, ctx: ExecutionContext) -> PullRequest:
        """Drive PR_CREATION: push -> create draft PR -> comment -> persist."""
        if ctx.db is None or ctx.task_id is None:
            raise GitHubAgentError("ExecutionContext.db is required for PR creation")
        db = ctx.db

        from app.git.worktree_manager import WorktreeManager

        worktree = WorktreeManager(db).get_or_create_for_task(task)

        title, body = build_pr_content(task, db)

        # (1) push the branch to the fork (real git, audited). Under the
        # stub/mock provider there is no network, so the real push is
        # skipped and an audit trail says so (the push itself is covered by
        # its own runtime tests + the real-GitHub e2e).
        if os.environ.get("FORGEMIND_MOCK_GITHUB") == "1":
            self._audit(db, task.id, "github.push_skipped", {"reason": "mock provider"})
        else:
            push = await ToolPipeline(db).invoke(
                "git.push",
                {"worktree_id": str(worktree.id)},
                set(self.capabilities),
                ctx,
            )
            if push.status != "EXECUTED":
                raise GitHubAgentError(
                    f"git.push failed for task {task.id}: "
                    f"{push.error or push.denial_reason}"
                )

        # (2) open the DRAFT PR (target resolved server-side by the tool).
        pr_call = await ToolPipeline(db).invoke(
            "github.create_pr",
            {
                "worktree_id": str(worktree.id),
                "title": title,
                "body": body,
            },
            set(self.capabilities),
            ctx,
        )
        if pr_call.status != "EXECUTED":
            raise GitHubAgentError(
                f"github.create_pr failed for task {task.id}: "
                f"{pr_call.error or pr_call.denial_reason}"
            )
        out = pr_call.output or {}

        # (3) comment the PR link on the source issue, when there is one.
        # Best-effort by design: the PR already exists; a comment is an
        # annotation and must never fail the whole task over it.
        if task.issue_number is not None:
            comment_body = (
                f"ForgeMind opened a draft PR for this issue: {out.get('url', '')}"
            )
            comment = await ToolPipeline(db).invoke(
                "github.comment_issue",
                {"number": task.issue_number, "body": comment_body},
                set(self.capabilities),
                ctx,
            )
            if comment.status != "EXECUTED":
                self._audit(
                    db,
                    task.id,
                    "github.comment_failed",
                    {"error": comment.error or comment.denial_reason},
                )

        # (4) persist the PR row — an unpersisted PR is nothing.
        pr = PullRequest(
            repo=out.get("repo", ""),
            branch=out.get("branch", ""),
            number=int(out.get("number") or 0),
            url=out.get("url", ""),
            status="draft",
        )
        if not pr.repo or not pr.url or pr.number < 1:
            raise GitHubAgentError(
                f"github.create_pr returned an unusable payload for task {task.id}: {out}"
            )
        row = PullRequestRow(
            task_id=task.id,
            repo=pr.repo,
            branch=pr.branch,
            number=pr.number,
            url=pr.url,
            status=pr.status,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        logger.info(
            "Draft PR #%s opened for task %s on %s (branch %s)",
            pr.number,
            task.id,
            pr.repo,
            pr.branch,
        )
        return pr

    def _audit(self, db, task_id: uuid.UUID, action: str, details: dict) -> None:
        db.add(
            AuditLog(
                task_id=task_id,
                actor=self.name,
                action=action,
                entity_type="task",
                entity_id=str(task_id),
                details=details,
            )
        )
        db.commit()


def build_github() -> GitHubAgent | None:
    """Construct the GitHub agent. Returns None when no client is configured
    (no token and no mock flag) — its state's tasks then fail cleanly at
    PR_CREATION instead of hanging."""
    from app.github import build_github_client

    client = build_github_client()
    if client is None:
        return None
    return GitHubAgent()
