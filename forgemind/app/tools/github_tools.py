"""``github.*`` tools (architecture doc section F, Phase 10).

- ``github.get_issue``     (github.read,  LOW)    — read an UPSTREAM issue.
- ``github.comment_issue`` (github.write, MEDIUM) — comment on an issue
  (used to link the PR back to the source issue).
- ``github.create_pr``     (github.write, HIGH)   — the first genuinely
  HIGH-risk tool, and the phase's approval-gated surface. It opens a DRAFT
  PR on the FORK.

Security posture, enforced here (never by convention):

- Every tool resolves its target SERVER-SIDE from the task's repository
  row. There is no ``owner``/``repo``/``head``/``base`` input field at all,
  so no agent (or injected prompt) can redirect a call to an arbitrary
  repo — the upstream reference is used only for READS (get_issue /
  comment link), and PR creation targets ``repositories.fork_url`` or fails
  closed. ``repositories.fork_url == repositories.url`` is a SecurityError.
- ``github.create_pr`` opens a draft by default and the agent always does —
  the second safety layer under the AWAITING_APPROVAL human gate.
- There is no ``github.merge`` tool, capability, or client method anywhere;
  the policy engine additionally denies it by name as a second layer.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict, Field

from app.github.client import GitHubClient, IssueData, PRData
from app.github.errors import GitHubConfigError
from app.github.slug import parse_github_slug
from app.models import Repository, Task, Worktree
from app.tools.base import ExecutionContext, Tool


def _client() -> GitHubClient:
    """The active client (stub in tests / key-less dev, real otherwise).

    ``build_github_client`` returns None when neither a token nor the mock
    flag is configured — that case fails closed here with a clear config
    error rather than a confusing network failure three steps later.
    """
    from app.github import build_github_client

    client = build_github_client()
    if client is None:
        raise GitHubConfigError(
            "no GitHub client configured — set GITHUB_TOKEN or FORGEMIND_MOCK_GITHUB=1"
        )
    return client


def _task_and_repository(ctx: ExecutionContext) -> tuple[Task, Repository]:
    """Resolve the calling task + its repository row (server-side seam).

    Tools require a task context; without it there is no server-side target
    to resolve and the call fails closed.
    """
    if ctx.db is None or ctx.task_id is None:
        raise GitHubConfigError("github tools require a task context (task_id + db)")
    task = ctx.db.get(Task, ctx.task_id)
    if task is None:
        raise GitHubConfigError(f"task {ctx.task_id} not found")
    repository = ctx.db.get(Repository, task.repository_id)
    if repository is None:
        raise GitHubConfigError(f"repository row missing for task {ctx.task_id}")
    return task, repository


def _worktree(ctx: ExecutionContext, worktree_id: uuid.UUID) -> Worktree:
    if ctx.db is None:
        raise GitHubConfigError("github tools require a task context (db)")
    wt = ctx.db.get(Worktree, worktree_id)
    if wt is None or wt.status != "active":
        raise GitHubConfigError(f"no active worktree {worktree_id}")
    return wt


def _fork_slug_parts(repository: Repository) -> tuple[str, str]:
    """The fork's ``(owner, repo)`` that PRs are created against.

    Strictly derived from ``repositories.fork_url``; unset or identical to
    the upstream means FAIL CLOSED — never a fallback to ``url``.
    """
    if not repository.fork_url:
        raise GitHubConfigError(
            "no fork configured for this repository (repositories.fork_url unset) — "
            "github.create_pr must target a fork, never the upstream"
        )
    if repository.fork_url == repository.url:
        raise GitHubConfigError(
            "repositories.fork_url must differ from repositories.url — "
            "creating PRs against the upstream reference is structurally forbidden"
        )
    return parse_github_slug(repository.fork_url)


class GetIssueInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: int = Field(ge=1)


class CommentIssueInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: int = Field(ge=1)
    body: str = Field(min_length=1, max_length=50_000)


class CommentIssueOutput(BaseModel):
    number: int
    posted: bool = True


class CreatePrInput(BaseModel):
    """The ONLY inputs the caller may give: which worktree, and the title/body.

    ``extra="forbid"`` means a smuggled ``owner``/``repo``/``head``/``base``
    field is REJECTED at validation before anything runs — there is no
    caller-supplied path to redirect the PR target. Everything else is
    resolved server-side from the task's repository row.
    """

    model_config = ConfigDict(extra="forbid")

    worktree_id: uuid.UUID
    title: str = Field(min_length=1, max_length=256)
    body: str = Field(min_length=1, max_length=50_000)


class CreatePrOutput(BaseModel):
    repo: str  # the FORK slug, never the upstream
    branch: str
    number: int
    url: str
    status: str  # draft | open


class GitHubGetIssueTool(Tool):
    name = "github.get_issue"
    description = (
        "Read a GitHub issue from the task's UPSTREAM repository (reads only)."
    )
    input_schema = GetIssueInput
    output_schema = IssueData
    capabilities: list[str] = ["github.read"]
    risk = "LOW"

    async def execute(self, input: GetIssueInput, ctx: ExecutionContext) -> IssueData:
        _, repository = _task_and_repository(ctx)
        owner, repo = parse_github_slug(repository.url)
        return await _client().get_issue(owner, repo, input.number)


class GitHubCommentIssueTool(Tool):
    name = "github.comment_issue"
    description = "Post a comment on the task's issue (upstream) — used to link the PR."
    input_schema = CommentIssueInput
    output_schema = CommentIssueOutput
    capabilities: list[str] = ["github.write"]
    risk = "MEDIUM"

    async def execute(
        self, input: CommentIssueInput, ctx: ExecutionContext
    ) -> CommentIssueOutput:
        _, repository = _task_and_repository(ctx)
        owner, repo = parse_github_slug(repository.url)
        await _client().comment_on_issue(owner, repo, input.number, input.body)
        return CommentIssueOutput(number=input.number, posted=True)


class GitHubCreatePrTool(Tool):
    name = "github.create_pr"
    description = (
        "Open a DRAFT pull request on the repository's FORK, from the task's "
        "worktree branch onto the fork's default branch. The target is "
        "resolved server-side from repositories.fork_url — never the upstream."
    )
    input_schema = CreatePrInput
    output_schema = CreatePrOutput
    capabilities: list[str] = ["github.write"]
    risk = "HIGH"

    async def execute(
        self, input: CreatePrInput, ctx: ExecutionContext
    ) -> CreatePrOutput:
        _, repository = _task_and_repository(ctx)
        wt = _worktree(ctx, input.worktree_id)
        fork_owner, fork_repo = _fork_slug_parts(repository)
        base = repository.default_branch or "main"
        pr: PRData = await _client().create_pr(
            fork_owner,
            fork_repo,
            head=wt.branch_name,
            base=base,
            title=input.title,
            body=input.body,
            draft=True,
        )
        return CreatePrOutput(
            repo=pr.repo,
            branch=pr.branch,
            number=pr.number,
            url=pr.url,
            status=pr.status,
        )


GITHUB_TOOLS: list[Tool] = [
    GitHubGetIssueTool(),
    GitHubCommentIssueTool(),
    GitHubCreatePrTool(),
]
