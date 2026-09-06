"""``github.*`` tools (architecture doc section F, Phase 10 + Phase 12).

- ``github.get_issue``     (github.read,  LOW)    — read an UPSTREAM issue.
- ``github.comment_issue`` (github.write, MEDIUM) — comment on an issue
  (used to link the PR back to the source issue).
- ``github.create_pr``     (github.write, HIGH)   — the first genuinely
  HIGH-risk tool, and the phase's approval-gated surface. It opens a DRAFT
  PR on the FORK.
- ``github.merge_pr``      (github.merge, HIGH)    — Phase 12's GATED
  squash merge. Only reachable through ``POST /tasks/{id}/merge``, never
  through an autonomous agent loop.

Security posture, enforced here (never by convention):

- Every tool resolves its target SERVER-SIDE from the task's repository
  row. There is no ``owner``/``repo``/``head``/``base`` input field at all,
  so no agent (or injected prompt) can redirect a call to an arbitrary
  repo — the upstream reference is used only for READS (get_issue /
  comment link), and PR creation targets ``repositories.fork_url`` or fails
  closed. ``repositories.fork_url == repositories.url`` is a SecurityError.
- ``github.create_pr`` opens a draft by default and the agent always does —
  the second safety layer under the AWAITING_APPROVAL human gate.
- ``github.merge_pr`` enforces THREE hard gates inside ``execute`` before
  it ever calls the GitHub API:
  1. The task has an ``Approval`` row with ``action="approve"`` AND the
     task status is ``COMPLETED`` (post-approval).
  2. The fork slug (``pull_requests.repo``) is in
     ``settings.merge_allowed_repo_set`` (empty = everything denied).
  3. A fresh staleness check (VERIFICATION's pattern): the PR is still
     open, GitHub reports ``mergeable == True``, and the current base SHA
     matches what was captured when the PR was created
     (``pull_requests.base_sha``). No rebase is needed.
  Only then is ``client.merge_pr`` reached (squash merge only).
- The policy engine's ``ExplicitDenyRule`` denies the name ``github.merge``
  by name as a second layer (belt-and-suspenders) — our tool is
  ``github.merge_pr``, so this denial does NOT apply. The capability gate
  (``github.merge``) is the first gate; the three checks above are the
  second.
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
    base_sha: str | None = None


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
            base_sha=pr.base_sha,
        )


# -- Phase 12: gated squash merge -------------------------------------------


class MergePrInput(BaseModel):
    """The ONLY input the caller may give: which task's PR to merge.

    Everything else — the PR number, fork slug, approval check, MERGE_ALLOWED_REPOS,
    staleness check — is resolved SERVER-SIDE. ``extra="forbid"`` rejects a
    smuggled ``owner``/``repo``/``number`` field at validation.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: uuid.UUID


class MergePrOutput(BaseModel):
    """Structured output for the merge action. Business denials (not
    approved, not allowlisted, stale, already merged) return ``merged=False``
    with a specific ``denial_reason`` — the tool does NOT raise. Genuine
    GitHub API errors raise and become ``FAILED`` (pipeline writes the row).
    The endpoint translates both to a human-facing ``AuditLog`` row."""

    merged: bool
    merge_commit_sha: str | None = None
    denial_reason: str | None = None
    repo: str | None = None
    pr_number: int | None = None


class GitHubMergePrTool(Tool):
    """Gated squash-merge of the task's draft PR.

    Only reachable through ``POST /tasks/{id}/merge`` (the human-operator
    merge endpoint) — never through an autonomous agent loop.  Three hard
    gates inside ``execute`` enforce fail-closed security:

    1. **Approval**: latest ``Approvals`` row for the task must have
       ``action="approve"`` AND the task must be in ``COMPLETED`` (the
       post-approval terminal state).  The tool re-checks this independently
       of the endpoint's own check (defense in depth).
    2. **Repo allowlist**: the fork slug (``pull_requests.repo``) must be in
       ``settings.merge_allowed_repo_set``.  Empty = everything denied.
    3. **Staleness** (VERIFICATION's pattern): re-fetch the PR via the
       GitHub API (the fresh source of truth) and compare against the
       persisted ``pull_requests.base_sha`` — must be equal.  Also require
       PR ``state="open"`` and ``mergeable=True``.

    Only after all three gates pass is ``client.merge_pr`` called (squash
    merge only).  On success: ``pull_requests.merged_at`` and
    ``pull_requests.merge_commit_sha`` are set atomically.
    """

    name = "github.merge_pr"
    description = (
        "Gated squash-merge of the task's PR into its fork. Requires: "
        "approval evidence, the fork in MERGE_ALLOWED_REPOS, and a fresh "
        "staleness check (PR still open, base unchanged). Returns merged: "
        "True + sha, or merged: False + a specific denial reason."
    )
    input_schema = MergePrInput
    output_schema = MergePrOutput
    capabilities: list[str] = ["github.merge"]
    risk = "HIGH"

    async def execute(
        self, input: MergePrInput, ctx: ExecutionContext
    ) -> MergePrOutput:
        if ctx.db is None or ctx.task_id is None:
            raise GitHubConfigError("github.merge_pr requires a task context (db)")

        from sqlalchemy import select

        from app.models import PullRequest, Task
        from app.models.base import utcnow
        from app.config import get_settings

        task = ctx.db.get(Task, ctx.task_id)
        if task is None:
            raise GitHubConfigError(f"task {ctx.task_id} not found")

        pr = ctx.db.scalar(
            select(PullRequest)
            .where(PullRequest.task_id == task.id)
            .order_by(PullRequest.created_at.desc(), PullRequest.id.desc())
            .limit(1)
        )
        if pr is None:
            raise GitHubConfigError(f"no pull request recorded for task {task.id}")

        # -- Gate 1: already merged? --
        if pr.merged_at is not None:
            return MergePrOutput(
                merged=False,
                denial_reason="PR already merged",
                repo=pr.repo,
                pr_number=pr.number,
            )

        # -- Gate 2: approved? --
        from app.models import Approval as ApprovalModel

        latest_approval = ctx.db.scalar(
            select(ApprovalModel)
            .where(ApprovalModel.task_id == task.id)
            .order_by(ApprovalModel.created_at.desc(), ApprovalModel.id.desc())
            .limit(1)
        )
        task_status_ok = task.status == "COMPLETED"
        approval_ok = (
            latest_approval is not None and latest_approval.action == "approve"
        )
        if not (task_status_ok and approval_ok):
            return MergePrOutput(
                merged=False,
                denial_reason=(
                    "task not approved"
                    if not approval_ok
                    else f"task status is {task.status!r}, expected COMPLETED"
                ),
                repo=pr.repo,
                pr_number=pr.number,
            )

        # -- Gate 3: repo in MERGE_ALLOWED_REPOS? --
        settings = get_settings()
        allowed = settings.merge_allowed_repo_set
        repo_slug = pr.repo.strip().lower()
        if repo_slug not in allowed:
            return MergePrOutput(
                merged=False,
                denial_reason=(
                    f"repo {pr.repo!r} not in MERGE_ALLOWED_REPOS "
                    f"(allowed: {sorted(allowed) or '(empty — merge disabled everywhere)'})"
                ),
                repo=pr.repo,
                pr_number=pr.number,
            )

        # -- Gate 4: fresh staleness check (VERIFICATION pattern) --
        _, repository = _task_and_repository(ctx)
        if not repository.fork_url:
            raise GitHubConfigError(
                "repository.fork_url is unset — cannot resolve fork for merge"
            )
        from app.github.slug import parse_github_slug

        fork_owner, fork_repo = parse_github_slug(repository.fork_url)

        try:
            fresh = await _client().get_pr(fork_owner, fork_repo, pr.number)
        except Exception as exc:  # noqa: BLE001 — API error at staleness = fail closed
            return MergePrOutput(
                merged=False,
                denial_reason=f"staleness check failed: {exc}",
                repo=pr.repo,
                pr_number=pr.number,
            )

        # PR must still be open.
        if fresh.state and fresh.state not in ("open", "draft"):
            return MergePrOutput(
                merged=False,
                denial_reason=f"PR is {fresh.state!r}, expected open",
                repo=pr.repo,
                pr_number=pr.number,
            )

        # GitHub must report mergeable == True.
        if fresh.mergeable is False:
            return MergePrOutput(
                merged=False,
                denial_reason=(
                    f"GitHub reports mergeable=False "
                    f"(mergeable_state={fresh.mergeable_state!r})"
                ),
                repo=pr.repo,
                pr_number=pr.number,
            )

        # Base-branch drift: current base_sha must match persisted base_sha.
        if pr.base_sha is not None and fresh.base_sha != pr.base_sha:
            return MergePrOutput(
                merged=False,
                denial_reason=(
                    f"base branch drifted since PR was opened "
                    f"(expected {pr.base_sha}, got {fresh.base_sha})"
                ),
                repo=pr.repo,
                pr_number=pr.number,
            )

        # -- All gates passed: execute the merge. --
        merge_sha = await _client().merge_pr(
            fork_owner, fork_repo, pr.number, merge_method="squash"
        )

        pr.merged_at = utcnow()
        pr.merge_commit_sha = merge_sha
        ctx.db.commit()

        return MergePrOutput(
            merged=True,
            merge_commit_sha=merge_sha,
            repo=pr.repo,
            pr_number=pr.number,
        )


GITHUB_TOOLS: list[Tool] = [
    GitHubGetIssueTool(),
    GitHubCommentIssueTool(),
    GitHubCreatePrTool(),
    GitHubMergePrTool(),
]
