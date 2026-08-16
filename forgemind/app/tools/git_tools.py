"""``git.*`` tools (architecture doc section F).

Reads are gated on ``git.read``, writes (``git.create_branch``,
``git.commit``) on ``git.write`` — enforced by the pipeline. No push tool
exists in this phase; nothing here ever touches the default branch.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from app.git.operations import CommitInfo, GitOperations, GitStatus
from app.git.worktree_manager import WorktreeManager
from app.models import Worktree
from app.tools.base import ExecutionContext, Tool


class WorktreeInput(BaseModel):
    worktree_id: uuid.UUID


class StatusOutput(BaseModel):
    status: GitStatus


class DiffInput(WorktreeInput):
    staged: bool = False


class DiffOutput(BaseModel):
    diff: str


class LogInput(WorktreeInput):
    limit: int = Field(default=20, ge=1, le=100)


class LogOutput(BaseModel):
    commits: list[CommitInfo]


class CreateBranchInput(WorktreeInput):
    name: str = Field(min_length=1, max_length=255, pattern=r"^[^\s~^:?*\[\]\\]+$")


class CreateBranchOutput(BaseModel):
    branch: str


class CommitInput(WorktreeInput):
    message: str = Field(min_length=1, max_length=10_000)


class CommitOutput(BaseModel):
    sha: str


def _ops_for(ctx: ExecutionContext, worktree_id: uuid.UUID) -> tuple[GitOperations, Worktree]:
    """Resolve the worktree row + path; tools fail clearly without a DB."""
    if ctx.db is None:
        raise RuntimeError("ExecutionContext.db is required for git tools")
    manager = WorktreeManager(ctx.db)
    path = manager.path_for(worktree_id)
    wt = ctx.db.get(Worktree, worktree_id)
    return GitOperations(path, base_commit=wt.base_commit if wt else None), wt


class StatusTool(Tool):
    name = "git.status"
    description = "Worktree status: branch, staged/unstaged/untracked files."
    input_schema = WorktreeInput
    output_schema = StatusOutput
    capabilities: list[str] = ["git.read"]
    risk = "LOW"

    async def execute(self, input: WorktreeInput, ctx: ExecutionContext) -> StatusOutput:
        ops, _ = _ops_for(ctx, input.worktree_id)
        return StatusOutput(status=ops.status())


class DiffTool(Tool):
    name = "git.diff"
    description = "Worktree diff (unstaged by default, or staged with staged=true)."
    input_schema = DiffInput
    output_schema = DiffOutput
    capabilities: list[str] = ["git.read"]
    risk = "LOW"

    async def execute(self, input: DiffInput, ctx: ExecutionContext) -> DiffOutput:
        ops, _ = _ops_for(ctx, input.worktree_id)
        return DiffOutput(diff=ops.diff(staged=input.staged))


class LogTool(Tool):
    name = "git.log"
    description = "Recent commit history of the worktree branch."
    input_schema = LogInput
    output_schema = LogOutput
    capabilities: list[str] = ["git.read"]
    risk = "LOW"

    async def execute(self, input: LogInput, ctx: ExecutionContext) -> LogOutput:
        ops, _ = _ops_for(ctx, input.worktree_id)
        return LogOutput(commits=ops.log(limit=input.limit))


class CreateBranchTool(Tool):
    name = "git.create_branch"
    description = "Create a branch at the worktree's base commit (never main)."
    input_schema = CreateBranchInput
    output_schema = CreateBranchOutput
    capabilities: list[str] = ["git.write"]
    risk = "MEDIUM"

    async def execute(self, input: CreateBranchInput, ctx: ExecutionContext) -> CreateBranchOutput:
        ops, _ = _ops_for(ctx, input.worktree_id)
        return CreateBranchOutput(branch=ops.create_branch(input.name))


class CommitTool(Tool):
    name = "git.commit"
    description = "Stage all worktree changes and commit with a fixed system identity."
    input_schema = CommitInput
    output_schema = CommitOutput
    capabilities: list[str] = ["git.write"]
    risk = "MEDIUM"

    async def execute(self, input: CommitInput, ctx: ExecutionContext) -> CommitOutput:
        ops, _ = _ops_for(ctx, input.worktree_id)
        return CommitOutput(sha=ops.commit(input.message))


GIT_TOOLS: list[Tool] = [StatusTool(), DiffTool(), LogTool(), CreateBranchTool(), CommitTool()]
