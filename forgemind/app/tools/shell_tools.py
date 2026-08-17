"""``shell.*`` tools (architecture doc section F, Phase 8).

``shell.run_test`` is the FIRST tool that executes a subprocess. Its input
schema deliberately contains NO command field: the command comes exclusively
from ``repositories.test_command`` (detected + validated at discovery time,
re-validated by the runner at invocation). ``extra="forbid"`` on the input
schema means even a smuggled extra argument (e.g. a hostile agent trying to
inject a command) is rejected at validation before anything runs — proving
structurally that there is no agent-input path into the command.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict

from app.git.worktree_manager import WorktreeManager
from app.models import Repository, Worktree
from app.shell.runner import CommandRunner
from app.tools.base import ExecutionContext, Tool


class RunTestInput(BaseModel):
    """The only input: which worktree to test. No command, no arguments."""

    model_config = ConfigDict(extra="forbid")

    worktree_id: uuid.UUID


class RunTestOutput(BaseModel):
    exit_code: int | None
    output: str
    timed_out: bool
    duration_ms: int


class RunTestTool(Tool):
    name = "shell.run_test"
    description = (
        "Run the repository's configured test command inside the worktree. "
        "The command is server-side configuration, never agent input."
    )
    input_schema = RunTestInput
    output_schema = RunTestOutput
    capabilities: list[str] = ["shell.test"]
    risk = "LOW"

    async def execute(self, input: RunTestInput, ctx: ExecutionContext) -> RunTestOutput:
        if ctx.db is None:
            raise RuntimeError("ExecutionContext.db is required for shell tools")

        manager = WorktreeManager(ctx.db)
        path = manager.path_for(input.worktree_id)
        wt = ctx.db.get(Worktree, input.worktree_id)
        if wt is None:
            raise RuntimeError(f"worktree row missing for {input.worktree_id}")
        repository = ctx.db.get(Repository, wt.repository_id)
        if repository is None:
            raise RuntimeError(f"repository row missing for worktree {input.worktree_id}")

        from app.config import get_settings

        runner = CommandRunner(
            path,
            repository.test_command,
            get_settings().test_timeout_seconds,
        )
        result = runner.run()  # raises TestCommandError for a mis-stored command
        return RunTestOutput(
            exit_code=result.exit_code,
            output=result.output,
            timed_out=result.timed_out,
            duration_ms=result.duration_ms,
        )


SHELL_TOOLS: list[Tool] = [RunTestTool()]
