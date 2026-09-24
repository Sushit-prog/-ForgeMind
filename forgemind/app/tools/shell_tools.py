"""``shell.*`` tools (architecture doc section F, Phases 8 and 13).

``shell.run_test`` is the FIRST tool that executes a subprocess. Its input
schema deliberately contains NO command field: the command comes exclusively
from ``repositories.test_command`` (detected + validated at discovery time,
re-validated by the runner at invocation). ``extra="forbid"`` on the input
schema means even a smuggled extra argument (e.g. a hostile agent trying to
inject a command) is rejected at validation before anything runs — proving
structurally that there is no agent-input path into the command.

``shell.install_deps`` (Phase 13) has the SAME no-command shape and posture:
the install command comes exclusively from ``repositories.install_command``
(server-side, validated by ``install_policy``), run inside the task's own
venv. Its risk is MEDIUM (a subprocess that writes into the venv — and via
``pip install -e`` mutates the worktree's install metadata), one tier above
``shell.run_test``'s read-only LOW.
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

    async def execute(
        self, input: RunTestInput, ctx: ExecutionContext
    ) -> RunTestOutput:
        if ctx.db is None:
            raise RuntimeError("ExecutionContext.db is required for shell tools")

        manager = WorktreeManager(ctx.db)
        path = manager.path_for(input.worktree_id)
        wt = ctx.db.get(Worktree, input.worktree_id)
        if wt is None:
            raise RuntimeError(f"worktree row missing for {input.worktree_id}")
        repository = ctx.db.get(Repository, wt.repository_id)
        if repository is None:
            raise RuntimeError(
                f"repository row missing for worktree {input.worktree_id}"
            )

        from app.config import get_settings
        from app.shell.provision import venv_bin_dir, venv_path_for

        # Phase 13: run the suite with the task's OWN venv on PATH (when it
        # exists — repos without install_command fall back to ambient PATH).
        venv_bin = venv_bin_dir(
            venv_path_for(repository.id, wt.task_id, manager.cache_dir)
        )
        runner = CommandRunner(
            path,
            repository.test_command,
            get_settings().test_timeout_seconds,
            venv_bin=venv_bin,
        )
        # subprocess.run with up to test_timeout_seconds — a full pipeline
        # stage of blocking I/O; must never run on the event loop (Fix 3).
        import asyncio

        result = await asyncio.to_thread(
            runner.run
        )  # raises TestCommandError for a mis-stored command
        return RunTestOutput(
            exit_code=result.exit_code,
            output=result.output,
            timed_out=result.timed_out,
            duration_ms=result.duration_ms,
        )


class InstallDepsInput(BaseModel):
    """The only input: which worktree to provision. No command, no arguments."""

    model_config = ConfigDict(extra="forbid")

    worktree_id: uuid.UUID


class InstallDepsOutput(BaseModel):
    """Whether anything was installed and what pip/venv said."""

    installed: bool
    skipped: bool = False
    exit_code: int | None
    output: str
    timed_out: bool
    duration_ms: int


class InstallDepsTool(Tool):
    name = "shell.install_deps"
    description = (
        "Provision the task's per-worktree venv and install the repository's "
        "declared dependencies. The command is server-side configuration "
        "(repositories.install_command), never agent input."
    )
    input_schema = InstallDepsInput
    output_schema = InstallDepsOutput
    capabilities: list[str] = ["shell.install"]
    risk = "MEDIUM"

    async def execute(
        self, input: InstallDepsInput, ctx: ExecutionContext
    ) -> InstallDepsOutput:
        if ctx.db is None:
            raise RuntimeError("ExecutionContext.db is required for shell tools")

        from app.shell.provision import install_dependencies

        # venv creation + pip = blocking subprocess I/O; never on the event
        # loop (Fix 3), exactly like shell.run_test.
        import asyncio

        result = await asyncio.to_thread(install_dependencies, ctx.db, input.worktree_id)
        return InstallDepsOutput(
            installed=result.installed,
            skipped=result.skipped,
            exit_code=result.exit_code,
            output=result.output,
            timed_out=result.timed_out,
            duration_ms=result.duration_ms,
        )


SHELL_TOOLS: list[Tool] = [RunTestTool(), InstallDepsTool()]
