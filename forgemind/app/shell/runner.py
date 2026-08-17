"""Subprocess execution for ``shell.run_test`` (architecture doc section F).

The ONLY place this phase executes code: an argument-list ``subprocess.run``
(never ``shell=True``), run in the task's worktree with a hard timeout and
captured stdout/stderr. The command comes from ``repositories.test_command``
— validated at store time by ``command_policy`` and RE-validated here before
execution, so a mis-stored or legacy value (e.g. discovery that predates the
policy) is refused with a clear error instead of executed. There is no agent
input path into this module at all.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from app.shell.command_policy import TestCommandError, validate_test_command

# Upper bound on captured output (defense against a test that floods stdout).
MAX_CAPTURED_OUTPUT = 500_000


class CommandResult:
    """The raw subprocess outcome — pre-parse, pre-LLM ground truth."""

    def __init__(
        self,
        *,
        exit_code: int | None,
        output: str,
        timed_out: bool = False,
        duration_ms: int,
        error: str | None = None,
    ) -> None:
        self.exit_code = exit_code
        self.output = output[:MAX_CAPTURED_OUTPUT]
        self.timed_out = timed_out
        self.duration_ms = duration_ms
        self.error = error


class CommandRunner:
    """Run the repository's validated test command inside a worktree."""

    def __init__(self, worktree_path: Path, test_command: str, timeout_seconds: float) -> None:
        self.worktree_path = worktree_path
        self.test_command = test_command
        self.timeout_seconds = timeout_seconds

    def run(self) -> CommandResult:
        # Re-validate the STORED value (not agent input — there is no agent
        # input): a legacy/mis-stored command must never execute.
        if not self.test_command or not self.test_command.strip():
            raise TestCommandError(
                "no validated test_command for this repository — discovery never "
                "stored one; run discovery before invoking shell.run_test"
            )
        tokens = validate_test_command(self.test_command)

        started = time.perf_counter()
        try:
            proc = subprocess.run(
                tokens,
                cwd=str(self.worktree_path),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            duration = int((time.perf_counter() - started) * 1000)
            partial = (exc.stdout or "") + (exc.stderr or "")
            return CommandResult(
                exit_code=None,
                timed_out=True,
                duration_ms=duration,
                output=partial,
            )
        except OSError as exc:
            duration = int((time.perf_counter() - started) * 1000)
            return CommandResult(
                exit_code=None,
                duration_ms=duration,
                output="",
                error=f"failed to run test command: {exc}",
            )

        duration = int((time.perf_counter() - started) * 1000)
        output = (proc.stdout or "") + (proc.stderr or "")
        return CommandResult(
            exit_code=proc.returncode,
            duration_ms=duration,
            output=output,
        )
