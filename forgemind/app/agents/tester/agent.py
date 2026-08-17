"""The Test Agent (architecture doc sections 9, 41, and E).

The one agent with NO LLM call at all. Section 41's principle — "process
exit code + structured test parser", not "LLM decides" — applies most
literally here: run the repository's configured test command exactly once
via ``shell.run_test`` (through the pipeline, for audit), parse the exit
code + output deterministically, persist a ``TestRun``, done. There is no
judgment call anywhere in this agent, so there is nothing for an LLM to
hallucinate and nothing for a prompt injection to redirect.
"""

from __future__ import annotations

import logging
import uuid
from typing import ClassVar

from app.agents.base import Agent
from app.agents.tester.schema import TestResult, parse_test_run
from app.execution import ToolPipeline
from app.models import Failure, TestRun, Task
from app.tools.base import ExecutionContext

logger = logging.getLogger(__name__)

# Test output is untrusted data persisted for the trace and fed to the
# Debugger — truncated, never trusted as instructions.
MAX_PERSISTED_OUTPUT = 200_000


class TestError(RuntimeError):
    """The test run could not be produced at all (no pipeline, no DB)."""


class TestAgent(Agent):
    name: ClassVar[str] = "tester"
    description: ClassVar[str] = (
        "Runs the repository's configured test command and parses the result."
    )
    capabilities: ClassVar[list[str]] = ["shell.test"]

    async def run(self, task: Task, worktree, ctx: ExecutionContext) -> TestResult:
        """One deterministic ``shell.run_test`` call, parsed and persisted."""
        if ctx.db is None:
            raise TestError("ExecutionContext.db is required for testing")
        db = ctx.db

        result = await ToolPipeline(db).invoke(
            "shell.run_test",
            {"worktree_id": str(worktree.id)},
            set(self.capabilities),
            ctx,
        )

        if result.status != "EXECUTED":
            # DENIED (shouldn't happen — the tester holds shell.test) or a
            # FAILED tool call (e.g. no configured test_command): the run
            # errored, distinct from a clean failing exit code.
            logger.warning(
                "shell.run_test not executed for task %s: %s (%s)",
                task.id, result.status, result.error or result.denial_reason,
            )
            parsed = TestResult(status="error", exit_code=None)
            self._persist(
                db, task, worktree, parsed,
                output=result.error or "", duration_ms=0, timed_out=False,
            )
            return parsed

        parsed = parse_test_run(
            exit_code=result.output["exit_code"],
            output=result.output["output"],
            timed_out=result.output["timed_out"],
        )
        self._persist(
            db,
            task,
            worktree,
            parsed,
            output=result.output["output"],
            duration_ms=result.output["duration_ms"],
            timed_out=result.output["timed_out"],
        )
        logger.info(
            "Test run for task %s: %s (%d passed, %d failed, %dms)",
            task.id, parsed.status, parsed.passed, parsed.failed,
            result.output["duration_ms"],
        )
        return parsed

    def _persist(
        self,
        db,
        task: Task,
        worktree,
        result: TestResult,
        *,
        output: str,
        duration_ms: int,
        timed_out: bool,
    ) -> None:
        row = TestRun(
            task_id=task.id,
            worktree_id=getattr(worktree, "id", None),
            status=result.status,
            passed=result.passed,
            failed=result.failed,
            duration_ms=duration_ms,
            exit_code=result.exit_code,
            timed_out=timed_out,
            output=output[:MAX_PERSISTED_OUTPUT] or None,
        )
        db.add(row)
        db.flush()
        for failure in result.failures:
            db.add(Failure(test_run_id=row.id, test=failure.test, output=failure.output))
        db.commit()


def build_tester() -> TestAgent:
    """Construct the tester — no provider needed (zero LLM calls)."""
    return TestAgent()
