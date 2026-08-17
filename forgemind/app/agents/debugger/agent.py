"""The Debugger Agent (architecture doc sections 10 and E).

A read-only tool-use loop (the Phase 6/7 skeleton: propose -> pipeline ->
observation -> bounded -> forced synthesis) that investigates a failing
test run, then produces a ``FailureClassification`` for the task_lifecycle
branching.

Two structural facts worth stating up front:

1. FLAKINESS IS OBSERVED, NOT GUESSED. Before any LLM classification, the
   Debugger re-runs the suite EXACTLY ONCE via the Test Agent (the only
   place ``shell.test`` may be invoked — the Debugger itself holds no
   shell capability). If the re-run passes, the classification is
   FLAKY_TEST, set deterministically — the LLM never gets to guess "flaky"
   from a single failing run (and is explicitly forbidden from it in the
   prompt).

2. DENIED results are Research-style (expected, benign), NOT
   Developer-style (unexpected). The Debugger is read-only by contract, so
   a write proposal is a harmless mistake, not a sign of reaching outside
   its job — the loop feeds the denial back as an observation and keeps
   going. It is still audited loudly (``debugger.unexpected_denial``) so
   the boundary is provably enforced and the trace shows it.
"""

from __future__ import annotations

import logging
import uuid
from typing import ClassVar

from pydantic import BaseModel, Field, model_validator

from app.agents.base import Agent, structured_output_with_retries
from app.agents.debugger.prompt import (
    build_classification_correction,
    build_classification_messages,
    build_debugger_messages,
    observation_message,
)
from app.agents.debugger.schema import FailureClassification
from app.agents.tester.agent import TestAgent
from app.agents.tester.schema import TestResult, result_from_row
from app.config import get_settings
from app.execution import ToolPipeline
from app.llm.errors import LLMMalformedOutputError
from app.llm.provider import LLMProvider, Message
from app.models import AuditLog
from app.models import FailureClassification as FailureClassificationRow
from app.models import TestRun
from app.models import Task
from app.tools.base import ExecutionContext

logger = logging.getLogger(__name__)

# Tools the agent may propose (the pipeline still enforces the read-only
# capability set — this list is a prompt guard, not the boundary).
DEBUGGER_TOOL_NAMES = frozenset(
    {
        "repository.read_file",
        "repository.search",
        "repository.list_files",
        "git.status",
        "git.diff",
        "git.log",
    }
)


class DebuggerError(RuntimeError):
    """The failure could not be classified (hard failure — task FAILED)."""


class DebuggerToolCall(BaseModel):
    tool: str = Field(min_length=1)
    input: dict = Field(default_factory=dict)


class ToolCallProposal(BaseModel):
    tool_call: DebuggerToolCall | None = None
    final: bool = False

    @model_validator(mode="after")
    def _exactly_one(self) -> "ToolCallProposal":
        if self.tool_call is None and not self.final:
            raise ValueError("proposal must be a tool_call or final")
        if self.tool_call is not None and self.final:
            raise ValueError("proposal cannot be both a tool_call and final")
        return self


class Observation(BaseModel):
    tool: str
    status: str  # EXECUTED | DENIED | FAILED
    input: dict = Field(default_factory=dict)
    output: dict | None = None
    error: str | None = None
    denial_reason: str | None = None


class DebuggerAgent(Agent):
    name: ClassVar[str] = "debugger"
    description: ClassVar[str] = (
        "Investigates a failing test run and classifies why it failed."
    )
    # Read-only by contract (Section 10): no repo.write, no git.write, and
    # NO shell capability — the flakiness re-run goes through the Test Agent,
    # never through a tool the Debugger itself could propose.
    capabilities: ClassVar[list[str]] = ["repo.read", "git.read"]

    def __init__(
        self,
        provider: LLMProvider,
        *,
        max_tool_calls: int | None = None,
        timeout_retries: int | None = None,
        backoff_base_seconds: float = 0.5,
    ) -> None:
        self.provider = provider
        settings = get_settings()
        self.max_tool_calls = (
            settings.max_debugger_tool_calls
            if max_tool_calls is None
            else max_tool_calls
        )
        self.timeout_retries = (
            settings.llm_max_retries if timeout_retries is None else timeout_retries
        )
        self.backoff_base = backoff_base_seconds

    # -- the public contract -------------------------------------------------

    async def run(
        self,
        task: Task,
        test_result: TestResult,
        implementation,
        ctx: ExecutionContext,
    ) -> FailureClassification:
        """Investigate the failure (read-only), re-run once for flakiness,
        then classify. Returns a persisted ``FailureClassification``.

        ``test_result`` may be a Pydantic ``TestResult`` or a ``TestRun``
        row (the lifecycle passes either). ``implementation`` is the
        developer's summary (row or Pydantic — duck-typed).
        """
        if ctx.db is None:
            raise DebuggerError("ExecutionContext.db is required for debugging")
        db = ctx.db
        worktree = self._ensure_worktree(db, task)

        # Re-run check (Section 10): flakiness is observed, never guessed.
        rerun = await self._rerun_for_flakiness(db, task, worktree, ctx)
        first_run_id = getattr(test_result, "id", None) or self._latest_run_id(db, task)

        if rerun is not None and rerun.status == "passed":
            # The re-run passed — the original failure was intermittent.
            # Classified deterministically; the LLM is not consulted.
            classification = FailureClassification(
                category="FLAKY_TEST",
                root_cause=(
                    "The test suite failed on the first run but PASSED the "
                    "single re-run — the failure was intermittent, not a "
                    "defect in the implementation."
                ),
                fix_instruction=None,
                fixable=False,
                is_flaky=True,
            )
            self._audit(
                db,
                task.id,
                "debugger.flaky_detected",
                {
                    "first_run_id": str(first_run_id) if first_run_id else None,
                    "rerun_id": str(rerun.id) if rerun.id else None,
                    "note": "flaky failures never block the pipeline but are "
                    "never silently dropped from the trace",
                },
            )
            self._persist(db, task, classification, test_run_id=first_run_id)
            return classification

        # Normal classification path. If the re-run errored DIFFERENTLY than
        # the first run (timeout vs clean failure), that is not "flaky" —
        # it is a different failure mode; classify from the more informative
        # of the two runs and note the inconsistency.
        informative = self._more_informative(test_result, rerun)
        inconsistency = self._rerun_inconsistency(test_result, rerun)

        messages = build_debugger_messages(
            task,
            informative,
            implementation,
            fix_instruction=getattr(implementation, "fix_instruction", None),
        )
        if inconsistency:
            messages.append(
                Message(
                    role="user",
                    content=(
                        "<note>NOTE: the flakiness re-run produced a DIFFERENT "
                        f"failure mode than the first run: {inconsistency}. This is "
                        "not flakiness. Classify from the more informative run and "
                        "note the inconsistency in root_cause.</note>"
                    ),
                )
            )

        observations: list[Observation] = []
        for _ in range(self.max_tool_calls):
            try:
                proposal: ToolCallProposal = await structured_output_with_retries(
                    self.provider,
                    messages,
                    ToolCallProposal,
                    timeout_retries=self.timeout_retries,
                    backoff_base_seconds=self.backoff_base,
                )
            except LLMMalformedOutputError as exc:
                messages.append(
                    Message(
                        role="user",
                        content=(
                            f"Your response was invalid ({exc.detail}). Respond with JSON "
                            '{"tool_call": {"tool": "...", "input": {...}}} or {"final": true}.'
                        ),
                    )
                )
                continue

            if proposal.final:
                return await self._classify(
                    db, task, messages, observations, informative,
                    first_run_id, forced=False,
                )

            obs = await self._execute_tool(proposal.tool_call, worktree.id, db, task.id)
            observations.append(obs)
            messages.append(observation_message(obs))

        # Budget exhausted: force the classification (like Research's forced
        # synthesis) — a classification is required for routing, and the
        # LLM has seen the failure data.
        logger.warning(
            "Debugger tool budget (%d) exhausted for task %s — forcing classification",
            self.max_tool_calls, task.id,
        )
        db.add(
            AuditLog(
                task_id=task.id,
                actor=self.name,
                action="debugger.budget_exhausted",
                entity_type="task",
                entity_id=str(task.id),
                details={"max_tool_calls": self.max_tool_calls},
            )
        )
        db.commit()
        return await self._classify(
            db, task, messages, observations, informative,
            first_run_id, forced=True,
        )

    # -- internals -----------------------------------------------------------

    def _ensure_worktree(self, db, task):
        from app.git.worktree_manager import WorktreeManager

        return WorktreeManager(db).get_or_create_for_task(task)

    async def _rerun_for_flakiness(self, db, task, worktree, ctx):
        """Run the suite EXACTLY once more via the Test Agent.

        Returns the second ``TestRun`` row (persisted by the Test Agent so
        the trace records both runs), or None if the re-run itself could
        not be produced. A re-run that passes -> FLAKY. A re-run that fails
        the same way -> normal classification. A re-run that fails
        DIFFERENTLY (error vs failed, timeout vs clean exit) -> classified
        from the more informative run, inconsistency noted.
        """
        # A fresh tester-flavored context: the re-run's shell.run_test row is
        # attributed to the TESTER in the trace, not the debugger.
        tester_ctx = ExecutionContext(task_id=task.id, agent_type="tester", db=ctx.db)
        try:
            await TestAgent().run(task, worktree, tester_ctx)
        except Exception as exc:  # noqa: BLE001 — a failed re-run is not a crash
            logger.warning("debugger flakiness re-run failed for task %s: %s", task.id, exc)
            return None
        rerun = self._latest_run(db, task)
        if rerun is None:
            return None
        logger.info(
            "Debugger flakiness re-run for task %s: %s (%d passed, %d failed)",
            task.id, rerun.status, rerun.passed, rerun.failed,
        )
        return rerun

    def _latest_run(self, db, task) -> TestRun | None:
        from sqlalchemy import select

        return db.scalar(
            select(TestRun)
            .where(TestRun.task_id == task.id)
            .order_by(TestRun.created_at.desc(), TestRun.id.desc())
            .limit(1)
        )

    def _latest_run_id(self, db, task) -> uuid.UUID | None:
        run = self._latest_run(db, task)
        return run.id if run is not None else None

    def _more_informative(self, first, rerun):
        """The run that tells the Debugger more about the failure.

        ``failed`` (tests ran and failed) beats ``error`` (the run itself
        broke); a rerun that gives a clean signal beats a first run that
        did not. Returns a Pydantic ``TestResult``.
        """
        if rerun is not None:
            rerun_result = result_from_row(rerun)
            first_result = (
                result_from_row(first) if isinstance(first, TestRun) else first
            )
            if rerun.status == "failed" and first_result.status != "failed":
                return rerun_result
            if first_result.status == "failed" and rerun.status != "failed":
                return first_result
            return rerun_result
        return result_from_row(first) if isinstance(first, TestRun) else first

    def _rerun_inconsistency(self, first, rerun) -> str | None:
        """Describe how the re-run's failure mode differed from the first."""
        if rerun is None:
            return None
        first_result = (
            result_from_row(first) if isinstance(first, TestRun) else first
        )
        if first_result.status == "failed" and rerun.status == "error":
            return (
                f"first run failed with exit {first_result.exit_code}; "
                f"re-run errored (timed_out={rerun.timed_out})"
            )
        if first_result.status == "error" and rerun.status == "failed":
            return f"first run errored; re-run failed with exit {rerun.exit_code}"
        return None

    async def _execute_tool(
        self,
        call: DebuggerToolCall,
        worktree_id: uuid.UUID,
        db,
        task_id: uuid.UUID,
    ) -> Observation:
        """Run ONE proposal through the pipeline under Debugger's read-only
        capabilities. A denial (e.g. a write proposal) is Research-style:
        an expected, benign observation the loop adapts to — but it is
        audited so the boundary is provably enforced."""
        tool_input = dict(call.input)
        tool_input.pop("worktree_id", None)
        tool_input["worktree_id"] = str(worktree_id)

        ctx = ExecutionContext(task_id=task_id, agent_type=self.name, db=db)
        try:
            result = await ToolPipeline(db).invoke(
                call.tool, tool_input, set(self.capabilities), ctx
            )
        except Exception as exc:  # noqa: BLE001 — contract errors surface as FAILED obs
            logger.warning("debugger tool %s raised: %s", call.tool, exc)
            self._audit(
                db, task_id, "debugger.unexpected_denial",
                {"tool": call.tool, "surfaced_as": "error", "error": str(exc)},
            )
            return Observation(
                tool=call.tool, status="FAILED", input=tool_input, error=str(exc)
            )

        if result.status == "DENIED":
            # Read-only agent proposing a write: a harmless mistake the loop
            # absorbs (Research-style) — but audited so the capability
            # boundary is verifiable, not assumed.
            logger.warning(
                "Debugger tool %s denied: %s", call.tool, result.denial_reason
            )
            self._audit(
                db, task_id, "debugger.unexpected_denial",
                {
                    "tool": call.tool,
                    "surfaced_as": "denied",
                    "denial_reason": result.denial_reason,
                },
            )

        return Observation(
            tool=call.tool,
            status=result.status,
            input=tool_input,
            output=result.output,
            error=result.error,
            denial_reason=result.denial_reason,
        )

    async def _classify(
        self,
        db,
        task: Task,
        messages: list[Message],
        observations: list[Observation],
        informative,
        test_run_id,
        *,
        forced: bool,
    ) -> FailureClassification:
        """Produce the final classification: retry ONCE on malformed or
        schema-invalid output; a second failure raises (never a fabricated
        category — the classification drives routing, so guessing is worse
        than failing)."""
        synth = build_classification_messages(messages, forced)
        last_error: str | None = None
        for _ in range(2):  # one correction retry
            try:
                classification: FailureClassification = await structured_output_with_retries(
                    self.provider,
                    synth,
                    FailureClassification,
                    timeout_retries=self.timeout_retries,
                    backoff_base_seconds=self.backoff_base,
                )
            except LLMMalformedOutputError as exc:
                last_error = str(exc)
                synth = build_classification_correction(synth, last_error)
                continue
            self._persist(db, task, classification, test_run_id=test_run_id)
            logger.info(
                "Failure classified for task %s: %s (fixable=%s)",
                task.id, classification.category, classification.fixable,
            )
            return classification

        raise DebuggerError(
            f"Debugger could not produce a valid FailureClassification for task "
            f"{task.id} after the retry-once path: {last_error}"
        )

    def _persist(
        self, db, task: Task, classification: FailureClassification, *, test_run_id
    ) -> None:
        db.add(
            FailureClassificationRow(
                task_id=task.id,
                test_run_id=test_run_id,
                category=classification.category,
                root_cause=classification.root_cause,
                fix_instruction=classification.fix_instruction,
                fixable=classification.fixable,
                is_flaky=classification.is_flaky,
            )
        )
        db.commit()

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


def build_debugger() -> DebuggerAgent:
    """Construct the debugger agent from settings/env (worker entrypoint)."""
    from app.agents.planner.agent import build_provider

    return DebuggerAgent(build_provider(role="debugger"))
