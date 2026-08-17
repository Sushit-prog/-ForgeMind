"""The Reviewer Agent (architecture doc sections 11 and E).

A read-only tool-use loop (the Phase 6/8 skeleton: propose -> pipeline ->
observation -> bounded -> forced synthesis) that independently critiques
the developer's commit from the diff + test result ONLY.

The independence guarantee is STRUCTURAL, not instructional:

- ``run`` takes ``commit_sha`` and ``test_result`` — there is NO
  ImplementationSummary parameter, so the developer's self-reported
  summary/deviations cannot be threaded into the context without changing
  the function signature (the Reviewer prompt builder likewise has no
  summary field).
- The lifecycle passes the commit sha and the TestResult; the Reviewer
  builds its own judgment from the git.diff observation.

Denials are Developer-style (unexpected), not Research-style: the Reviewer
has every capability it legitimately needs (read-only), so a write
proposal (or anything denied) signals the model reaching outside its job
and is audited loudly as ``reviewer.unexpected_denial``.
"""

from __future__ import annotations

import logging
import uuid
from typing import ClassVar

from pydantic import BaseModel, Field, model_validator

from app.agents.base import Agent, structured_output_with_retries
from app.agents.reviewer.prompt import (
    build_reviewer_messages,
    build_verdict_correction,
    build_verdict_messages,
    observation_message,
)
from app.agents.reviewer.schema import ReviewResult
from app.config import get_settings
from app.execution import ToolPipeline
from app.llm.errors import LLMMalformedOutputError
from app.llm.provider import LLMProvider, Message
from app.models import AuditLog
from app.models import ReviewResult as ReviewResultRow
from app.models import Task
from app.tools.base import ExecutionContext

logger = logging.getLogger(__name__)

# Tools the agent may propose (the pipeline still enforces the read-only
# capability set — this list is a prompt guard, not the boundary).
REVIEWER_TOOL_NAMES = frozenset(
    {
        "repository.read_file",
        "repository.search",
        "repository.list_files",
        "git.diff",
        "git.status",
        "git.log",
    }
)


class ReviewerError(RuntimeError):
    """The review could not be produced (hard failure — task FAILED)."""


class ReviewerToolCall(BaseModel):
    tool: str = Field(min_length=1)
    input: dict = Field(default_factory=dict)


class ToolCallProposal(BaseModel):
    tool_call: ReviewerToolCall | None = None
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


class ReviewerAgent(Agent):
    name: ClassVar[str] = "reviewer"
    description: ClassVar[str] = (
        "Independently critiques an implementation commit from its diff "
        "and test result."
    )
    # Read-only by contract: no repo.write, no git.write. The Reviewer
    # judges the change; it never makes one.
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
            settings.max_reviewer_tool_calls
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
        commit_sha: str,
        test_result,
        ctx: ExecutionContext,
    ) -> ReviewResult:
        """Review the commit's diff (read-only) and return a persisted
        ``ReviewResult``.

        ``test_result`` may be a Pydantic ``TestResult`` or a ``TestRun``
        row (duck-typed). ``commit_sha`` is the ONLY window into the
        implementation — the developer's summary is not passed in, and no
        summary-shaped data is constructed anywhere in this agent.
        """
        if ctx.db is None:
            raise ReviewerError("ExecutionContext.db is required for review")
        db = ctx.db
        worktree = self._ensure_worktree(db, task)

        messages = build_reviewer_messages(task, commit_sha, test_result)
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
                return await self._verdict(
                    db, task, messages, observations, commit_sha, forced=False
                )

            obs = await self._execute_tool(proposal.tool_call, worktree.id, db, task.id)
            observations.append(obs)
            messages.append(observation_message(obs))

        # Budget exhausted: force the verdict (like Research's forced
        # synthesis) — a decision is required for routing.
        logger.warning(
            "Reviewer tool budget (%d) exhausted for task %s — forcing verdict",
            self.max_tool_calls, task.id,
        )
        db.add(
            AuditLog(
                task_id=task.id,
                actor=self.name,
                action="reviewer.budget_exhausted",
                entity_type="task",
                entity_id=str(task.id),
                details={"max_tool_calls": self.max_tool_calls},
            )
        )
        db.commit()
        return await self._verdict(
            db, task, messages, observations, commit_sha, forced=True
        )

    # -- internals -----------------------------------------------------------

    def _ensure_worktree(self, db, task):
        from app.git.worktree_manager import WorktreeManager

        return WorktreeManager(db).get_or_create_for_task(task)

    async def _execute_tool(
        self,
        call: ReviewerToolCall,
        worktree_id: uuid.UUID,
        db,
        task_id: uuid.UUID,
    ) -> Observation:
        """Run ONE proposal through the pipeline under Reviewer's read-only
        capabilities. A denial is UNEXPECTED (Developer-style): the Reviewer
        has every capability it legitimately needs, so a write proposal is
        the model reaching outside its job — audited loudly, loop survives.
        """
        tool_input = dict(call.input)
        tool_input.pop("worktree_id", None)
        tool_input["worktree_id"] = str(worktree_id)

        ctx = ExecutionContext(task_id=task_id, agent_type=self.name, db=db)
        try:
            result = await ToolPipeline(db).invoke(
                call.tool, tool_input, set(self.capabilities), ctx
            )
        except Exception as exc:  # noqa: BLE001 — contract errors surface as FAILED obs
            logger.warning("reviewer tool %s raised: %s", call.tool, exc)
            self._audit(
                db, task_id, "reviewer.unexpected_denial",
                {"tool": call.tool, "surfaced_as": "error", "error": str(exc)},
            )
            return Observation(
                tool=call.tool, status="FAILED", input=tool_input, error=str(exc)
            )

        if result.status == "DENIED":
            logger.warning(
                "Reviewer tool %s denied: %s", call.tool, result.denial_reason
            )
            self._audit(
                db, task_id, "reviewer.unexpected_denial",
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

    async def _verdict(
        self,
        db,
        task: Task,
        messages: list[Message],
        observations: list[Observation],
        commit_sha: str,
        *,
        forced: bool,
    ) -> ReviewResult:
        """Produce the final verdict: retry ONCE on malformed or
        schema-invalid output; a second failure raises (never a fabricated
        decision — the decision drives routing)."""
        synth = build_verdict_messages(messages, observations, forced)
        last_error: str | None = None
        for _ in range(2):  # one correction retry
            try:
                review: ReviewResult = await structured_output_with_retries(
                    self.provider,
                    synth,
                    ReviewResult,
                    timeout_retries=self.timeout_retries,
                    backoff_base_seconds=self.backoff_base,
                )
            except LLMMalformedOutputError as exc:
                last_error = str(exc)
                synth = build_verdict_correction(synth, last_error)
                continue
            self._persist(db, task, review, commit_sha)
            logger.info(
                "Review for task %s: %s (%d issues)",
                task.id, review.decision, len(review.issues),
            )
            return review

        raise ReviewerError(
            f"Reviewer could not produce a valid ReviewResult for task "
            f"{task.id} after the retry-once path: {last_error}"
        )

    def _persist(
        self, db, task: Task, review: ReviewResult, commit_sha: str
    ) -> None:
        db.add(
            ReviewResultRow(
                task_id=task.id,
                commit_sha=commit_sha,
                decision=review.decision,
                severity=review.severity,
                issues=[issue.model_dump() for issue in review.issues],
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


def build_reviewer() -> ReviewerAgent:
    """Construct the reviewer agent from settings/env (worker entrypoint)."""
    from app.agents.planner.agent import build_provider

    return ReviewerAgent(build_provider(role="reviewer"))
