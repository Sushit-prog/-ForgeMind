"""The Security Agent (architecture doc sections 12 and E).

A read-only tool-use loop (the Phase 6/8 skeleton: propose -> pipeline ->
observation -> bounded -> forced synthesis) that runs the Section-12
checklist against the developer's commit from the diff ONLY.

Independence is STRUCTURAL, one step past the Reviewer: ``run`` takes
``commit_sha`` and nothing else implementation-derived — no summary, no
ReviewResult, no test result. Security runs after Reviewer approval but is
blind to what the Reviewer said, so a security finding can never be the
reviewer's opinion recycled.

Denials are Developer-style (unexpected): the Security agent has every
capability it legitimately needs, so a write proposal is the model
reaching outside its job — audited loudly as ``security.unexpected_denial``.
"""

from __future__ import annotations

import logging
import uuid
from typing import ClassVar

from pydantic import BaseModel, Field, model_validator

from app.agents.base import Agent, structured_output_with_retries
from app.agents.security.prompt import (
    build_security_messages,
    build_verdict_correction,
    build_verdict_messages,
    observation_message,
)
from app.agents.security.schema import SecurityResult
from app.config import get_settings
from app.execution import ToolPipeline
from app.llm.errors import LLMMalformedOutputError
from app.llm.provider import LLMProvider, Message
from app.models import AuditLog
from app.models import SecurityResult as SecurityResultRow
from app.models import Task
from app.tools.base import ExecutionContext

logger = logging.getLogger(__name__)

# Tools the agent may propose (the pipeline still enforces the read-only
# capability set — this list is a prompt guard, not the boundary).
SECURITY_TOOL_NAMES = frozenset(
    {
        "repository.read_file",
        "repository.search",
        "repository.list_files",
        "git.diff",
        "git.status",
        "git.log",
    }
)


class SecurityError(RuntimeError):
    """The security verdict could not be produced (task FAILED)."""


class SecurityToolCall(BaseModel):
    tool: str = Field(min_length=1)
    input: dict = Field(default_factory=dict)


class ToolCallProposal(BaseModel):
    tool_call: SecurityToolCall | None = None
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


class SecurityAgent(Agent):
    name: ClassVar[str] = "security"
    description: ClassVar[str] = (
        "Runs a security checklist against an implementation commit's diff."
    )
    # Read-only, no exceptions (Section 12): no repo.write, no git.write.
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
            settings.max_security_tool_calls
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
        ctx: ExecutionContext,
    ) -> SecurityResult:
        """Scan the commit's diff (read-only) and return a persisted
        ``SecurityResult``.

        ``commit_sha`` is the ONLY window into the implementation. The
        Reviewer's verdict is deliberately not passed in, so Security can
        never agree with it by construction.
        """
        if ctx.db is None:
            raise SecurityError("ExecutionContext.db is required for security review")
        db = ctx.db
        worktree = self._ensure_worktree(db, task)

        messages = build_security_messages(task, commit_sha)
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

        # Budget exhausted: force the verdict.
        logger.warning(
            "Security tool budget (%d) exhausted for task %s — forcing verdict",
            self.max_tool_calls, task.id,
        )
        db.add(
            AuditLog(
                task_id=task.id,
                actor=self.name,
                action="security.budget_exhausted",
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
        call: SecurityToolCall,
        worktree_id: uuid.UUID,
        db,
        task_id: uuid.UUID,
    ) -> Observation:
        """Run ONE proposal through the pipeline under Security's read-only
        capabilities. A denial is UNEXPECTED (Developer-style): audited
        loudly as ``security.unexpected_denial``, loop survives.
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
            logger.warning("security tool %s raised: %s", call.tool, exc)
            self._audit(
                db, task_id, "security.unexpected_denial",
                {"tool": call.tool, "surfaced_as": "error", "error": str(exc)},
            )
            return Observation(
                tool=call.tool, status="FAILED", input=tool_input, error=str(exc)
            )

        if result.status == "DENIED":
            logger.warning(
                "Security tool %s denied: %s", call.tool, result.denial_reason
            )
            self._audit(
                db, task_id, "security.unexpected_denial",
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
    ) -> SecurityResult:
        """Produce the final verdict: retry ONCE on malformed or
        schema-invalid output; a second failure raises."""
        synth = build_verdict_messages(messages, observations, forced)
        last_error: str | None = None
        for _ in range(2):  # one correction retry
            try:
                result: SecurityResult = await structured_output_with_retries(
                    self.provider,
                    synth,
                    SecurityResult,
                    timeout_retries=self.timeout_retries,
                    backoff_base_seconds=self.backoff_base,
                )
            except LLMMalformedOutputError as exc:
                last_error = str(exc)
                synth = build_verdict_correction(synth, last_error)
                continue
            self._persist(db, task, result, commit_sha)
            logger.info(
                "Security verdict for task %s: %s (%d findings)",
                task.id, result.decision, len(result.findings),
            )
            return result

        raise SecurityError(
            f"Security could not produce a valid SecurityResult for task "
            f"{task.id} after the retry-once path: {last_error}"
        )

    def _persist(
        self, db, task: Task, result: SecurityResult, commit_sha: str
    ) -> None:
        db.add(
            SecurityResultRow(
                task_id=task.id,
                commit_sha=commit_sha,
                decision=result.decision,
                findings=[finding.model_dump() for finding in result.findings],
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


def build_security() -> SecurityAgent:
    """Construct the security agent from settings/env (worker entrypoint)."""
    from app.agents.planner.agent import build_provider

    return SecurityAgent(build_provider(role="security"))
