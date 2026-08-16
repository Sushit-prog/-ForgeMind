"""The Research Agent (architecture doc sections 7 and E).

The first MULTI-TURN tool-use agent: a bounded loop where the LLM proposes
tool calls, the Phase 3 pipeline executes them under Research's fixed
read-only capability set, results feed back as DATA, and the loop ends
with a schema-validated, content-cross-checked ``ResearchArtifact``.

The read-only boundary is structural, not conventional: every proposal
goes through ``ToolPipeline.invoke`` with ``capabilities=ResearchAgent.
capabilities`` (repo.read, git.read, github.read). A proposal like
``filesystem.write_file`` or ``git.commit`` is DENIED and audited by the
pipeline — the agent never gets to "just try it".
"""

from __future__ import annotations

import logging
import uuid
from typing import ClassVar

from pydantic import BaseModel, Field, model_validator

from app.agents.base import Agent, structured_output_with_retries
from app.agents.researcher.prompt import (
    build_artifact_correction,
    build_research_messages,
    build_synthesis_messages,
    observation_message,
)
from app.agents.researcher.schema import (
    ResearchArtifact,
    observed_paths,
    unobserved_files,
)
from app.config import get_settings
from app.execution import ToolPipeline
from app.llm.errors import LLMMalformedOutputError
from app.llm.provider import LLMProvider, Message
from app.models import AuditLog
from app.models import ResearchArtifact as ResearchArtifactRow
from app.models import Task
from app.tools.base import ExecutionContext

logger = logging.getLogger(__name__)

# Tools the agent may propose (the pipeline still enforces capabilities).
RESEARCH_TOOL_NAMES = frozenset(
    {
        "repository.read_file",
        "repository.search",
        "repository.list_files",
        "git.status",
        "git.diff",
        "git.log",
    }
)


class ResearchError(RuntimeError):
    """Research could not produce an artifact (hard failure)."""


class ResearchToolCall(BaseModel):
    tool: str = Field(min_length=1)
    input: dict = Field(default_factory=dict)


class ToolCallProposal(BaseModel):
    tool_call: ResearchToolCall | None = None
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


class ResearchAgent(Agent):
    name: ClassVar[str] = "researcher"
    description: ClassVar[str] = "Read-only investigation of the repository."
    # Read-only by contract (Section 7): no repo.write, no git.write.
    capabilities: ClassVar[list[str]] = ["repo.read", "git.read", "github.read"]

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
            settings.max_research_tool_calls if max_tool_calls is None else max_tool_calls
        )
        self.timeout_retries = (
            settings.llm_max_retries if timeout_retries is None else timeout_retries
        )
        self.backoff_base = backoff_base_seconds

    # -- the public contract -------------------------------------------------

    async def run(
        self, task: Task, plan_step, ctx: ExecutionContext
    ) -> ResearchArtifact:
        if ctx.db is None:
            raise ResearchError("ExecutionContext.db is required for research")
        db = ctx.db

        worktree = self._ensure_worktree(db, task)
        messages = build_research_messages(task, plan_step, repo_metadata=None)
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
                # Bad proposal JSON: tell the LLM the format, keep looping
                # (bounded by the budget). Never crash the loop.
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
                return await self._synthesize(db, task, messages, observations, forced=False)

            obs = await self._execute_tool(proposal.tool_call, worktree.id, db, task.id)
            observations.append(obs)
            messages.append(observation_message(obs))

        # Budget exhausted without a final answer: force synthesis, audit it.
        logger.warning(
            "Research tool budget (%d) exhausted for task %s — forcing synthesis",
            self.max_tool_calls,
            task.id,
        )
        db.add(
            AuditLog(
                task_id=task.id,
                actor=self.name,
                action="research.budget_exhausted",
                entity_type="task",
                entity_id=str(task.id),
                details={"max_tool_calls": self.max_tool_calls},
            )
        )
        db.commit()
        return await self._synthesize(db, task, messages, observations, forced=True)

    # -- internals -----------------------------------------------------------

    def _ensure_worktree(self, db, task):
        from app.git.worktree_manager import WorktreeManager

        return WorktreeManager(db).get_or_create_for_task(task)

    async def _execute_tool(
        self,
        call: ResearchToolCall,
        worktree_id: uuid.UUID,
        db,
        task_id: uuid.UUID,
    ) -> Observation:
        """Run ONE proposal through the pipeline under Research's capabilities.

        The worktree_id is injected server-side: the LLM names the tool and
        its path/query input, never a workspace handle. Any tool outside the
        research set (or any write tool) is DENIED by the pipeline and the
        denial is audited — the observation tells the LLM exactly that.
        """
        # Strip any worktree_id the LLM might have invented — the task's
        # worktree is the only one it may ever touch.
        tool_input = dict(call.input)
        tool_input.pop("worktree_id", None)
        tool_input["worktree_id"] = str(worktree_id)

        ctx = ExecutionContext(task_id=task_id, agent_type=self.name, db=db)
        try:
            result = await ToolPipeline(db).invoke(
                call.tool, tool_input, set(self.capabilities), ctx
            )
        except Exception as exc:  # noqa: BLE001 — contract errors surface as FAILED obs
            logger.warning("research tool %s raised: %s", call.tool, exc)
            return Observation(
                tool=call.tool, status="FAILED", input=tool_input, error=str(exc)
            )

        return Observation(
            tool=call.tool,
            status=result.status,
            input=tool_input,
            output=result.output,
            error=result.error,
            denial_reason=result.denial_reason,
        )

    async def _synthesize(
        self,
        db,
        task: Task,
        messages: list[Message],
        observations: list[Observation],
        *,
        forced: bool,
    ) -> ResearchArtifact:
        """Produce the final artifact: retry-once on malformed OR on files
        never observed; then accept-with-warning (audited) rather than
        failing the whole task."""
        observed = observed_paths(observations)
        synth = build_synthesis_messages(messages, observed, forced)

        last_artifact: ResearchArtifact | None = None
        for _ in range(2):  # one correction retry
            try:
                artifact: ResearchArtifact = await structured_output_with_retries(
                    self.provider,
                    synth,
                    ResearchArtifact,
                    timeout_retries=self.timeout_retries,
                    backoff_base_seconds=self.backoff_base,
                )
            except LLMMalformedOutputError as exc:
                synth = build_artifact_correction(synth, str(exc))
                continue

            last_artifact = artifact
            missing = unobserved_files(artifact, observed)
            if not missing:
                self._persist(db, task, artifact)
                logger.info(
                    "Research artifact persisted for task %s (confidence %s, %d files)",
                    task.id, artifact.confidence, len(artifact.relevant_files),
                )
                return artifact

            # Grounding violation: it referenced files it never saw.
            synth = build_artifact_correction(
                synth,
                "referenced files never observed: "
                + ", ".join(missing)
                + ". Only reference files from your observations.",
            )

        # Both attempts failed the grounding check. POLICY: accept-with-
        # warning — a wrong file reference in a hypothesis artifact is
        # recoverable downstream (Developer verifies), while failing the
        # whole task over it is disproportionate. The discrepancy is
        # audited and logged loudly.
        assert last_artifact is not None
        missing = unobserved_files(last_artifact, observed)
        logger.error(
            "Research artifact accepted WITH unobserved file references for task %s: %s",
            task.id,
            missing,
        )
        db.add(
            AuditLog(
                task_id=task.id,
                actor=self.name,
                action="artifact.files_unverified",
                entity_type="research_artifact",
                entity_id=str(task.id),
                details={"unobserved": missing},
            )
        )
        self._persist(db, task, last_artifact)
        return last_artifact

    def _persist(self, db, task: Task, artifact: ResearchArtifact) -> None:
        db.add(
            ResearchArtifactRow(
                task_id=task.id,
                root_cause_hypothesis=artifact.root_cause_hypothesis,
                relevant_files=artifact.relevant_files,
                relevant_tests=artifact.relevant_tests,
                evidence=artifact.evidence,
                confidence=artifact.confidence,
            )
        )
        db.commit()


def build_researcher() -> ResearchAgent:
    """Construct the research agent from settings/env (worker entrypoint)."""
    from app.agents.planner.agent import build_provider

    return ResearchAgent(build_provider())
