"""The Developer Agent (architecture doc sections 7 and E).

The second write-capable agent: a bounded tool-use loop where the LLM
proposes tool calls, the Phase 3 pipeline executes them under Developer's
fixed capability set (repo.read, repo.write, git.read, git.write — no
shell.*, no github.*), results feed back as DATA, and the loop ends with a
schema-validated, content-cross-checked ``ImplementationSummary`` backed by
a REAL commit on the task's isolated worktree.

Two asymmetries vs. the Research Agent (Phase 6), both deliberate:

1. A DENIED/unknown tool result here is UNEXPECTED — the developer already
   holds every write capability it legitimately needs, so a denial (or a
   proposal for an unregistered shell.*/github.* tool) signals the LLM is
   reaching outside its actual job (verify its own work, ship a PR). It is
   audited as ``developer.unexpected_denial``, distinctly from Research's
   expected-denial case.

2. No commit = hard failure, not forced synthesis. Research without an
   artifact is degraded-but-usable; an implementation without a commit is
   nothing. The loop raises ``DeveloperError`` and persists an explicit
   INCOMPLETE marker row (Phase 5's INVALID-plan-row pattern) so the failure
   is never silent.

One-commit contract: the loop enforces EXACTLY ONE ``git.commit`` per run —
structurally, not by convention. After the first successful commit, further
``filesystem.write_file`` and ``git.commit`` proposals are denied at the
agent level (audited ``developer.post_commit_proposal``), so the single
commit provably represents the full change and no uncommitted residue can
accumulate. Chosen over allow-and-squash: squashing requires fragile history
surgery, while structural enforcement is deterministic and gives a Reviewer
one unambiguous commit to diff.
"""

from __future__ import annotations

import logging
import uuid
from typing import ClassVar

from pydantic import BaseModel, Field, model_validator

from app.agents.base import Agent, structured_output_with_retries
from app.agents.developer.prompt import (
    build_developer_messages,
    build_summary_correction,
    build_synthesis_messages,
    observation_message,
)
from app.agents.developer.schema import (
    ImplementationSummary,
    ImplementationSummaryDraft,
    files_changed_mismatch,
    research_deviations,
    written_paths,
)
from app.config import get_settings
from app.execution import ToolPipeline
from app.llm.errors import LLMMalformedOutputError
from app.llm.provider import LLMProvider, Message
from app.models import AuditLog
from app.models import ImplementationSummary as ImplementationSummaryRow
from app.models import Task
from app.tools.base import ExecutionContext
from app.tools.registry import ToolNotFoundError

logger = logging.getLogger(__name__)

# Tools the agent may propose (the pipeline still enforces capabilities).
DEVELOPER_TOOL_NAMES = frozenset(
    {
        "repository.read_file",
        "repository.search",
        "repository.list_files",
        "filesystem.write_file",
        "git.status",
        "git.diff",
        "git.log",
        "git.commit",
        "git.create_branch",
    }
)


class DeveloperError(RuntimeError):
    """Implementation could not produce a commit (hard failure — no summary)."""


class DeveloperToolCall(BaseModel):
    tool: str = Field(min_length=1)
    input: dict = Field(default_factory=dict)


class ToolCallProposal(BaseModel):
    tool_call: DeveloperToolCall | None = None
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


class DeveloperAgent(Agent):
    name: ClassVar[str] = "developer"
    description: ClassVar[str] = "Implements a plan step and commits it on the isolated worktree."
    # Write-capable, but structurally bounded: no shell.*, no github.* —
    # build/test verification is a later phase's job, not Developer's.
    capabilities: ClassVar[list[str]] = ["repo.read", "repo.write", "git.read", "git.write"]

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
            settings.max_developer_tool_calls if max_tool_calls is None else max_tool_calls
        )
        self.timeout_retries = (
            settings.llm_max_retries if timeout_retries is None else timeout_retries
        )
        self.backoff_base = backoff_base_seconds

    # -- the public contract -------------------------------------------------

    async def run(
        self, task: Task, plan_step, research, ctx: ExecutionContext
    ) -> ImplementationSummary:
        """Tool-use loop: read -> write -> commit ONCE -> grounded summary.

        Bounded by ``max_tool_calls``. Must produce at least one commit
        before emitting a summary — zero commits is a hard failure.
        """
        if ctx.db is None:
            raise DeveloperError("ExecutionContext.db is required for implementation")
        db = ctx.db

        worktree = self._ensure_worktree(db, task)
        messages = build_developer_messages(task, plan_step, research)
        observations: list[Observation] = []
        commit_sha: str | None = None
        committed = False

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
                if not committed:
                    self._fail_no_commit(
                        db, task, plan_step, worktree.id, "no_commit_before_final"
                    )
                return await self._synthesize(
                    db, task, plan_step, worktree.id, research,
                    messages, observations, commit_sha, forced=False,
                )

            obs = await self._execute_tool(
                proposal.tool_call, worktree.id, db, task.id, committed=committed
            )
            observations.append(obs)
            if obs.status == "EXECUTED" and obs.tool == "git.commit":
                committed = True
                commit_sha = (obs.output or {}).get("sha")
            messages.append(observation_message(obs))

        # Budget exhausted without a final answer. With a commit, force the
        # summary (like Research); without one, hard-fail — an implementation
        # with no commit is not a degraded artifact, it is nothing.
        logger.warning(
            "Developer tool budget (%d) exhausted for task %s (committed=%s) — %s",
            self.max_tool_calls, task.id, committed,
            "forcing synthesis" if committed else "hard failure (no commit)",
        )
        db.add(
            AuditLog(
                task_id=task.id,
                actor=self.name,
                action="developer.budget_exhausted",
                entity_type="task",
                entity_id=str(task.id),
                details={"max_tool_calls": self.max_tool_calls, "committed": committed},
            )
        )
        db.commit()
        if not committed:
            self._fail_no_commit(db, task, plan_step, worktree.id, "no_commit_budget_exhausted")
        return await self._synthesize(
            db, task, plan_step, worktree.id, research,
            messages, observations, commit_sha, forced=True,
        )

    # -- internals -----------------------------------------------------------

    def _ensure_worktree(self, db, task):
        from app.git.worktree_manager import WorktreeManager

        return WorktreeManager(db).get_or_create_for_task(task)

    async def _execute_tool(
        self,
        call: DeveloperToolCall,
        worktree_id: uuid.UUID,
        db,
        task_id: uuid.UUID,
        *,
        committed: bool,
    ) -> Observation:
        """Run ONE proposal through the pipeline under Developer's capabilities.

        The worktree_id is injected server-side, as in Research. The
        one-commit contract is enforced HERE, before the pipeline: once the
        single commit exists, no further writes or commits may happen, so the
        commit provably represents the full change.
        """
        tool_input = dict(call.input)
        tool_input.pop("worktree_id", None)
        tool_input["worktree_id"] = str(worktree_id)

        if committed and call.tool in ("filesystem.write_file", "git.commit"):
            reason = (
                "the run already has its single commit — further writes and commits "
                "are denied; the commit must represent the full change. Respond "
                '{"final": true}.'
            )
            self._audit(
                db, task_id, "developer.post_commit_proposal",
                {"tool": call.tool, "denial_reason": reason},
            )
            return Observation(
                tool=call.tool, status="DENIED", input=tool_input, denial_reason=reason
            )

        ctx = ExecutionContext(task_id=task_id, agent_type=self.name, db=db)
        try:
            result = await ToolPipeline(db).invoke(
                call.tool, tool_input, set(self.capabilities), ctx
            )
        except ToolNotFoundError as exc:
            # Unknown tool: a contract error, but for the developer it is a
            # signal the LLM is reaching outside its job (shell.*, github.*
            # don't exist yet). Audited as unexpected — Research treats an
            # unknown tool as a benign probe; here it is not.
            logger.warning("Developer proposed unknown tool %s: %s", call.tool, exc)
            self._audit(
                db, task_id, "developer.unexpected_denial",
                {"tool": call.tool, "surfaced_as": "unknown_tool", "error": str(exc)},
            )
            return Observation(
                tool=call.tool, status="FAILED", input=tool_input, error=str(exc)
            )
        except Exception as exc:  # noqa: BLE001 — contract errors surface as FAILED obs
            logger.warning("developer tool %s raised: %s", call.tool, exc)
            return Observation(
                tool=call.tool, status="FAILED", input=tool_input, error=str(exc)
            )

        if result.status == "DENIED":
            # Pipeline denial (capability/policy gate). The developer holds
            # every capability it legitimately needs, so a denial is
            # UNEXPECTED — audited distinctly from Research's expected case.
            logger.warning(
                "Developer tool %s unexpectedly denied: %s", call.tool, result.denial_reason
            )
            self._audit(
                db, task_id, "developer.unexpected_denial",
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

    def _fail_no_commit(self, db, task: Task, plan_step, worktree_id, reason: str):
        """Persist an explicit INCOMPLETE marker (Phase 5's INVALID-row pattern)
        and raise — an implementation with no commit is a hard failure."""
        db.add(
            ImplementationSummaryRow(
                task_id=task.id,
                step_id=getattr(plan_step, "id", None),
                worktree_id=worktree_id,
                commit_sha=None,
                files_changed=[],
                summary="INCOMPLETE — developer finished without producing a commit.",
                tests_added=[],
                deviations_from_research=None,
                status="INCOMPLETE",
            )
        )
        db.add(
            AuditLog(
                task_id=task.id,
                actor=self.name,
                action="developer.no_commit",
                entity_type="implementation_summary",
                entity_id=str(task.id),
                details={"reason": reason},
            )
        )
        db.commit()
        raise DeveloperError(
            f"Developer produced no commit for task {task.id} ({reason}) — "
            "an implementation with zero commits is a contract violation, not a "
            "degraded outcome"
        )

    async def _synthesize(
        self,
        db,
        task: Task,
        plan_step,
        worktree_id: uuid.UUID,
        research,
        messages: list[Message],
        observations: list[Observation],
        commit_sha: str,
        *,
        forced: bool,
    ) -> ImplementationSummary:
        """Produce the final summary: retry-once on malformed output, on
        files_changed not matching what was actually written, or on an
        unexplained divergence from research; then accept-with-warning
        (audited) rather than failing the whole task.

        Policy decision: writes get the SAME accept-with-warning-after-one-
        retry treatment as Research's reads, not stricter. The committed diff
        is independently verifiable ground truth (a later Reviewer diffs the
        commit), so a summary-field mismatch is recoverable downstream —
        failing the whole task over it would be disproportionate.
        """
        written = written_paths(observations)
        researched = research.relevant_files or []
        deviations = research_deviations(written, researched)
        synth = build_synthesis_messages(messages, written, researched, forced)

        last: ImplementationSummaryDraft | None = None
        for _ in range(2):  # one correction retry
            try:
                draft: ImplementationSummaryDraft = await structured_output_with_retries(
                    self.provider,
                    synth,
                    ImplementationSummaryDraft,
                    timeout_retries=self.timeout_retries,
                    backoff_base_seconds=self.backoff_base,
                )
            except LLMMalformedOutputError as exc:
                synth = build_summary_correction(synth, str(exc))
                continue

            last = draft
            mismatch = files_changed_mismatch(draft, written)
            unexplained = bool(deviations) and not (draft.deviations_from_research or "").strip()
            if not mismatch and not unexplained:
                summary = ImplementationSummary(commit_sha=commit_sha, **draft.model_dump())
                self._persist(db, task, plan_step, worktree_id, summary)
                logger.info(
                    "Implementation summary persisted for task %s (commit %s, %d files)",
                    task.id, commit_sha, len(summary.files_changed),
                )
                return summary

            problems: list[str] = []
            if mismatch:
                problems.append(
                    "files_changed does not match the files actually written via "
                    f"filesystem.write_file — expected {sorted(written)}, got {draft.files_changed}"
                )
            if unexplained:
                problems.append(
                    "you wrote files the research did not flag "
                    f"({deviations}) — deviations_from_research must explain why"
                )
            synth = build_summary_correction(synth, "; ".join(problems))

        # Both attempts failed the grounding check. POLICY: accept-with-
        # warning — same reasoning as Phase 6, applied to writes (the
        # committed diff is verifiable ground truth downstream). Audited
        # loudly, never silent.
        assert last is not None
        mismatch = files_changed_mismatch(last, written)
        if mismatch:
            logger.error(
                "Implementation summary accepted WITH unverified files_changed for task %s: %s",
                task.id, mismatch,
            )
            db.add(
                AuditLog(
                    task_id=task.id,
                    actor=self.name,
                    action="implementation.files_unverified",
                    entity_type="implementation_summary",
                    entity_id=str(task.id),
                    details={"mismatch": mismatch},
                )
            )
        unexplained = bool(deviations) and not (last.deviations_from_research or "").strip()
        if unexplained:
            logger.error(
                "Implementation summary accepted WITHOUT explaining deviations for task %s: %s",
                task.id, deviations,
            )
            db.add(
                AuditLog(
                    task_id=task.id,
                    actor=self.name,
                    action="implementation.deviations_unexplained",
                    entity_type="implementation_summary",
                    entity_id=str(task.id),
                    details={"deviations": deviations},
                )
            )
        if mismatch or unexplained:
            db.commit()
        summary = ImplementationSummary(commit_sha=commit_sha, **last.model_dump())
        self._persist(db, task, plan_step, worktree_id, summary)
        return summary

    def _persist(
        self, db, task: Task, plan_step, worktree_id: uuid.UUID, summary: ImplementationSummary
    ) -> None:
        db.add(
            ImplementationSummaryRow(
                task_id=task.id,
                step_id=getattr(plan_step, "id", None),
                worktree_id=worktree_id,
                commit_sha=summary.commit_sha,
                files_changed=summary.files_changed,
                summary=summary.summary,
                tests_added=summary.tests_added,
                deviations_from_research=summary.deviations_from_research,
                status="COMPLETE",
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


def build_developer() -> DeveloperAgent:
    """Construct the developer agent from settings/env (worker entrypoint)."""
    from app.agents.planner.agent import build_provider

    return DeveloperAgent(build_provider())
