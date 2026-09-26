"""The Planning Agent (architecture doc sections 6.1 and E).

``PlanningAgent.run`` is the first real LLM-driven component: task
objective in -> schema-validated Plan out -> persisted -> returned. It
drives the state machine's PLANNING -> RESEARCHING transition (see
``task_lifecycle.advance_task_with_agents``).

Retry policy (the seam the milestone calls out):

- Transient failures (``LLMTimeoutError``, 5xx/429) are retried with
  bounded exponential backoff INSIDE each attempt — a slow provider must
  not burn the malformed-output retry budget.
- The malformed/invalid retry is exactly ONCE: attempt 1, then one
  correction attempt with the rejection reason. A second failure raises
  ``PlanValidationError`` (never a silent fallback) and the caller marks
  the task FAILED — never ESCALATED (that is reserved for replan-budget
  exhaustion).
"""

from __future__ import annotations

import json
import logging
import os
from typing import ClassVar

from app.agents.base import Agent, structured_output_with_retries
from app.agents.planner.prompt import (
    build_correction_messages,
    build_planning_messages,
)
from app.agents.planner.schema import Plan, PlanValidationError, validate_plan_dag
from app.config import get_settings
from app.execution.tool_pipeline import redact_sensitive
from app.llm.errors import LLMMalformedOutputError, LLMTimeoutError
from app.llm.provider import LLMProvider, Message
from app.models import Plan as PlanRow
from app.models import PlanStep as PlanStepRow
from app.models import Task
from app.tools.base import ExecutionContext

logger = logging.getLogger(__name__)

# Raw LLM output preserved for debugging (Section 47) — truncated + redacted.
MAX_RAW_STORED = 8_000


class PlannerConfigError(RuntimeError):
    """No LLM provider is configured (no key, no mock flag)."""


def _redact_raw(raw: str) -> str:
    """Redact secrets from raw LLM output before persisting/logging."""
    try:
        data = json.loads(raw)
        return json.dumps(redact_sensitive(data), ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        # Unparseable raw output: truncate — truncation is itself a leak
        # mitigation for malformed/injected text.
        return raw[:MAX_RAW_STORED]


class PlanningAgent(Agent):
    name: ClassVar[str] = "planner"
    description: ClassVar[str] = (
        "Produces a schema-validated dependency-graph plan for a task."
    )
    # Section 6.1: the planner does NOT write code and cannot invoke any
    # tool. An empty capability set makes that structural, not conventional.
    capabilities: ClassVar[list[str]] = []

    def __init__(
        self,
        provider: LLMProvider,
        *,
        timeout_retries: int | None = None,
        backoff_base_seconds: float = 0.5,
    ) -> None:
        self.provider = provider
        settings = get_settings()
        self.timeout_retries = (
            settings.llm_max_retries if timeout_retries is None else timeout_retries
        )
        self.backoff_base = backoff_base_seconds

    # -- the public contract -------------------------------------------------

    async def run(self, task: Task, ctx: ExecutionContext) -> Plan:
        repo_metadata = None
        if ctx.db is not None and task.repository_id is not None:
            try:
                from app.models import Repository
                from app.repository.discovery import RepositoryDiscovery

                repository = ctx.db.get(Repository, task.repository_id)
                if repository is not None:
                    repo_metadata = RepositoryDiscovery().get_cached_metadata(
                        repository
                    )
            except Exception:  # noqa: BLE001 — metadata is a nice-to-have
                logger.debug("planner: repo metadata unavailable", exc_info=True)

        messages = build_planning_messages(task, repo_metadata)
        last_raw = ""

        for attempt in range(2):  # initial + exactly ONE correction retry
            try:
                plan = await self._call_with_timeout_retries(messages)
            except LLMMalformedOutputError as exc:
                last_raw = exc.raw_output
                messages = build_correction_messages(messages, str(exc))
                continue
            except LLMTimeoutError:
                # Transient retries exhausted inside _call_with_timeout_retries;
                # do NOT burn the correction attempt on a timeout.
                self._persist_failure(ctx, task, last_raw, "llm_timeout")
                raise

            try:
                validate_plan_dag(plan)
            except PlanValidationError as exc:
                last_raw = plan.model_dump_json()
                messages = build_correction_messages(messages, str(exc))
                continue

            self._persist_success(ctx, task, plan)
            return plan

        # Both attempts failed: preserve the raw output, raise loudly.
        self._persist_failure(ctx, task, last_raw, "plan_invalid")
        raise PlanValidationError(
            "LLM failed to produce a valid plan after the retry-once path",
            raw_output=last_raw,
        )

    # -- internals -----------------------------------------------------------

    async def _call_with_timeout_retries(self, messages: list[Message]) -> Plan:
        """One Plan call with bounded transient retries (shared helper)."""
        result = await structured_output_with_retries(
            self.provider,
            messages,
            Plan,
            timeout_retries=self.timeout_retries,
            backoff_base_seconds=self.backoff_base,
        )
        return result  # type: ignore[return-value]

    def _persist_success(self, ctx: ExecutionContext, task: Task, plan: Plan) -> None:
        if ctx.db is None:
            raise RuntimeError("ExecutionContext.db is required to persist a plan")
        db = ctx.db
        row = PlanRow(
            task_id=task.id,
            status="ACTIVE",
            raw_llm_output=_redact_raw(plan.model_dump_json()),
        )
        db.add(row)
        db.flush()

        llm_id_to_row: dict[str, object] = {}
        for i, step in enumerate(plan.steps):
            step_row = PlanStepRow(
                plan_id=row.id,
                step_type=step.step_type,
                sequence=i,
                params={
                    "llm_step_id": step.id,
                    "description": step.description,
                    "depends_on": step.depends_on,
                },
                status="PENDING",
            )
            db.add(step_row)
            db.flush()
            llm_id_to_row[step.id] = step_row

        # The table has a single depends_on FK: point it at the step's FIRST
        # dependency (the full DAG is preserved in params + raw_llm_output).
        for step in plan.steps:
            if step.depends_on:
                first = llm_id_to_row.get(step.depends_on[0])
                if first is not None:
                    llm_id_to_row[step.id].depends_on = first.id  # type: ignore[attr-defined]

        db.commit()
        logger.info(
            "Planner persisted plan %s for task %s (%d steps)",
            row.id,
            task.id,
            len(plan.steps),
        )

    def _persist_failure(
        self, ctx: ExecutionContext, task: Task, raw_output: str, reason: str
    ) -> None:
        """Persist an INVALID plan row so the raw output is never lost."""
        if ctx.db is None:
            return
        db = ctx.db
        row = PlanRow(
            task_id=task.id,
            status="INVALID",
            raw_llm_output=_redact_raw(raw_output),
        )
        db.add(row)
        db.commit()
        logger.error(
            "Planner failed for task %s (%s); raw output preserved", task.id, reason
        )


def build_provider(role: str = "planner"):
    """Construct the LLM provider from settings/env (shared by all agents).

    Order: a real backend when any API key is configured (OpenRouter,
    Groq, NVIDIA, Inception); else the stub provider when
    ``FORGEMIND_MOCK_LLM=1`` (tests / key-less dev); else a clear
    ``PlannerConfigError``.

    ``role`` selects the stub provider's per-schema script (research vs
    developer propose different first tool calls), so each agent builds its
    own provider with a script correct for ITS loop. On the real-provider
    path ``role`` selects the model (LLM_MODEL_<ROLE>) and optional
    fallback chain (LLM_MODEL_<ROLE>_FALLBACKS): when a chain is configured
    and has more than one entry, a FallbackLLMProvider hops to the next
    model only after the current one's bounded transient-retry budget is
    exhausted; otherwise the plain single-model provider, exactly as
    before.

    Each chain entry may carry a backend prefix selecting its endpoint:
    ``groq::slug`` -> Groq, ``nvidia::slug`` -> NVIDIA's integrate API,
    ``inception::slug`` -> Inception's Mercury API, bare slug -> OpenRouter
    (default). Chains may mix backends freely.
    """
    from app.llm.config import (
        get_fallback_models_for_role,
        get_model_for_role,
        split_backend_slug,
    )
    from app.llm.fallback import FallbackLLMProvider
    from app.llm.mock import StubLLMProvider, default_by_schema
    from app.llm.openai_compat import BACKENDS, OpenAICompatibleProvider

    settings = get_settings()
    if os.environ.get("FORGEMIND_MOCK_LLM") == "1":
        flaky = os.environ.get("FORGEMIND_MOCK_LLM_FLAKY") == "1"
        retry = None
        if role == "developer":
            from app.llm.mock import (
                COMMIT_PROPOSAL,
                FINAL_PROPOSAL,
                WRITE_RETRY_PROPOSAL,
            )

            # A developer run AFTER any replan (debugger/reviewer/security)
            # receives the fix instruction as DATA — the stub then proposes
            # the FIXED write (mock-only message-keyed queue; real providers
            # respond to the instruction naturally).
            retry = {
                "retry_by_schema": {
                    "ToolCallProposal": [
                        WRITE_RETRY_PROPOSAL,
                        COMMIT_PROPOSAL,
                        FINAL_PROPOSAL,
                    ]
                },
                "retry_marker": "FIX INSTRUCTION (DATA)",
            }
        elif role == "reviewer":
            # Reject-then-approve loop: the reviewer reads src/app.py; the
            # FIRST review sees VALUE = 2 -> REJECT (default queue); after
            # the developer applies the fix (VALUE = 3), the SECOND review's
            # read observation contains "VALUE = 3" -> APPROVE (retry queue).
            # Enabled by env so the plain happy path (approve first time)
            # stays the default for other e2e tests.
            if os.environ.get("FORGEMIND_MOCK_REVIEW_REJECT") == "1":
                from app.llm.mock import (
                    FIXED_VALUE_MARKER,
                    REVIEW_APPROVE,
                    REVIEW_REJECT,
                )

                retry = {
                    "retry_by_schema": {"ReviewResult": [REVIEW_APPROVE]},
                    "retry_marker": FIXED_VALUE_MARKER,
                }
                by_schema = default_by_schema(agent=role)
                by_schema["ReviewResult"] = [REVIEW_REJECT]
                return StubLLMProvider(by_schema=by_schema, **retry)
        elif role == "security":
            # Fail-then-pass loop, same marker mechanism as the reviewer.
            if os.environ.get("FORGEMIND_MOCK_SECURITY_FAIL") == "1":
                from app.llm.mock import (
                    FIXED_VALUE_MARKER,
                    SECURITY_FAIL,
                    SECURITY_PASS,
                )

                retry = {
                    "retry_by_schema": {"SecurityResult": [SECURITY_PASS]},
                    "retry_marker": FIXED_VALUE_MARKER,
                }
                by_schema = default_by_schema(agent=role)
                by_schema["SecurityResult"] = [SECURITY_FAIL]
                return StubLLMProvider(by_schema=by_schema, **retry)
        return StubLLMProvider(
            by_schema=default_by_schema(flaky_planner=flaky, agent=role), **retry or {}
        )
    backend_keys = {
        "openrouter": settings.openrouter_api_key,
        "groq": settings.groq_api_key,
        "nvidia": settings.nvidia_api_key,
        "inception": settings.inception_api_key,
    }

    def _key_for(backend: str) -> str:
        key = backend_keys.get(backend)
        if not key:
            raise PlannerConfigError(
                f"role '{role}': no API key configured for backend '{backend}' "
                f"(set {backend.upper()}_API_KEY)"
            )
        return key

    primary = get_model_for_role(role)
    fallbacks = get_fallback_models_for_role(role)
    # Unset models are dropped: an unset primary must not silently become
    # a fallback hop (a missing model fails loudly, as before).
    chain_entries = [model_name for model_name in (primary, *fallbacks) if model_name]

    resolved: list[tuple[str, str, str, str]] = []  # (entry, slug, backend, api_key)
    for model_name in chain_entries:
        backend, slug = split_backend_slug(model_name)
        if backend not in BACKENDS:
            raise PlannerConfigError(
                f"role '{role}': unknown backend '{backend}' in model "
                f"'{model_name}' (known: {sorted(BACKENDS)})"
            )
        resolved.append((model_name, slug, backend, _key_for(backend)))

    if not resolved and not any(backend_keys.values()):
        raise PlannerConfigError(
            "no LLM provider configured: set OPENROUTER_API_KEY / GROQ_API_KEY / "
            "NVIDIA_API_KEY / INCEPTION_API_KEY (and LLM_MODEL_<ROLE>), or "
            "FORGEMIND_MOCK_LLM=1 for key-less development"
        )

    def _provider(entry: tuple[str, str, str, str]) -> OpenAICompatibleProvider:
        _, slug, backend, api_key = entry
        return OpenAICompatibleProvider(
            api_key=api_key,
            base_url=BACKENDS[backend],
            model=slug,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    if len(resolved) < 2:
        if resolved:
            return _provider(resolved[0])
        # Keys configured but no models for this role: keep today's behavior
        # of returning a real-endpoint provider with no model (fails loudly
        # at call time naming the missing LLM_MODEL_<ROLE>). Uses whichever
        # OpenRouter-compatible endpoint the deployment prefers.
        return OpenAICompatibleProvider(
            api_key=settings.openrouter_api_key or "",
            base_url=settings.openrouter_base_url,
            model=None,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    return FallbackLLMProvider(
        [(entry[0], _provider(entry)) for entry in resolved],
        max_retries=settings.llm_max_retries,
        fallback_on_malformed=settings.llm_fallback_on_malformed,
        max_malformed_hops=settings.llm_max_malformed_hops,
    )


def build_planner() -> PlanningAgent:
    """Construct the planner from settings/env — used by the worker."""
    return PlanningAgent(build_provider())
