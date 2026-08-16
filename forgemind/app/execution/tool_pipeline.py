"""The tool pipeline (architecture doc section F).

Every tool invocation goes through exactly this sequence:

    1. validate  — raw_input against the tool's ``input_schema``
    2. capability check — every ``tool.capabilities`` present in the agent's set
    3. policy check — ``PolicyEngine.evaluate`` (deterministic, fail-closed)
    4. risk assessment — recorded on the audit row (HIGH gates arrive later)
    5. execute — with exception capture
    6. audit — exactly ONE ``tool_calls`` row, whatever the outcome

Guarantees, and how they're kept:

- **Exactly one row per invocation.** The row is inserted once (status
  ``ALLOWED``) after all gates pass, then updated in place to ``EXECUTED``
  or ``FAILED``. Denied calls get a single ``DENIED`` row. A crash between
  admit and execute leaves ``ALLOWED`` — informative, never a missing row.
- **Deny wins.** The policy engine returns the first DENY; an ALLOW vote
  never overrides it. Nothing in the pipeline can re-allow a denial.
- **No secrets in the audit row.** Inputs/outputs are redacted before
  storage (``redact_sensitive``) — the same posture as Phase 1's URL
  redaction, applied to structured data.
- **No partial execution.** A validation error raises before any gate runs
  and before ``execute`` is ever called; the tool never sees bad input.
- **Contract errors raise.** Unknown tool (``ToolNotFoundError``) and
  malformed input (``ToolInputValidationError``) propagate to the caller —
  they are not tool invocations and produce no row, and they are never a
  silent no-op.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from app.models import ToolCall, ToolCallStatus
from app.policies.engine import PolicyEngine
from app.tools.base import ExecutionContext, Tool
from app.tools.registry import ToolNotFoundError, ToolRegistry

logger = logging.getLogger(__name__)

# Keys whose values are redacted before persisting input/output. Extend as
# real tools with secrets arrive (tokens, keys, passwords).
SENSITIVE_KEYS = frozenset(
    {
        "password",
        "passwd",
        "token",
        "access_token",
        "refresh_token",
        "api_key",
        "apikey",
        "authorization",
        "secret",
        "client_secret",
    }
)
REDACTED = "***REDACTED***"


def redact_sensitive(value: Any, keys: frozenset[str] = SENSITIVE_KEYS) -> Any:
    """Recursively redact values under sensitive keys (dicts/lists/scalars).

    Lists are traversed in place (their contents may hold secrets under
    sensitive keys); scalar leaves pass through untouched.
    """
    if isinstance(value, dict):
        return {
            k: (REDACTED if k.lower() in keys else redact_sensitive(v, keys))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(v, keys) for v in value]
    return value


class ToolInputValidationError(ValueError):
    """Raised when raw_input fails the tool's input_schema validation."""

    def __init__(self, tool_name: str, errors: list[dict]) -> None:
        self.tool_name = tool_name
        self.errors = errors
        super().__init__(f"Invalid input for tool {tool_name!r}: {errors}")


class ToolResult(BaseModel):
    """What the pipeline returns to the caller."""

    tool_name: str
    status: str  # DENIED | EXECUTED | FAILED
    output: dict | None = None
    error: str | None = None
    denial_reason: str | None = None
    latency_ms: int | None = None


class ToolPipeline:
    def __init__(
        self,
        db: Session,
        registry: ToolRegistry | None = None,
        policy_engine: PolicyEngine | None = None,
    ) -> None:
        self.db = db
        self.registry = registry
        self.policy_engine = policy_engine

    def _default_registry(self) -> ToolRegistry:
        if self.registry is None:
            from app.tools import build_runtime_registry

            self.registry = build_runtime_registry()
        return self.registry

    def _default_policy_engine(self) -> PolicyEngine:
        if self.policy_engine is None:
            self.policy_engine = PolicyEngine()
        return self.policy_engine

    # -- the five steps ------------------------------------------------------

    async def invoke(
        self,
        tool_name: str,
        raw_input: dict,
        agent_capabilities: set[str],
        ctx: ExecutionContext,
    ) -> ToolResult:
        """Run the full validate -> capability -> policy -> execute -> audit
        sequence. Every allowed/denied call writes exactly one ``tool_calls``
        row; contract errors (unknown tool, invalid input) raise."""
        # (0) resolve — unknown tool is a contract error, not an invocation.
        # ``ToolNotFoundError`` propagates to the caller (never a silent no-op).
        tool = self._default_registry().get(tool_name)

        # (1) validate — no partial execution: nothing runs on bad input.
        try:
            validated: BaseModel = tool.input_schema.model_validate(raw_input)
        except ValidationError as exc:
            raise ToolInputValidationError(tool_name, exc.errors()) from exc

        # (2) capability check — every required capability must be present.
        missing = [cap for cap in tool.capabilities if cap not in agent_capabilities]
        if missing:
            reason = f"missing required capability: {', '.join(sorted(missing))}"
            logger.warning("Tool %s denied: %s", tool_name, reason)
            return self._record_denied(tool, validated, ctx, reason)

        # (3) policy check — deterministic, fail-closed; DENY is final.
        decision = self._default_policy_engine().evaluate(tool, validated, agent_capabilities)
        if not decision.allowed:
            logger.warning("Tool %s denied by policy rule %s: %s", tool_name, decision.rule, decision.reason)
            return self._record_denied(tool, validated, ctx, decision.reason)

        # (4) execute under an audit row (status ALLOWED -> EXECUTED/FAILED).
        row = ToolCall(
            task_id=ctx.task_id,
            step_id=ctx.step_id,
            agent_type=ctx.agent_type,
            tool_name=tool.name,
            # mode="json": UUIDs/datetimes become JSON-safe scalars for the row.
            input=redact_sensitive(validated.model_dump(mode="json")),
            status=ToolCallStatus.ALLOWED.value,
            risk=tool.risk,
        )
        self.db.add(row)
        self.db.flush()  # assign id; the audit row must not depend on execute

        started = time.perf_counter()
        try:
            output = await tool.execute(validated, ctx)
        except Exception as exc:  # noqa: BLE001 — any tool error is a FAILED row
            latency_ms = int((time.perf_counter() - started) * 1000)
            row.status = ToolCallStatus.FAILED.value
            row.output = {"error": str(exc)}
            row.latency_ms = latency_ms
            self.db.commit()  # audit row survives whatever the caller does
            logger.error("Tool %s failed after %dms: %s", tool_name, latency_ms, exc)
            return ToolResult(tool_name=tool_name, status="FAILED", error=str(exc), latency_ms=latency_ms)

        latency_ms = int((time.perf_counter() - started) * 1000)
        row.status = ToolCallStatus.EXECUTED.value
        row.output = redact_sensitive(output.model_dump(mode="json"))
        row.latency_ms = latency_ms
        self.db.commit()
        logger.info("Tool %s executed in %dms", tool_name, latency_ms)
        return ToolResult(
            tool_name=tool_name,
            status="EXECUTED",
            output=output.model_dump(mode="json"),
            latency_ms=latency_ms,
        )

    def _record_denied(
        self,
        tool: Tool,
        validated: BaseModel,
        ctx: ExecutionContext,
        reason: str,
    ) -> ToolResult:
        """Write the single DENIED row for a gate failure (never executes)."""
        self.db.add(
            ToolCall(
                task_id=ctx.task_id,
                step_id=ctx.step_id,
                agent_type=ctx.agent_type,
                tool_name=tool.name,
                input=redact_sensitive(validated.model_dump(mode="json")),
                status=ToolCallStatus.DENIED.value,
                denial_reason=reason,
                risk=tool.risk,
            )
        )
        self.db.commit()
        return ToolResult(tool_name=tool.name, status="DENIED", denial_reason=reason)


def make_execution_context(
    *,
    task_id: uuid.UUID | None = None,
    step_id: uuid.UUID | None = None,
    agent_type: str | None = None,
    db=None,
) -> ExecutionContext:
    """Convenience builder for tests and callers."""
    return ExecutionContext(task_id=task_id, step_id=step_id, agent_type=agent_type, db=db)
