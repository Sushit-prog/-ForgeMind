"""Tool-call audit model (architecture doc sections F and G).

One row per tool invocation attempt — ALLOWED/DENIED/EXECUTED/FAILED —
written by the ``ToolPipeline``. Append-only: rows are inserted once and
only their terminal status/output fields are updated in place by the same
pipeline run. ``input``/``output`` are stored redacted (see pipeline).
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, JsonType, utcnow


class ToolCallStatus(str, enum.Enum):
    """Terminal state of a tool call:

    - ``ALLOWED``: passed every gate (capability + policy) — the transient
      state before ``execute`` runs (or a deferred-execution admit in a
      future queued flow).
    - ``DENIED``: rejected by the capability check or the policy engine.
    - ``EXECUTED``: ``execute`` returned successfully.
    - ``FAILED``: ``execute`` raised — error recorded, pipeline survives.
    """

    ALLOWED = "ALLOWED"
    DENIED = "DENIED"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"


class ToolCall(Base):
    __tablename__ = "tool_calls"
    __table_args__ = (
        Index("ix_tool_calls_task_id", "task_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    # Nullable so calls outside a task context still audit (the one-row
    # guarantee must never depend on a task existing).
    task_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("tasks.id", ondelete="SET NULL"), nullable=True
    )
    step_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("task_steps.id", ondelete="SET NULL"), nullable=True
    )
    agent_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    input: Mapped[dict] = mapped_column(JsonType, nullable=False)  # redacted
    output: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    denial_reason: Mapped[str | None] = mapped_column(String(512), nullable=True)
    risk: Mapped[str] = mapped_column(String(16), nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, server_default=func.now()
    )
