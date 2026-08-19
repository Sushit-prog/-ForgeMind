"""Task and task-step models (architecture doc sections C and G)."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, JsonType, utcnow


class TaskStatus(str, enum.Enum):
    """State-machine states for a task (architecture doc section D).

    Phase 1 only ever sets ``CREATED``; the remaining states are defined now
    so the column is ready for the Phase 4 orchestrator, which enforces legal
    transitions in code, never by asking the LLM.
    """

    CREATED = "CREATED"
    PLANNING = "PLANNING"
    RESEARCHING = "RESEARCHING"
    IMPLEMENTING = "IMPLEMENTING"
    TESTING = "TESTING"
    DEBUGGING = "DEBUGGING"
    REVIEWING = "REVIEWING"
    SECURITY_REVIEW = "SECURITY_REVIEW"
    VERIFICATION = "VERIFICATION"
    PR_CREATION = "PR_CREATION"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RECOVERING = "RECOVERING"
    REPLANNING = "REPLANNING"
    ESCALATED = "ESCALATED"


class Task(Base):
    """A single ForgeMind task: objective + repo + status + budget."""

    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    repository_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("repositories.id", ondelete="RESTRICT"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=TaskStatus.CREATED.value
    )

    # Optional GitHub issue this task originated from (Phase 10) — the
    # target of github.get_issue reads and the PR-comment link. Nullable:
    # a task may be issue-less (objective-only).
    issue_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Budget — from the domain model: max_cost/max_tokens/max_runtime/max_replans.
    # max_replans defaults to a bounded value (Section D: replan-budget
    # exhaustion -> ESCALATED, never an unbounded failure loop). The column
    # stays nullable so per-task overrides can raise or disable the cap.
    max_cost: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    max_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_runtime: Mapped[int | None] = mapped_column(Integer, nullable=True)  # seconds
    max_replans: Mapped[int | None] = mapped_column(Integer, nullable=True, default=3)
    replan_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=func.now(),
        onupdate=utcnow,
    )

    repository: Mapped["Repository"] = relationship(back_populates="tasks")  # noqa: F821


class TaskStep(Base):
    """An executed step of a task (typed: research|implement|test|debug|...)."""

    __tablename__ = "task_steps"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    step_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")
    input: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    output: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=func.now(),
    )
