"""Plan and plan-step models (architecture doc sections C and G)."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, JsonType, utcnow

# ForgeMind plan-step types (architecture doc section C).
PLAN_STEP_TYPES = (
    "research",
    "implement",
    "test",
    "debug",
    "review",
    "security",
    "github",
)


class Plan(Base):
    """A schema-validated execution plan produced by the Planning Agent."""

    __tablename__ = "plans"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ACTIVE")
    # Raw LLM output for debugging/reproducibility (Section 47). Stored
    # redacted + truncated; set on both successful and failed plans.
    raw_llm_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=func.now(),
        onupdate=utcnow,
    )


class PlanStep(Base):
    """A typed step within a plan — a dependency graph via ``depends_on``."""

    __tablename__ = "plan_steps"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    plan_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("plans.id", ondelete="CASCADE"), nullable=False
    )
    step_type: Mapped[str] = mapped_column(String(32), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    depends_on: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("plan_steps.id", ondelete="SET NULL"), nullable=True
    )
    params: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PENDING")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, server_default=func.now()
    )
