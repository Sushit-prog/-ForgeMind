"""FailureClassification model (architecture doc section G, Phase 8).

Persists what the Debugger Agent concluded about a failing test run: the
Section-10 category, a root-cause explanation, a CONCRETE fix instruction
handed to the Developer's next run (never "fix the error"), whether the
failure is code-fixable at all, and whether it was determined to be flaky
via the single re-run. The row links the task and the failing TestRun, so
downstream Review can trust the classification without re-deriving it.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, utcnow


class FailureClassification(Base):
    __tablename__ = "failure_classifications"
    __table_args__ = (
        Index("ix_failure_classifications_task_id", "task_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    test_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("test_runs.id", ondelete="SET NULL"), nullable=True
    )
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    root_cause: Mapped[str] = mapped_column(Text, nullable=False)
    fix_instruction: Mapped[str | None] = mapped_column(Text, nullable=True)
    fixable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_flaky: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, server_default=func.now()
    )
