"""TestRun + Failure models (architecture doc section G, Phase 8).

``TestRun`` persists one real execution of the repository's test command:
status (passed/failed/error — error covers hang-timeouts and no-tests),
counts, exit code, duration, whether it timed out, and the raw output
(truncated). ``Failure`` rows are the parsed individual failures — test
name + output — linked to their run. The Debugger's flakiness re-run
creates a SECOND TestRun row, so the trace records both runs.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, utcnow


class TestRun(Base):
    __tablename__ = "test_runs"
    __table_args__ = (
        Index("ix_test_runs_task_id", "task_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    worktree_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("worktrees.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)  # passed|failed|error
    passed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    timed_out: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    output: Mapped[str | None] = mapped_column(Text, nullable=True)  # truncated
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, server_default=func.now()
    )

    failures: Mapped[list["Failure"]] = relationship(
        back_populates="test_run", cascade="all, delete-orphan"
    )


class Failure(Base):
    """A single parsed test failure within a run (test name + output)."""

    __tablename__ = "failures"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    test_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("test_runs.id", ondelete="CASCADE"), nullable=False
    )
    test: Mapped[str] = mapped_column(String(512), nullable=False)
    output: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, server_default=func.now()
    )

    test_run: Mapped["TestRun"] = relationship(back_populates="failures")
