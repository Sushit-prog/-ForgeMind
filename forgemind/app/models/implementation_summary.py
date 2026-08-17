"""ImplementationSummary model (architecture doc section G, Phase 7).

Persists what the Developer Agent did: the commit sha on the task's
worktree, the files that changed, a plain-text summary, tests touched, and
an optional explanation of any divergence from the research hypothesis.
``files_changed`` is cross-checked against the files actually written via
``filesystem.write_file`` during the tool-use loop before persistence (see
``agents.developer``).

``status`` is COMPLETE for a normal persisted summary, or INCOMPLETE for
the explicit failure marker written when the developer finished without
producing a commit (the Phase 5 INVALID-plan-row pattern: preserve what
happened, never lose it silently). ``commit_sha`` is nullable so the
INCOMPLETE marker can record the failure without inventing a commit.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, JsonType, utcnow


class ImplementationSummary(Base):
    __tablename__ = "implementation_summaries"
    __table_args__ = (
        Index("ix_implementation_summaries_task_id", "task_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    step_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("plan_steps.id", ondelete="SET NULL"), nullable=True
    )
    worktree_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("worktrees.id", ondelete="SET NULL"), nullable=True
    )
    commit_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    files_changed: Mapped[list] = mapped_column(JsonType, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    tests_added: Mapped[list] = mapped_column(JsonType, nullable=False)
    deviations_from_research: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="COMPLETE")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, server_default=func.now()
    )
