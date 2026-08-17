"""ReviewResult model (architecture doc section G, Phase 9).

Persists what the Reviewer Agent concluded about the developer's commit:
an APPROVE/REQUEST_CHANGES/REJECT decision, the per-issue details
(description, severity, file, line), and an overall severity. The row
links the task and the reviewed commit, so downstream Review/Verification
can trust the decision without re-deriving it — and the Developer's next
run (after a REQUEST_CHANGES/REJECT) receives the issues as its fix
instruction.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, JsonType, utcnow


class ReviewResult(Base):
    __tablename__ = "review_results"
    __table_args__ = (
        Index("ix_review_results_task_id", "task_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    commit_sha: Mapped[str] = mapped_column(String(64), nullable=False)
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    issues: Mapped[list] = mapped_column(JsonType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, server_default=func.now()
    )
