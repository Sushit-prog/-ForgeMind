"""Approval model (architecture doc section G, Phase 10).

``approvals`` records the HUMAN decision at the AWAITING_APPROVAL
checkpoint: action ``approve`` or ``reject`` (never auto-written by the
worker), with an optional reason. The row is append-only evidence that a
person actually reviewed ForgeMind's PR — approval means "I reviewed the
PR and consider the task done", NOT "merge this" (merging stays manual on
GitHub; this system never merges anything).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, utcnow

APPROVAL_ACTIONS = ("approve", "reject")


class Approval(Base):
    __tablename__ = "approvals"
    __table_args__ = (Index("ix_approvals_task_id", "task_id"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    action: Mapped[str] = mapped_column(String(16), nullable=False)  # approve|reject
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=func.now(),
    )
