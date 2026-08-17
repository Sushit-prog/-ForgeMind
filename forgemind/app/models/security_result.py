"""SecurityResult model (architecture doc section G, Phase 9).

Persists what the Security Agent concluded about the developer's commit:
a PASS/FAIL decision plus the per-finding checklist categories (injection,
secrets, unsafe subprocess/network, path traversal, auth/authz) with
file/line/description/severity. Links the task and the reviewed commit.
After a FAIL, the findings become the Developer's fix instruction for the
next run.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, JsonType, utcnow


class SecurityResult(Base):
    __tablename__ = "security_results"
    __table_args__ = (
        Index("ix_security_results_task_id", "task_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    commit_sha: Mapped[str] = mapped_column(String(64), nullable=False)
    decision: Mapped[str] = mapped_column(String(16), nullable=False)
    findings: Mapped[list] = mapped_column(JsonType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, server_default=func.now()
    )
