"""ResearchArtifact model (architecture doc section G).

Persists what the Research Agent concluded: root-cause hypothesis, the
files/tests it claims are relevant, the evidence it actually gathered, and
a confidence score. The relevant-files content is cross-checked against
the tool-use loop before persistence (see ``agents.researcher``).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, JsonType, utcnow


class ResearchArtifact(Base):
    __tablename__ = "research_artifacts"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    root_cause_hypothesis: Mapped[str] = mapped_column(Text, nullable=False)
    relevant_files: Mapped[list] = mapped_column(JsonType, nullable=False)
    relevant_tests: Mapped[list] = mapped_column(JsonType, nullable=False)
    evidence: Mapped[list] = mapped_column(JsonType, nullable=False)
    confidence: Mapped[float] = mapped_column(Numeric(4, 2), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, server_default=func.now()
    )
