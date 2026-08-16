"""Capability model (architecture doc sections G and H).

Capabilities are the unit of access control: agents are assigned a subset
(e.g. ``repo.read``, ``git.write``, ``shell.test``), and every tool call is
checked against the calling agent's capability set before execution.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, utcnow

RISK_LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


class Capability(Base):
    __tablename__ = "capabilities"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Default risk for a tool gated on this capability; per-tool risk may differ.
    risk: Mapped[str] = mapped_column(String(16), nullable=False, default="MEDIUM")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, server_default=func.now()
    )
