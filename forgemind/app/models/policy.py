"""Policy model (architecture doc sections G and H).

Policies are deterministic rules evaluated by the policy engine *outside*
the LLM's context — e.g. shell-command allowlist, no-push-to-main.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, JsonType, utcnow


class Policy(Base):
    __tablename__ = "policies"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Structured rule definition, e.g. {"type": "shell_allowlist", "allow": [...]}.
    rule: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False, default="MEDIUM")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
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
