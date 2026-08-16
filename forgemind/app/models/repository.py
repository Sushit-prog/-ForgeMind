"""Repository and worktree models (architecture doc sections C and G)."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, JsonType, utcnow

WORKTREE_STATUSES = ("active", "merged", "discarded")


class Repository(Base):
    """A ForgeMind-tracked git repository."""

    __tablename__ = "repositories"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    url: Mapped[str] = mapped_column(String(2048), nullable=False, unique=True)
    default_branch: Mapped[str | None] = mapped_column(String(255), nullable=True)
    local_path: Mapped[str | None] = mapped_column(Text, nullable=True)  # worktree root
    languages: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    test_command: Mapped[str | None] = mapped_column(Text, nullable=True)
    lint_command: Mapped[str | None] = mapped_column(Text, nullable=True)
    build_command: Mapped[str | None] = mapped_column(Text, nullable=True)
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

    tasks: Mapped[list["Task"]] = relationship(back_populates="repository")  # noqa: F821


class Worktree(Base):
    """An isolated git worktree for a task — never a shared clone, never main."""

    __tablename__ = "worktrees"
    __table_args__ = (
        # Specified in architecture doc section G.
        Index("ix_worktrees_task_id", "task_id"),
        Index("ix_worktrees_repository_id", "repository_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    repository_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("repositories.id", ondelete="RESTRICT"), nullable=False
    )
    branch_name: Mapped[str] = mapped_column(String(255), nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    base_commit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, server_default=func.now()
    )
