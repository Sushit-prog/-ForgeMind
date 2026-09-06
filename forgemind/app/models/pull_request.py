"""PullRequest model (architecture doc section G, Phase 10 + Phase 12).

Persists the draft PR the GitHub Agent opened on the FORK:
``repo`` is always the fork slug (e.g. ``sushit-prog/pydantic-ai``), never
the upstream reference. ``status`` starts ``draft`` (the agent always opens
a draft) and moves on to ``awaiting_approval``/``approved``/``rejected``
as the human checkpoint progresses.

Phase 12 adds the merge state, set ONLY by the gated ``github.merge_pr``
tool after a human has approved the task:

- ``base_sha`` — the base branch SHA captured when the PR was created. The
  pre-merge staleness check (VERIFICATION's pattern: persisted reference vs
  freshly re-fetched truth) compares it against the PR's CURRENT base SHA,
  so a rebase-requiring change to the base branch is detected, not assumed.
- ``merged_at`` / ``merge_commit_sha`` — set atomically by a successful
  squash merge. A task can be ``COMPLETED`` with these unset (approved,
  never merged) or set (approved, then merged) — there is no ``MERGED``
  terminal state.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, utcnow

PR_STATUSES = ("draft", "open", "awaiting_approval", "approved", "rejected")


class PullRequest(Base):
    __tablename__ = "pull_requests"
    __table_args__ = (
        Index("ix_pull_requests_task_id", "task_id"),
        Index("ix_pull_requests_repo_branch", "repo", "branch"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    # The FORK the PR lives on ("owner/repo" of repositories.fork_url).
    repo: Mapped[str] = mapped_column(String(2048), nullable=False)
    branch: Mapped[str] = mapped_column(String(255), nullable=False)
    number: Mapped[int] = mapped_column(Integer, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="draft")
    # Phase 12 merge state (set only by github.merge_pr, post-approval).
    base_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    merged_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    merge_commit_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        server_default=func.now(),
    )
