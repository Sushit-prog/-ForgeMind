"""PullRequest schema (architecture doc section G, Phase 10 + Phase 12).

The typed artifact the GitHub Agent returns. ``repo`` is always the FORK
slug (e.g. ``sushit-prog/pydantic-ai``), ``status`` reflects what was
actually created on GitHub (a draft by default). ``awaiting_approval`` /
``approved`` / ``rejected`` are the human-checkpoint progression that the
API endpoints drive. ``base_sha`` is the base branch SHA captured at PR
creation — the reference the Phase 12 merge staleness check compares the
fresh base against.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PRStatus = Literal["draft", "open", "awaiting_approval", "approved", "rejected"]


class PullRequest(BaseModel):
    repo: str = Field(min_length=1, max_length=2048)
    branch: str = Field(min_length=1)
    number: int = Field(ge=1)
    url: str = Field(min_length=1)
    status: PRStatus = "draft"
    base_sha: str | None = None
