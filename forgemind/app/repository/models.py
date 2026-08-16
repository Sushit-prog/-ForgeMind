"""Repository-domain value objects."""

from __future__ import annotations

from pydantic import BaseModel


class SearchMatch(BaseModel):
    """One search hit: file (worktree-relative), 1-based line, snippet."""

    path: str
    line: int
    snippet: str
