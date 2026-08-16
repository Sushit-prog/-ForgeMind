"""Git/repository runtime errors (architecture doc section J)."""

from __future__ import annotations

import uuid


class GitOperationError(RuntimeError):
    """A git command failed (bad URL, duplicate branch, empty commit, ...)."""


class SecurityError(GitOperationError):
    """Base class for security-relevant failures — adversarial input, not
    ordinary tool errors. Callers (and tests) can catch this to treat
    attacks differently from routine failures."""


class DirtyWorktreeError(GitOperationError):
    """A worktree path/branch already exists on disk — discard first.

    Raised by ``WorktreeManager.create`` when a stale or dirty leftover
    blocks creating a fresh worktree for the task (Section J's
    "discard and recreate from base_commit" path must be used explicitly).
    """


class WorktreeNotFoundError(LookupError):
    """The worktree row is missing, not active, or its directory is gone."""

    def __init__(self, worktree_id: uuid.UUID | None = None, *, detail: str = "") -> None:
        self.worktree_id = worktree_id
        self.detail = detail
        message = detail or f"worktree not found: {worktree_id}"
        super().__init__(message)


class PathTraversalError(SecurityError):
    """A path resolved outside the worktree root — a security-relevant event.

    Raised for ``../`` climbs, absolute paths, and symlinks escaping the
    worktree. No read happens after this is raised.
    """

    def __init__(self, attempted: str, resolved: str = "") -> None:
        self.attempted = attempted
        self.resolved = resolved
        detail = f"path escapes worktree root: {attempted!r}"
        if resolved:
            detail += f" (resolved to {resolved!r})"
        super().__init__(detail)
