"""GitHub REST API errors (architecture doc section J, Phase 10).

A small taxonomy so the lifecycle can distinguish the cases that matter:

- ``GitHubAuthError``  — bad/expired token (401) or forbidden (403 when
  rate limit is NOT the cause). Permanent: the task fails loudly at
  PR_CREATION, never a confusing generic HTTP error.
- ``GitHubRateLimitError`` — 429, or 403 with exhausted rate-limit headers.
  TRANSIENT: the caller retries with bounded backoff (Phase 5's pattern).
- ``GitHubNotFoundError`` — 404 (missing issue/repo). Permanent, but
  distinct from auth so a bad URL is diagnosable.

Everything else surfaces as the base ``GitHubError``. There is no merge
method here or anywhere in this package — never.
"""

from __future__ import annotations


class GitHubError(RuntimeError):
    """Base class for all GitHub API failures."""


class GitHubAuthError(GitHubError):
    """The token is missing, invalid, or lacks access (401/403, not limits)."""


class GitHubRateLimitError(GitHubError):
    """Rate limited (429 or 403 with exhausted limit headers) — transient.

    Carries the ``retry_after`` seconds when GitHub supplied one, so the
    caller can back off payload-free instead of guessing.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        super().__init__(message)


class GitHubNotFoundError(GitHubError):
    """The issue or repository does not exist (404) — a config error."""


class GitHubConfigError(GitHubError):
    """The task/repository cannot be used with the GitHub API.

    Raised when the repository URL is not a GitHub URL the client can
    address, when ``repositories.fork_url`` is unset for a write, etc. —
    the "fail closed, never fall back to upstream" boundary.
    """
