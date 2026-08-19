"""GitHub client + tools (Phase 10).

The thin REST wrapper (``client.py``), URL-to-slug parsing (``slug.py``),
error taxonomy (``errors.py``), and the deterministic ``StubGitHubClient``
used by tests and key-less dev (``stub.py``). There is deliberately NO
``merge`` method here and no ``github.merge`` capability anywhere in the
codebase — merging stays a manual action on GitHub.
"""

from __future__ import annotations

import logging
import os

from app.config import get_settings
from app.github.client import IssueData, GitHubClient, PRData
from app.github.errors import GitHubError  # noqa: F401  (public surface)

logger = logging.getLogger(__name__)


def build_github_client() -> GitHubClient | None:
    """The client for the worker, chosen from settings (like the LLM
    providers): the deterministic stub wins under ``FORGEMIND_MOCK_GITHUB=1``
    (tests / key-less dev), a configured ``GITHUB_TOKEN`` yields the real
    client, and an unconfigured token returns ``None`` — its state's tasks
    then fail cleanly at PR_CREATION instead of hanging.
    """
    settings = get_settings()
    if os.environ.get("FORGEMIND_MOCK_GITHUB") == "1":
        from app.github.stub import StubGitHubClient

        logger.info("Using stub GitHub client (FORGEMIND_MOCK_GITHUB=1)")
        return StubGitHubClient()  # type: ignore[return-value]
    if settings.github_token:
        return GitHubClient(
            settings.github_token,
            base_url=settings.github_base_url,
            timeout_seconds=settings.github_api_timeout_seconds,
            max_retries=settings.github_max_retries,
        )
    logger.warning("No GitHub token configured — PR_CREATION tasks will fail")
    return None


__all__ = ["GitHubClient", "IssueData", "PRData", "build_github_client"]
