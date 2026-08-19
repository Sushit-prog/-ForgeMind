"""Deterministic stub GitHub client (tests / key-less dev, Phase 10).

The ``FORGEMIND_MOCK_GITHUB=1`` analogue of ``llm/mock.py``: records every
call, serves deterministic PR/issue data, and implements the SAME
idempotency seam as the real client (``find_open_pr_for_branch`` returns a
previously created PR), so the agent's reuse-on-retry path is exercised for
real. No network, no token.

Only the REST API is stubbed here. ``git.push`` is still exercised for REAL
against a local fork in tests — the stub never touches the git binary.
"""

from __future__ import annotations

import logging

from app.github.client import IssueData, PRData
from app.github.errors import GitHubNotFoundError

logger = logging.getLogger(__name__)


class StubGitHubClient:
    """In-memory deterministic stand-in for ``GitHubClient``."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._prs: dict[tuple[str, str, str], PRData] = {}
        # seeded issues: (owner, repo, number) -> IssueData. Unseeded
        # lookups return a deterministic default (never raise), so tests
        # that don't care about the issue can just call get_issue.
        self._issues: dict[tuple[str, str, int], IssueData] = {}
        self.comments: list[dict] = []
        self._next_number = 1

    def seed_issue(
        self,
        owner: str,
        repo: str,
        number: int,
        *,
        title: str = "Stubbed issue",
        html_url: str | None = None,
        body: str | None = None,
    ) -> None:
        self._issues[(owner, repo, number)] = IssueData(
            number=number,
            title=title,
            state="open",
            html_url=html_url or f"https://github.com/{owner}/{repo}/issues/{number}",
            body=body,
        )

    async def get_issue(self, owner: str, repo: str, number: int) -> IssueData:
        self.calls.append(
            {"op": "get_issue", "owner": owner, "repo": repo, "number": number}
        )
        seeded = self._issues.get((owner, repo, number))
        if seeded is not None:
            return seeded
        return IssueData(
            number=number,
            title="Stubbed issue",
            state="open",
            html_url=f"https://github.com/{owner}/{repo}/issues/{number}",
            body=None,
        )

    async def find_open_pr_for_branch(
        self, owner: str, repo: str, branch: str
    ) -> PRData | None:
        self.calls.append(
            {
                "op": "find_open_pr_for_branch",
                "owner": owner,
                "repo": repo,
                "branch": branch,
            }
        )
        return self._prs.get((owner, repo, branch))

    async def create_pr(
        self,
        owner: str,
        repo: str,
        *,
        head: str,
        base: str,
        title: str,
        body: str,
        draft: bool = True,
    ) -> PRData:
        self.calls.append(
            {
                "op": "create_pr",
                "owner": owner,
                "repo": repo,
                "head": head,
                "base": base,
                "draft": draft,
            }
        )
        existing = self._prs.get((owner, repo, head))
        if existing is not None:
            logger.info(
                "Stub reusing existing PR #%s for %s:%s", existing.number, owner, head
            )
            return existing
        number = self._next_number
        self._next_number += 1
        pr = PRData(
            repo=f"{owner}/{repo}",
            branch=head,
            number=number,
            url=f"https://github.com/{owner}/{repo}/pull/{number}",
            status="draft",
            base_ref=base,
        )
        self._prs[(owner, repo, head)] = pr
        return pr

    async def comment_on_issue(
        self, owner: str, repo: str, number: int, body: str
    ) -> None:
        self.calls.append(
            {"op": "comment_on_issue", "owner": owner, "repo": repo, "number": number}
        )
        if (owner, repo, number) not in self._issues:
            raise GitHubNotFoundError(
                f"stub: issue {owner}/{repo}#{number} not seeded — seed it in the test"
            )
        self.comments.append(
            {"owner": owner, "repo": repo, "number": number, "body": body}
        )
