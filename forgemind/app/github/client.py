"""Thin GitHub REST API wrapper (architecture doc section F, Phase 10 + 12).

Covers exactly the operations this system needs — read an issue, find an
existing open PR for a branch (idempotency), create a draft PR, comment on
an issue, and Phase 12's GATED squash merge (``get_pr`` + ``merge_pr``).

Auth + safety rules:

- The token comes from settings (env / .env) and is never logged; error
  bodies are truncated and the token-bearing ``Authorization`` header is
  never surfaced.
- 401/403 (non-limit) -> ``GitHubAuthError``: permanent, the task fails
  loudly at PR_CREATION. 429 / 403-with-exhausted-limit ->
  ``GitHubRateLimitError``: TRANSIENT, retried with bounded backoff (the
  Phase 5 pattern). 404 -> ``GitHubNotFoundError``: a config error, never
  a silent no-op.
- Every method is server-side-resolved by the caller from repository
  rows; the client itself never guesses a target.
- ``merge_pr`` exists ONLY as a building block for the gated
  ``github.merge_pr`` tool. Nothing calls it except that tool, and the tool
  enforces approval evidence, MERGE_ALLOWED_REPOS, and a fresh staleness
  check before it is ever reached.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from pydantic import BaseModel, Field

from app.github.errors import (
    GitHubAuthError,
    GitHubError,
    GitHubNotFoundError,
    GitHubRateLimitError,
)

logger = logging.getLogger(__name__)

# HTTP statuses worth a bounded retry (the LLM provider's transient set).
TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


def is_transient_github_error(exc: Exception) -> bool:
    """True for a GitHub failure worth a bounded retry."""
    if isinstance(exc, GitHubRateLimitError):
        return True
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, GitHubError):
        return False  # auth / not-found / config are permanent
    return False


class IssueData(BaseModel):
    """What the system reads from an upstream issue (github.get_issue)."""

    number: int
    title: str
    state: str
    html_url: str
    body: str | None = None


class PRData(BaseModel):
    """A created (or found) pull request on the FORK.

    ``repo`` is always the fork slug (e.g. ``sushit-prog/pydantic-ai``);
    the upstream reference never appears here. ``status`` is the GitHub
    PR state: ``draft`` when we created it (always) or ``open`` for an
    existing PR we reused.

    Phase 12 adds the merge-check inputs: ``state`` (open/closed), GitHub's
    own ``mergeable`` / ``mergeable_state``, the head/base SHAs, and the
    base SHA at PR-open time (``base_sha``) that the fresh staleness check
    compares against. All optional so existing constructions keep working.
    """

    repo: str
    branch: str
    number: int
    url: str
    status: str = Field(default="draft")
    base_ref: str | None = None
    state: str | None = None
    mergeable: bool | None = None
    mergeable_state: str | None = None
    head_sha: str | None = None
    base_sha: str | None = None
    merged: bool | None = None


class GitHubClient:
    """One slim client for the GitHub REST API (PAT auth)."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = "https://api.github.com",
        timeout_seconds: float = 20.0,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not token:
            raise GitHubAuthError("no GitHub token configured (GITHUB_TOKEN)")
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_base = backoff_base_seconds
        # Test seam: an injected MockTransport replaces the real network.
        self._transport = transport

    # -- HTTP plumbing -------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
    ) -> httpx.Response:
        """One bounded-retry request: transient failures back off, permanent
        failures raise immediately (never swallowed, never retried)."""
        attempt = 0
        while True:
            try:
                kwargs: dict = {"timeout": self.timeout_seconds}
                if self._transport is not None:
                    kwargs["transport"] = self._transport
                async with httpx.AsyncClient(**kwargs) as client:
                    resp = await client.request(
                        method,
                        f"{self.base_url}{path}",
                        headers=self._headers(),
                        params=params,
                        json=json_body,
                    )
            except httpx.TimeoutException as exc:
                if attempt >= self.max_retries:
                    raise GitHubError(
                        f"GitHub request timed out after {attempt + 1} attempts"
                    ) from exc
                attempt += 1
                await asyncio.sleep(self.backoff_base * (2 ** (attempt - 1)))
                continue

            if resp.status_code == 401:
                raise GitHubAuthError("GitHub API rejected the token (401)")
            if resp.status_code == 403:
                remaining = resp.headers.get("X-RateLimit-Remaining")
                if remaining is not None and remaining.strip() == "0":
                    retry_after = _retry_after_seconds(resp)
                    if attempt < self.max_retries:
                        await _backoff(attempt, self.backoff_base, retry_after)
                        attempt += 1
                        continue
                    raise GitHubRateLimitError(
                        "GitHub API rate limit exhausted (403)", retry_after=retry_after
                    )
                raise GitHubAuthError(
                    f"GitHub API denied the request (403): {resp.text[:300]}"
                )
            if resp.status_code == 429:
                retry_after = _retry_after_seconds(resp)
                if attempt < self.max_retries:
                    await _backoff(attempt, self.backoff_base, retry_after)
                    attempt += 1
                    continue
                raise GitHubRateLimitError(
                    "GitHub API rate limit hit (429)", retry_after=retry_after
                )
            if resp.status_code == 404:
                raise GitHubNotFoundError(
                    f"GitHub resource not found: {method} {path} (404) — check owner/repo/issue"
                )
            if resp.status_code in TRANSIENT_STATUSES:
                if attempt >= self.max_retries:
                    raise GitHubError(
                        f"GitHub request failed after {attempt + 1} attempts "
                        f"(HTTP {resp.status_code})"
                    )
                attempt += 1
                await asyncio.sleep(self.backoff_base * (2 ** (attempt - 1)))
                continue
            if resp.status_code >= 400:
                raise GitHubError(
                    f"GitHub {method} {path} failed (HTTP {resp.status_code}): {resp.text[:300]}"
                )
            return resp

    # -- operations ----------------------------------------------------------

    async def get_issue(self, owner: str, repo: str, number: int) -> IssueData:
        """Read an upstream issue (reads go to the upstream reference)."""
        resp = await self._request("GET", f"/repos/{owner}/{repo}/issues/{number}")
        data = resp.json()
        try:
            return IssueData(
                number=data["number"],
                title=data["title"],
                state=data["state"],
                html_url=data["html_url"],
                body=data.get("body"),
            )
        except (KeyError, TypeError) as exc:
            raise GitHubError(f"unexpected GitHub issue payload: {exc}") from exc

    async def find_open_pr_for_branch(
        self, owner: str, repo: str, branch: str
    ) -> PRData | None:
        """The open PR whose head is ``owner:branch``, or None.

        Idempotency seam (spec: PR already exists for this branch -> detect
        and reuse). ``repo`` is the FORK — this phase never looks for PRs on
        the upstream reference.
        """
        resp = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/pulls",
            params={"state": "open", "head": f"{owner}:{branch}", "per_page": 10},
        )
        for item in resp.json() or []:
            head = item.get("head") or {}
            if head.get("ref") == branch and item.get("state") == "open":
                return PRData(
                    repo=f"{owner}/{repo}",
                    branch=branch,
                    number=item["number"],
                    url=item["html_url"],
                    status="open",
                    base_ref=(item.get("base") or {}).get("ref"),
                    state=item.get("state"),
                    mergeable=item.get("mergeable"),
                    mergeable_state=item.get("mergeable_state"),
                    head_sha=(head.get("sha") if head else None),
                    base_sha=((item.get("base") or {}).get("sha")),
                    merged=item.get("merged"),
                )
        return None

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
        """Open a pull request against the FORK (head + base are both on
        the fork; the upstream reference is never a PR target here).

        ``draft=True`` is the default and the agent always passes it — the
        draft layer sits under the human-approval gate: even if the gate
        were bypassed, a draft PR on the operator's own fork cannot look
        like a ready-to-merge contribution.
        """
        existing = await self.find_open_pr_for_branch(owner, repo, head)
        if existing is not None:
            logger.info(
                "Reusing existing open PR #%d for %s:%s", existing.number, owner, head
            )
            return existing
        resp = await self._request(
            "POST",
            f"/repos/{owner}/{repo}/pulls",
            json_body={
                "title": title,
                "head": head,
                "base": base,
                "body": body,
                "draft": draft,
            },
        )
        data = resp.json()
        try:
            pr = PRData(
                repo=f"{owner}/{repo}",
                branch=head,
                number=data["number"],
                url=data["html_url"],
                status="draft" if data.get("draft") else "open",
                base_ref=(data.get("base") or {}).get("ref"),
                state=data.get("state"),
                mergeable=data.get("mergeable"),
                mergeable_state=data.get("mergeable_state"),
                head_sha=((data.get("head") or {}).get("sha")),
                base_sha=(data.get("base") or {}).get("sha"),
                merged=data.get("merged"),
            )
        except (KeyError, TypeError) as exc:
            raise GitHubError(f"unexpected GitHub PR payload: {exc}") from exc
        logger.info(
            "Created draft PR #%s on %s/%s (head=%s)", pr.number, owner, repo, head
        )
        return pr

    async def comment_on_issue(
        self, owner: str, repo: str, number: int, body: str
    ) -> None:
        """Post ``body`` as a comment on the issue. Used to link the PR."""
        await self._request(
            "POST",
            f"/repos/{owner}/{repo}/issues/{number}/comments",
            json_body={"body": body},
        )

    # -- Phase 12: gated merge ------------------------------------------------

    async def get_pr(self, owner: str, repo: str, number: int) -> PRData:
        """Fetch the PR's CURRENT state from GitHub (the fresh source of truth
        for the pre-merge staleness check — never the cached DB row)."""
        resp = await self._request("GET", f"/repos/{owner}/{repo}/pulls/{number}")
        data = resp.json()
        try:
            return PRData(
                repo=f"{owner}/{repo}",
                branch=((data.get("head") or {}).get("ref") or ""),
                number=data["number"],
                url=data["html_url"],
                status="draft" if data.get("draft") else data.get("state", "open"),
                base_ref=(data.get("base") or {}).get("ref"),
                state=data.get("state"),
                mergeable=data.get("mergeable"),
                mergeable_state=data.get("mergeable_state"),
                head_sha=((data.get("head") or {}).get("sha")),
                base_sha=(data.get("base") or {}).get("sha"),
                merged=data.get("merged"),
            )
        except (KeyError, TypeError) as exc:
            raise GitHubError(f"unexpected GitHub PR payload: {exc}") from exc

    async def merge_pr(
        self, owner: str, repo: str, number: int, *, merge_method: str = "squash"
    ) -> str:
        """Squash-merge the PR into its base branch; return the merge commit SHA.

        Only reached through the gated ``github.merge_pr`` tool — the tool
        verifies approval evidence, MERGE_ALLOWED_REPOS, and a fresh
        staleness check BEFORE calling this. A non-mergeable PR (405/409 or
        any >= 400) raises ``GitHubError`` and the tool records it as a
        failed merge.
        """
        resp = await self._request(
            "PUT",
            f"/repos/{owner}/{repo}/pulls/{number}/merge",
            json_body={"merge_method": merge_method},
        )
        data = resp.json()
        sha = data.get("sha") if isinstance(data, dict) else None
        if not sha:
            raise GitHubError(
                f"GitHub merge of {owner}/{repo}#{number} returned no sha: "
                f"{resp.text[:300]}"
            )
        logger.info("Merged %s/%s#%s (squash, sha=%s)", owner, repo, number, sha)
        return sha


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    """Best-effort Retry-After header -> seconds (rate-limit backoff)."""
    value = resp.headers.get("Retry-After")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


async def _backoff(
    attempt: int, backoff_base: float, retry_after: float | None = None
) -> None:
    """Sleep between transient retries: the server's Retry-After wins when
    GitHub provides one, otherwise exponential backoff (Phase 5's pattern)."""
    if retry_after is not None:
        await asyncio.sleep(min(retry_after, 10.0))
    else:
        await asyncio.sleep(backoff_base * (2**attempt))
