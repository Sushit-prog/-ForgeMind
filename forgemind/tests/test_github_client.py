"""GitHub REST client error taxonomy + bounded transient retry (Phase 10)."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.github.client import GitHubClient
from app.github.errors import (
    GitHubAuthError,
    GitHubNotFoundError,
    GitHubRateLimitError,
)


def make_client(handler, *, max_retries=3) -> GitHubClient:
    transport = httpx.MockTransport(handler)
    return GitHubClient(
        "test-token", timeout_seconds=5, max_retries=max_retries, transport=transport
    )


def run(coro):
    return asyncio.run(coro)


def test_get_issue_ok() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-token"
        return httpx.Response(
            200,
            json={
                "number": 12,
                "title": "The bug",
                "state": "open",
                "html_url": "https://github.com/o/r/issues/12",
                "body": "detail",
            },
        )

    issue = run(make_client(handler).get_issue("o", "r", 12))
    assert issue.number == 12
    assert issue.title == "The bug"
    assert issue.html_url.endswith("/issues/12")


def test_auth_error_is_permanent() -> None:
    client = make_client(lambda req: httpx.Response(401, text="bad creds"))
    with pytest.raises(GitHubAuthError):
        run(client.get_issue("o", "r", 1))


def test_not_found_is_config_error() -> None:
    client = make_client(lambda req: httpx.Response(404, text="nope"))
    with pytest.raises(GitHubNotFoundError):
        run(client.get_issue("o", "r", 999))


def test_rate_limit_429_is_transient_and_retries() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(
            200,
            json={"number": 1, "title": "t", "state": "open", "html_url": "u"},
        )

    issue = run(make_client(handler).get_issue("o", "r", 1))
    assert issue.number == 1
    assert calls["n"] == 3  # two retries then success


def test_rate_limit_exhausts_after_retries() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429)

    client = make_client(handler, max_retries=2)
    with pytest.raises(GitHubRateLimitError):
        run(client.get_issue("o", "r", 1))
    assert calls["n"] == 3  # initial + 2 retries


def test_403_with_exhausted_limit_is_rate_limit_not_auth() -> None:
    client = make_client(
        lambda req: httpx.Response(
            403, headers={"X-RateLimit-Remaining": "0"}, text="limit"
        )
    )
    with pytest.raises(GitHubRateLimitError):
        run(client.get_issue("o", "r", 1))


def test_5xx_is_transient() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(503, text="busy")
        return httpx.Response(
            200,
            json={"number": 1, "title": "t", "state": "open", "html_url": "u"},
        )

    assert run(make_client(handler).get_issue("o", "r", 1)).number == 1
    assert calls["n"] == 2


def test_create_pr_always_draft_true() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":  # find_open_pr_for_branch probe
            return httpx.Response(200, json=[])
        captured["json"] = request.read()
        return httpx.Response(
            201,
            json={
                "number": 7,
                "html_url": "https://github.com/fork/repo/pull/7",
                "draft": True,
                "base": {"ref": "main"},
            },
        )

    pr = run(
        make_client(handler).create_pr(
            "fork",
            "repo",
            head="agent/task-1",
            base="main",
            title="t",
            body="b",
        )
    )
    import json

    payload = json.loads(captured["json"])
    assert payload["draft"] is True
    assert payload["head"] == "agent/task-1"
    assert payload["base"] == "main"
    assert pr.number == 7
    assert pr.status == "draft"
    assert pr.repo == "fork/repo"


def test_find_open_pr_reuses_existing() -> None:
    """create_pr must NOT duplicate when a PR for the branch already exists —
    it detects via find_open_pr_for_branch and reuses (Section 25 idempotency)."""
    seen_creates = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                json=[
                    {
                        "number": 3,
                        "state": "open",
                        "html_url": "https://github.com/fork/repo/pull/3",
                        "head": {"ref": "agent/task-1"},
                        "base": {"ref": "main"},
                    }
                ],
            )
        seen_creates["n"] += 1
        return httpx.Response(
            201,
            json={"number": 9, "html_url": "u", "draft": True, "base": {"ref": "main"}},
        )

    client = make_client(handler)
    pr = run(
        client.create_pr(
            "fork", "repo", head="agent/task-1", base="main", title="t", body="b"
        )
    )
    assert pr.number == 3  # reused, NOT created
    assert pr.status == "open"
    assert seen_creates["n"] == 0


def test_no_token_raises_at_construction() -> None:
    with pytest.raises(GitHubAuthError):
        GitHubClient("")


# --- Phase 12: get_pr / merge_pr --------------------------------------------


def test_get_pr_returns_current_state_and_base_sha() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/repos/fork/repo/pulls/7"
        return httpx.Response(
            200,
            json={
                "number": 7,
                "state": "open",
                "html_url": "https://github.com/fork/repo/pull/7",
                "mergeable": True,
                "mergeable_state": "clean",
                "draft": False,
                "head": {"ref": "agent/task-1", "sha": "head-abc"},
                "base": {"ref": "main", "sha": "base-def"},
                "merged": False,
            },
        )

    pr = run(make_client(handler).get_pr("fork", "repo", 7))
    assert pr.number == 7
    assert pr.state == "open"
    assert pr.mergeable is True
    assert pr.mergeable_state == "clean"
    assert pr.head_sha == "head-abc"
    assert pr.base_sha == "base-def"
    assert pr.base_ref == "main"
    assert pr.merged is False


def test_create_pr_captures_base_sha() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=[])
        return httpx.Response(
            201,
            json={
                "number": 8,
                "html_url": "https://github.com/fork/repo/pull/8",
                "draft": True,
                "state": "open",
                "base": {"ref": "main", "sha": "base-123"},
            },
        )

    pr = run(
        make_client(handler).create_pr(
            "fork", "repo", head="agent/task-1", base="main", title="t", body="b"
        )
    )
    assert pr.base_sha == "base-123"
    assert pr.base_ref == "main"


def test_merge_pr_puts_squash_and_returns_sha() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = request.url.path
        captured["json"] = request.read()
        return httpx.Response(200, json={"merged": True, "sha": "merge-abc"})

    sha = run(make_client(handler).merge_pr("fork", "repo", 7, merge_method="squash"))
    assert sha == "merge-abc"
    import json

    payload = json.loads(captured["json"])
    assert payload == {"merge_method": "squash"}
    assert captured["method"] == "PUT"
    assert captured["url"] == "/repos/fork/repo/pulls/7/merge"


def test_merge_pr_no_sha_is_fail_closed() -> None:
    """A 200 without a sha is impossible from GitHub's real API, but if it
    ever happens the tool must NOT pretend the merge happened."""
    from app.github.errors import GitHubError

    client = make_client(lambda req: httpx.Response(200, json={"merged": True}))
    with pytest.raises(GitHubError):
        run(client.merge_pr("fork", "repo", 7, merge_method="squash"))


def test_merge_pr_non_mergeable_405_is_error() -> None:
    from app.github.errors import GitHubError

    client = make_client(
        lambda req: httpx.Response(405, text="Pull request is not mergeable")
    )
    with pytest.raises(GitHubError):
        run(client.merge_pr("fork", "repo", 7, merge_method="squash"))
