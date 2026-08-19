"""GitHub URL -> owner/repo parsing (Phase 10 server-side seam)."""

import pytest

from app.github.errors import GitHubConfigError
from app.github.slug import github_slug, parse_github_slug


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://github.com/org/repo", ("org", "repo")),
        ("https://github.com/org/repo.git", ("org", "repo")),
        ("http://github.com/org/repo.git", ("org", "repo")),
        ("ssh://git@github.com/org/repo.git", ("org", "repo")),
        ("git://github.com/org/repo.git", ("org", "repo")),
        ("git@github.com:org/repo.git", ("org", "repo")),
        ("git@github.com:org/repo", ("org", "repo")),
        ("https://github.com/Org-Name/repo_123", ("Org-Name", "repo_123")),
    ],
)
def test_parse_github_slug(url, expected) -> None:
    assert parse_github_slug(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "",
        "https://github.com",  # no repo path
        "https://github.com/owner/",  # empty repo
        "https://gitlab.com/org/repo",  # not github.com
        "git@gitlab.com:org/repo.git",
        "file:///tmp/source",  # local clone — not a GitHub URL
        "/abs/path/to/repo",
    ],
)
def test_non_github_urls_fail_closed(url) -> None:
    with pytest.raises(GitHubConfigError):
        parse_github_slug(url)


def test_github_slug_returns_owner_slash_repo() -> None:
    assert github_slug("https://github.com/sushit-prog/pydantic-ai") == (
        "sushit-prog/pydantic-ai"
    )
