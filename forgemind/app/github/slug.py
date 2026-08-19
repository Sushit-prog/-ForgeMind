"""GitHub URL -> ``owner/repo`` parsing (Phase 10).

Server-side resolution seam: ``github.*`` tools and the client never accept
an arbitrary ``owner/repo`` from agent input — they derive the slug from
``repositories.url`` / ``repositories.fork_url`` here. Anything that is not
a GitHub-hosted URL raises ``GitHubConfigError`` (fail closed; there is no
fall back to guessing).
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from app.github.errors import GitHubConfigError

# https://github.com/owner/repo(.git) — plus http/ssh/git schemes.
# The host group may carry a userinfo prefix (git@github.com), stripped later.
_SLASH_URL = re.compile(
    r"^[a-z][a-z0-9+.-]*:/*(?:[^/@]+@)?(?P<host>[^/:]+)(?::\d+)?/(?P<owner>[^/]+)/"
    r"(?P<repo>[^/]+?)(?:\.git)?/?$"
)
# scp-style: git@github.com:owner/repo.git
_SCP_URL = re.compile(
    r"^(?:[^@]+@)?(?P<host>[^:]+):(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$"
)


def _host_of(raw: str) -> str:
    """Strip any ``user@`` userinfo prefix from a captured host."""
    return raw.rsplit("@", 1)[-1] if "@" in raw else raw


def parse_github_slug(url: str, *, host: str = "github.com") -> tuple[str, str]:
    """Return ``(owner, repo)`` for a GitHub repository URL.

    Accepts https/ssh/git/scp forms for ``github.com`` (the client's host).
    Non-GitHub hosts and malformed URLs raise ``GitHubConfigError`` so a
    bad repository configuration is diagnosed at the gate, not three steps
    later. ``repo`` is returned without a trailing ``.git``.
    """
    url = (url or "").strip()
    if not url:
        raise GitHubConfigError("repository URL is empty")

    match = _SCP_URL.match(url)
    if match is None:
        match = _SLASH_URL.match(url)
    if match is None:
        raise GitHubConfigError(f"not a GitHub URL for {host}: {url!r}")
    if _host_of(match.group("host")) != host:
        raise GitHubConfigError(
            f"not a GitHub URL for {host}: {url!r} "
            f"(host is {_host_of(match.group('host'))!r})"
        )
    owner = match.group("owner")
    repo = match.group("repo")
    if not owner or not repo:
        raise GitHubConfigError(f"cannot extract owner/repo from {url!r}")
    return owner, repo


def github_slug(url: str, *, host: str = "github.com") -> str:
    """``owner/repo`` slug for ``url``, as used in PR/issue data."""
    owner, repo = parse_github_slug(url, host=host)
    return f"{owner}/{repo}"


def is_github_url(url: str, *, host: str = "github.com") -> bool:
    """Cheap gate used by tools to fail closed before touching the network."""
    try:
        parse_github_slug(url, host=host)
        return True
    except GitHubConfigError:
        return False
