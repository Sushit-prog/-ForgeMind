"""Install-command policy (architecture doc section 27, Phase 13).

Mirror of ``command_policy`` but for dependency provisioning: the command
stored on ``repositories.install_command`` is a SERVER-SIDE value — detected
and validated at discovery time, never derived from agent/LLM input —
and ``shell.install_deps`` takes no command argument at all.

Rules (identical posture to ``command_policy``):

- The binary (first token) must be on the install allowlist, exact match
  (``pip`` for now; npm/go/cargo arrive with their own probe commands).
- Every token must be free of shell metacharacters — no ``;``, ``&&``,
  ``|``, redirects, backticks, substitution, globbing, quotes, escapes,
  ``~``, whitespace, or parent-escape path components.
- ``pip`` is free-form (its args — ``install -e ".[dev]"`` etc. — carry no
  operator-escape risk beyond the shared token checks above), the same trade
  the test policy already makes for pytest/ruff.
"""
from __future__ import annotations

# Exact binary match — the first token must be one of these, nothing else.
INSTALL_ALLOWED_BINARIES = frozenset({"pip"})

# Binaries whose arguments are validated individually (the shared checks
# below: metacharacters, absolute/home paths, drive letters, escapes).
INSTALL_FREE_FORM_BINARIES = frozenset({"pip"})

# Characters that would compose/pivot a command — never legal inside a token.
# Note: deliberately NARROWER than ``command_policy``'s set. The install
# body is executed ONLY as an argument list against the venv's fixed pip
# binary (never a shell backend), so pip's legitimate extras syntax
# (``.[dev]``) and glob characters are inert here. Dropped: ``{}*?[]``
# (globbing — meaningless without a shell). Kept: composition, redirection,
# substitution, quoting, escapes, whitespace, and path-escape chars.
INSTALL_METACHARS = set(";&|<>`$!()'\"\\\n\r\t~#")


class InstallCommandError(ValueError):
    """An install command is disallowed or malformed (security-relevant).

    Raised at discovery time (a rejected command is never stored) and at
    invocation time (a mis-stored value is never executed via ``pip``).
    """

    # Not a pytest test class (pytest collects classes named Test*).
    __test__ = False


def _reject(command: str, reason: str) -> None:
    raise InstallCommandError(f"install command {command!r} rejected: {reason}")


def _has_parent_escape(token: str) -> bool:
    """True iff ``token`` contains a ``..`` path component (POSIX separators)."""
    return token == ".." or token.startswith("../") or "/.." in token


def validate_install_command(command: str) -> list[str]:
    """Validate ``command`` and return its token list for execution.

    Raises ``InstallCommandError`` on any violation. The returned tokens are
    the ONLY command body the provisioner may run inside the venv — already
    validated, never re-derived from agent input.
    """
    import shlex

    command = (command or "").strip()
    if not command:
        _reject(command, "empty command")

    try:
        tokens = shlex.split(command, posix=True)
    except ValueError as exc:
        _reject(command, f"unparseable: {exc}")

    if not tokens:
        _reject(command, "no tokens")

    binary = tokens[0]
    if binary not in INSTALL_ALLOWED_BINARIES:
        _reject(command, f"binary {binary!r} is not on the install allowlist")
    if binary not in INSTALL_FREE_FORM_BINARIES:
        _reject(command, f"binary {binary!r} has no argument policy")

    for token in tokens:
        if any(ch in INSTALL_METACHARS for ch in token):
            _reject(command, f"token {token!r} contains shell metacharacters")
        if token.startswith(("/", "~")):
            _reject(command, f"token {token!r} is an absolute or home path")
        if len(token) >= 2 and token[1] == ":":
            _reject(command, f"token {token!r} is a drive-letter path")
        if _has_parent_escape(token):
            _reject(command, f"token {token!r} escapes the worktree root")

    return tokens