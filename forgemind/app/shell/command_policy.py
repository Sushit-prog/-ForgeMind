"""Test-command policy (architecture doc sections 27/H) — the security
foundation of the shell-execution phase.

The core guarantee: a test command is a SERVER-SIDE value — detected and
validated at discovery time, stored on ``repositories.test_command``, and
re-validated at invocation — and is NEVER derived from agent/LLM input.
``shell.run_test`` takes no command argument at all, so there is no
agent-controlled path into the subprocess: this policy validates the only
command that can ever reach the runner.

Validation rules (applied to the stored command string):

- The binary (first token) must be on the allowlist, exact match.
- The overall shape must match the allowed pattern for that binary
  (e.g. ``npm test`` / ``npm run test``, ``go test``, ``cargo test``).
- Every token must be free of shell metacharacters — no ``;``, ``&&``,
  ``|``, redirects, backticks, command substitution, globbing, quotes,
  escapes, ``~``, or whitespace (tokens are already split on whitespace).
- No token may be a path that escapes the worktree (``..`` components,
  absolute paths, drive letters). Flags like ``-q`` are legal — they are
  neither metacharacters nor paths — but ``-`` cannot appear as a bare
  option-escape (``--`` is still a flag, and metachar checks already make
  it inert; it is rejected only if it smuggles a path).

A command that fails validation is rejected LOUDLY — at discovery time the
repo row is never polluted with something that would be blocked at run
time; at invocation time the runner refuses to execute a mis-stored value.
"""

from __future__ import annotations

import shlex

# Exact binary match — the first token must be one of these, nothing else
# (no ./pytest, no pytest2, no wrapper scripts).
ALLOWED_BINARIES = frozenset({"pytest", "npm", "ruff", "mypy", "eslint", "tsc", "go", "cargo"})

# Binaries whose arguments are validated individually (flags/paths allowed,
# metacharacters and escapes rejected).
FREE_FORM_BINARIES = frozenset({"pytest", "ruff", "mypy", "eslint", "tsc"})

# Binaries whose arguments must match a fixed shape (package managers whose
# args select a script — never free-form paths).
FIXED_SHAPE = {
    "npm": (("test",), ("run", "test")),
    "go": (("test",),),
    "cargo": (("test",),),
}

# Characters the shell would interpret — never legal inside a token.
# Backslash is included so Windows-style paths (..\\..) are rejected here
# rather than slipping past the POSIX-separator checks below.
SHELL_METACHARS = set(";&|<>`$!(){}*?[]'\"\\\n\r\t~#")


class TestCommandError(ValueError):
    """A test command is disallowed or malformed (security-relevant).

    Raised at discovery time (a rejected command is never stored) and at
    invocation time (a mis-stored value is never executed). Distinct from
    ordinary tool errors so callers/tests can treat it as adversarial input.
    """

    # Not a pytest test class (pytest collects classes named Test*).
    __test__ = False


def _reject(command: str, reason: str) -> None:
    raise TestCommandError(f"test command {command!r} rejected: {reason}")


def _has_parent_escape(token: str) -> bool:
    """True iff ``token`` contains a ``..`` path component (POSIX separators)."""
    return token == ".." or token.startswith("../") or "/.." in token


def validate_test_command(command: str) -> list[str]:
    """Validate ``command`` and return its token list for execution.

    Raises ``TestCommandError`` on any violation. The returned tokens are
    the ONLY thing the runner may pass to ``subprocess.run`` — already
    validated, never re-derived from agent input.
    """
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
    if binary not in ALLOWED_BINARIES:
        _reject(command, f"binary {binary!r} is not on the allowlist")

    args = tokens[1:]
    if binary in FIXED_SHAPE:
        if tuple(args) not in FIXED_SHAPE[binary]:
            _reject(
                command,
                f"invalid arguments {args!r} for {binary!r} — allowed: "
                f"{[' '.join(a) for a in FIXED_SHAPE[binary]]}",
            )
    elif binary not in FREE_FORM_BINARIES:  # defensive: any future binary
        _reject(command, f"binary {binary!r} has no argument policy")

    for token in tokens:
        if any(ch in SHELL_METACHARS for ch in token):
            _reject(command, f"token {token!r} contains shell metacharacters")
        if token.startswith(("/", "~")):
            _reject(command, f"token {token!r} is an absolute or home path")
        if len(token) >= 2 and token[1] == ":":
            _reject(command, f"token {token!r} is a drive-letter path")
        if _has_parent_escape(token):
            _reject(command, f"token {token!r} escapes the worktree root")

    return tokens


def allowed_commands_doc() -> str:
    """Human-readable allowlist for prompts/logs."""
    return ", ".join(sorted(ALLOWED_BINARIES))
