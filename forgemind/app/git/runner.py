"""git subprocess runner.

The one place this phase talks to the git binary, and the security rules
are enforced here so they hold for every caller:

- Arguments are ALWAYS passed as a list to ``subprocess.run`` — never
  ``shell=True``, never string interpolation. Agent input can reach a
  command as an argument (e.g. a commit message or branch name) but can
  never become shell syntax.
- A fixed system identity is injected for author/committer, so agent
  input can never set ``git commit`` authorship.
- ``GIT_TERMINAL_PROMPT=0`` makes auth failures fail fast instead of
  hanging on an interactive credential prompt.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from app.git.errors import GitOperationError

# Fixed system identity — not settable by agent input.
GIT_ENV: dict[str, str] = {
    "GIT_AUTHOR_NAME": "ForgeMind Agent",
    "GIT_AUTHOR_EMAIL": "agent@forgemind.local",
    "GIT_COMMITTER_NAME": "ForgeMind Agent",
    "GIT_COMMITTER_EMAIL": "agent@forgemind.local",
    "GIT_TERMINAL_PROMPT": "0",
}


def run_git(
    cwd: Path | str,
    *args: str,
    check: bool = True,
    redact_args: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run ``git <args>`` in ``cwd`` with the fixed identity env.

    Raises ``GitOperationError`` (with the command's stderr) on non-zero
    exit when ``check`` is True. Text mode with explicit encoding so
    output is parsed deterministically on every platform.

    ``redact_args`` suppresses the argument list from error messages —
    Phase 10's ``git.push`` passes a credential-bearing URL as an argument
    and that must never reach a log line.
    """
    env = {**os.environ, **GIT_ENV}
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if check and proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        shown = "<redacted args>" if redact_args else " ".join(args)
        raise GitOperationError(f"git {shown} failed: {stderr}")
    return proc


def run_git_ok(cwd: Path | str, *args: str) -> bool:
    """True iff ``git <args>`` exits 0 (no exception, output ignored)."""
    return run_git(cwd, *args, check=False).returncode == 0
