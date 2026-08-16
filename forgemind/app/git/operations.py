"""Git operations on a worktree (architecture doc section F tool contracts).

Thin wrappers over the git binary via ``run_git`` (argument lists only —
never a shell string). ``create_branch`` starts from the worktree's stored
``base_commit``, never from main directly; ``commit`` refuses empty trees.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from app.git.errors import GitOperationError
from app.git.runner import run_git


class GitStatus(BaseModel):
    branch: str
    staged: list[str]
    unstaged: list[str]
    untracked: list[str]
    clean: bool


class CommitInfo(BaseModel):
    sha: str
    summary: str
    author: str
    date: str


class GitOperations:
    def __init__(self, worktree_path: Path, base_commit: str | None = None) -> None:
        self.path = worktree_path
        self.base_commit = base_commit

    # -- reads --------------------------------------------------------------

    def status(self) -> GitStatus:
        branch = run_git(self.path, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        porcelain = run_git(self.path, "status", "--porcelain").stdout
        staged: list[str] = []
        unstaged: list[str] = []
        untracked: list[str] = []
        for line in porcelain.splitlines():
            if not line:
                continue
            code, _, path = line[:2], "", line[3:]
            if code == "??":
                untracked.append(path)
                continue
            if code[0] != " ":
                staged.append(path)
            if len(code) > 1 and code[1] != " ":
                unstaged.append(path)
        return GitStatus(
            branch=branch,
            staged=staged,
            unstaged=unstaged,
            untracked=untracked,
            clean=not (staged or unstaged or untracked),
        )

    def diff(self, staged: bool = False) -> str:
        args = ["diff"] + (["--cached"] if staged else [])
        return run_git(self.path, *args).stdout

    def log(self, limit: int = 20) -> list[CommitInfo]:
        proc = run_git(
            self.path,
            "log",
            f"-n {limit}",
            "--format=%H%x1f%an%x1f%ai%x1f%s",
        )
        commits: list[CommitInfo] = []
        for line in proc.stdout.splitlines():
            parts = line.split("\x1f")
            if len(parts) == 4:
                commits.append(
                    CommitInfo(sha=parts[0], author=parts[1], date=parts[2], summary=parts[3])
                )
        return commits

    # -- writes -------------------------------------------------------------

    def create_branch(self, name: str) -> str:
        """Create ``name`` at the worktree's base commit (or HEAD).

        Uses ``git branch`` — a new ref, no checkout switch, never touching
        the default branch. An existing name fails loudly (git exits 1).
        """
        start = self.base_commit or "HEAD"
        run_git(self.path, "branch", name, start)
        return name

    def commit(self, message: str) -> str:
        """Stage all changes and commit; return the new commit sha.

        Refuses empty trees: if nothing changed, git itself errors and we
        surface it as a clear ``GitOperationError``.
        """
        message = (message or "").strip()
        if not message:
            raise GitOperationError("commit message is required")

        status = self.status()
        if status.clean:
            raise GitOperationError("nothing to commit — worktree has no changes")

        run_git(self.path, "add", "-A")
        proc = run_git(self.path, "commit", "-m", message)
        sha = run_git(self.path, "rev-parse", "HEAD").stdout.strip()
        if not sha:
            raise GitOperationError(f"commit produced no sha: {proc.stdout.strip()}")
        return sha
