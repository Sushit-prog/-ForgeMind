"""Repository discovery and caching (architecture doc section C).

One clone per repository, cached at ``repositories.local_clone_path`` —
never re-cloned per task. Worktrees are added from this clone; nothing
ever commits to it. The clone is ``--no-checkout`` so there is no default
branch checkout on disk at all — the only working trees in the system are
per-task worktrees.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from app.config import get_settings
from app.git.errors import GitOperationError
from app.git.runner import run_git, run_git_ok
from app.models import Repository

logger = logging.getLogger(__name__)


class RepositoryDiscovery:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = Path(cache_dir or get_settings().repo_cache_dir).resolve()

    def clone_if_absent(self, repository: Repository) -> Path:
        """Return the cached clone path for ``repository``, cloning if needed.

        A stale/missing cache dir is removed and re-cloned (the row's
        ``local_clone_path`` is the source of truth; a phantom path is
        never silently reused). Clone failures (bad URL, network, auth)
        raise ``GitOperationError`` HERE — at discovery time — not three
        steps later.
        """
        cached = Path(repository.local_clone_path) if repository.local_clone_path else None
        if cached is not None and cached.is_dir() and run_git_ok(cached, "rev-parse", "--git-dir"):
            return cached

        clone_path = self.cache_dir / "clones" / str(repository.id)
        if clone_path.exists():
            shutil.rmtree(clone_path)  # stale leftover — re-clone cleanly
        clone_path.parent.mkdir(parents=True, exist_ok=True)

        # Argument list only — the URL is a single argument, never shell.
        try:
            run_git(self.cache_dir, "clone", "--no-checkout", repository.url, str(clone_path))
        except GitOperationError as exc:
            if clone_path.exists():
                shutil.rmtree(clone_path, ignore_errors=True)
            raise GitOperationError(
                f"failed to clone {repository.url}: {exc}"
            ) from exc

        repository.local_clone_path = str(clone_path)
        logger.info("Cloned repository %s -> %s", repository.url, clone_path)
        return clone_path

    def default_branch(self, clone_path: Path) -> str:
        """The remote default branch name (origin/HEAD, then main/master)."""
        try:
            branch = run_git(
                clone_path, "symbolic-ref", "--short", "refs/remotes/origin/HEAD"
            ).stdout.strip()
            if branch:
                return branch.removeprefix("origin/")
        except GitOperationError:
            pass
        for candidate in ("main", "master"):
            if run_git_ok(clone_path, "rev-parse", "--verify", f"origin/{candidate}"):
                return candidate
        raise GitOperationError("cannot determine the repository's default branch")

    def default_branch_head(self, clone_path: Path) -> str:
        """The full sha of the default branch HEAD — the worktree base."""
        branch = self.default_branch(clone_path)
        sha = run_git(clone_path, "rev-parse", f"origin/{branch}").stdout.strip()
        if not sha:
            raise GitOperationError(f"no HEAD for default branch {branch!r}")
        return sha

    def get_cached_metadata(self, repository: Repository) -> dict:
        """Cached metadata from the repositories row (no re-analysis)."""
        return {
            "url": repository.url,
            "default_branch": repository.default_branch,
            "local_clone_path": repository.local_clone_path,
            "languages": repository.languages,
            "test_command": repository.test_command,
            "lint_command": repository.lint_command,
            "build_command": repository.build_command,
        }


def clone_cache_dir_for(repository_id: uuid.UUID, cache_dir: Path) -> Path:
    """Convenience: where a repository's clone lives under ``cache_dir``."""
    return cache_dir / "clones" / str(repository_id)
