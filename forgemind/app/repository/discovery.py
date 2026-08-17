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
import time
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
        """Return the cached clone path for ``repository``, cloning if needed,
        and ensure the repository has a validated ``test_command``.

        A stale/missing cache dir is removed and re-cloned (the row's
        ``local_clone_path`` is the source of truth; a phantom path is
        never silently reused). Clone failures (bad URL, network, auth)
        raise ``GitOperationError`` HERE — at discovery time — not three
        steps later.
        """
        clone_path = self._ensure_clone(repository)
        self._ensure_test_command(repository, clone_path)
        return clone_path

    def _ensure_clone(self, repository: Repository) -> Path:
        cached = Path(repository.local_clone_path) if repository.local_clone_path else None
        if cached is not None and cached.is_dir() and run_git_ok(cached, "rev-parse", "--git-dir"):
            return cached

        clone_path = self.cache_dir / "clones" / str(repository.id)
        if clone_path.exists():
            # A leftover path may be an IN-PROGRESS clone by a concurrent
            # worker (git creates the destination directory immediately).
            # Deleting it would corrupt the winner's clone — so wait briefly
            # for it to become a valid git dir first; only a still-invalid
            # path is treated as a true stale leftover and re-cloned.
            for _ in range(30):  # ~3s max — a local clone takes <1s
                if run_git_ok(clone_path, "rev-parse", "--git-dir"):
                    repository.local_clone_path = str(clone_path)
                    logger.info("Reusing concurrent clone %s", clone_path)
                    return clone_path
                if not clone_path.exists():
                    break  # the concurrent worker cleaned it up
                time.sleep(0.1)
            shutil.rmtree(clone_path, ignore_errors=True)  # truly stale
        clone_path.parent.mkdir(parents=True, exist_ok=True)

        # Argument list only — the URL is a single argument, never shell.
        try:
            run_git(self.cache_dir, "clone", "--no-checkout", repository.url, str(clone_path))
        except GitOperationError as exc:
            # A concurrent worker may have won the clone race: if the path
            # now exists as a valid git dir, reuse it instead of failing
            # (research can legitimately run twice on one task — Section D
            # concurrency). Anything else is a real clone failure.
            if clone_path.exists() and run_git_ok(clone_path, "rev-parse", "--git-dir"):
                logger.info("Clone race for %s — reusing concurrent clone", repository.url)
            else:
                if clone_path.exists():
                    shutil.rmtree(clone_path, ignore_errors=True)
                raise GitOperationError(
                    f"failed to clone {repository.url}: {exc}"
                ) from exc

        repository.local_clone_path = str(clone_path)
        logger.info("Cloned repository %s -> %s", repository.url, clone_path)
        return clone_path

    def _ensure_test_command(self, repository: Repository, clone_path: Path) -> None:
        """Detect + validate ``repositories.test_command`` at discovery time
        (Phase 8).

        The command is a SERVER-SIDE value, never agent input: detected from
        the repo's own setup files (never guessed by an LLM), validated
        against the ``command_policy`` allowlist HERE — a value that fails
        validation is rejected loudly and never stored, so nothing that
        would be blocked at run time ever reaches the runner. When no test
        setup is detected the field stays None and ``shell.run_test`` fails
        with a clear message at invocation instead of a confusing
        subprocess error.

        The clone is ``--no-checkout``, so marker files are read from the
        git TREE (``git ls-tree``), never from a working directory that
        does not exist.
        """
        if repository.test_command is not None:
            return
        command = detect_test_command(clone_path)
        if command is None:
            return
        from app.shell.command_policy import validate_test_command

        validate_test_command(command)  # fails loudly, never stores a bad value
        repository.test_command = command
        logger.info(
            "Detected test_command %r for repository %s", command, repository.url
        )

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


# Marker file -> detected test command. Detection is deterministic and
# conservative: a marker exists, the command is chosen from a fixed table —
# nothing here is ever derived from agent/LLM input, and every detected
# command must pass ``command_policy.validate_test_command`` before it is
# stored (a rejected command fails discovery loudly, never reaches run time).
TEST_COMMAND_MARKERS: list[tuple[str, str]] = [
    ("pyproject.toml", "pytest"),
    ("pytest.ini", "pytest"),
    ("tox.ini", "pytest"),
    ("setup.cfg", "pytest"),
    ("package.json", "npm test"),
    ("go.mod", "go test"),
    ("Cargo.toml", "cargo test"),
]


def detect_test_command(clone_path: Path) -> str | None:
    """Detect the repository's test command from its setup files.

    The clone is ``--no-checkout``, so markers are read from the git TREE
    of HEAD, not the filesystem. Returns None when no test setup is present
    (the repository has no runnable test suite; ``shell.run_test`` then
    fails with a clear "not configured" message at invocation).
    """
    try:
        listed = run_git(clone_path, "ls-tree", "--name-only", "HEAD").stdout.splitlines()
    except GitOperationError:
        logger.warning("test-command detection: cannot list tree of %s", clone_path)
        return None
    names = {line.strip() for line in listed}
    for marker, command in TEST_COMMAND_MARKERS:
        if marker in names:
            return command
    return None


def clone_cache_dir_for(repository_id: uuid.UUID, cache_dir: Path) -> Path:
    """Convenience: where a repository's clone lives under ``cache_dir``."""
    return cache_dir / "clones" / str(repository_id)
