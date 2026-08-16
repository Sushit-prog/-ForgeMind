"""Worktree manager (architecture doc sections F/H and J).

The ONLY place a branch is created in this phase. Per task:

- the repo is cloned once (cached) via ``RepositoryDiscovery``,
- ``git worktree add -b agent/task-{task_id} <path> <base_commit>`` creates
  an isolated working tree starting at the repo's default-branch HEAD (or
  an explicit base commit),
- every file/git operation resolves the worktree SERVER-SIDE from its id
  (``path_for``) — agents never supply filesystem paths.

No operation in this phase ever checks out, commits to, or otherwise
touches the default branch: the cached clone is ``--no-checkout`` (no
default-branch working tree exists on disk) and worktrees live on their
own branches.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.git.errors import DirtyWorktreeError, GitOperationError, WorktreeNotFoundError
from app.git.runner import run_git
from app.models import Repository, Worktree
from app.repository.discovery import RepositoryDiscovery

logger = logging.getLogger(__name__)


class WorktreeManager:
    def __init__(self, db: Session, cache_dir: Path | None = None) -> None:
        self.db = db
        self.cache_dir = Path(cache_dir or get_settings().repo_cache_dir).resolve()
        self.discovery = RepositoryDiscovery(cache_dir=self.cache_dir)

    # -- lifecycle ----------------------------------------------------------

    def create(
        self,
        task_id: uuid.UUID,
        repository: Repository,
        base_commit: str | None = None,
    ) -> Worktree:
        """Create the task's isolated worktree from a base commit.

        Defaults ``base_commit`` to the repository's default-branch HEAD
        (read-only — a starting point, never a branch we write to).
        """
        existing = self.db.scalar(
            select(Worktree).where(
                Worktree.task_id == task_id, Worktree.status == "active"
            )
        )
        if existing is not None:
            raise DirtyWorktreeError(
                f"task {task_id} already has an active worktree "
                f"({existing.path}) — discard it before recreating"
            )

        clone_path = self.discovery.clone_if_absent(repository)
        base = base_commit or self.discovery.default_branch_head(clone_path)

        branch = f"agent/task-{task_id}"
        wt_path = self.cache_dir / "worktrees" / f"{repository.id}-{task_id}"
        if wt_path.exists():
            raise DirtyWorktreeError(
                f"worktree path already exists on disk: {wt_path} — discard it first"
            )
        wt_path.parent.mkdir(parents=True, exist_ok=True)

        # Argument list only; branch/path are single arguments, never shell.
        run_git(clone_path, "worktree", "add", "-b", branch, str(wt_path), base)

        wt = Worktree(
            task_id=task_id,
            repository_id=repository.id,
            branch_name=branch,
            path=str(wt_path),
            base_commit=base,
            status="active",
        )
        self.db.add(wt)
        self.db.commit()
        self.db.refresh(wt)
        logger.info("Worktree %s created for task %s at %s", wt.id, task_id, base)
        return wt

    def discard(self, worktree_id: uuid.UUID) -> None:
        """Remove the worktree (git worktree remove) and mark it discarded.

        ``--force`` is used because discard is the cleanup/recovery path —
        Section J's "discard and recreate from base_commit" must always be
        able to reset a dirty worktree. The task branch is deleted too:
        ``git worktree remove`` leaves it behind, which would block the
        recreate step. If the directory is already gone (external
        deletion), the clone is pruned instead.
        """
        wt = self.db.get(Worktree, worktree_id)
        if wt is None:
            raise WorktreeNotFoundError(worktree_id)

        path = Path(wt.path)
        repository = self.db.get(Repository, wt.repository_id)
        clone = (
            Path(repository.local_clone_path)
            if repository and repository.local_clone_path
            else None
        )
        if clone is not None and clone.is_dir():
            if path.exists():
                try:
                    run_git(clone, "worktree", "remove", "--force", str(path))
                except GitOperationError as exc:
                    # Directory exists but git refuses — fall back to prune.
                    logger.warning("worktree remove failed, pruning: %s", exc)
                    run_git(clone, "worktree", "prune")
            else:
                run_git(clone, "worktree", "prune")
            # Branch may already be gone (e.g. after a crash mid-discard).
            run_git(clone, "branch", "-D", wt.branch_name, check=False)
        shutil.rmtree(path, ignore_errors=True)

        wt.status = "discarded"
        self.db.commit()
        logger.info("Worktree %s discarded", worktree_id)

    def path_for(self, worktree_id: uuid.UUID) -> Path:
        """Resolve a worktree id to its on-disk path — the security seam.

        Only ACTIVE worktrees resolve; a missing directory (manually
        deleted, simulating corruption) raises rather than operating on a
        phantom path.
        """
        wt = self.db.get(Worktree, worktree_id)
        if wt is None or wt.status != "active":
            raise WorktreeNotFoundError(worktree_id)
        path = Path(wt.path)
        if not path.is_dir():
            raise WorktreeNotFoundError(
                worktree_id, detail=f"worktree directory missing: {path}"
            )
        return path
