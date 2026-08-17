"""Worktree-scoped file access (the security boundary of this phase).

``FileAccess`` is constructed with a worktree ROOT (resolved server-side
from a ``worktree_id`` — agents never supply filesystem paths). Every
operation resolves the requested path and rejects anything that escapes
the root:

- ``../`` climbs and absolute paths are rejected after normalization.
- Symlinks are followed during resolution, so a link inside the worktree
  pointing outside resolves OUTSIDE and is rejected — never followed.
- The check is ``resolved.is_relative_to(root)`` on the FULLY RESOLVED
  path (symlinks expanded), which is the airtight form: there is no
  lexical trick that survives ``resolve()``.

Every rejection raises ``PathTraversalError`` (a ``SecurityError``) and is
logged as a security-relevant event, distinct from ordinary tool errors.
No read has happened at that point.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path

from app.git.errors import PathTraversalError, WorktreeNotFoundError
from app.repository.models import SearchMatch

logger = logging.getLogger(__name__)

MAX_SEARCH_RESULTS = 100


class FileAccess:
    def __init__(self, worktree_root: Path) -> None:
        self._root = worktree_root.resolve()
        if not self._root.is_dir():
            raise WorktreeNotFoundError(
                detail=f"worktree directory missing: {self._root}"
            )

    @property
    def root(self) -> Path:
        return self._root

    def _resolve(self, relative: str) -> Path:
        """Resolve ``relative`` against the root, rejecting escapes.

        ``resolve(strict=False)`` follows symlinks and normalizes ``..``
        without requiring the target to exist (reads check existence
        afterwards, so the traversal check runs even for would-be writes).
        """
        # Root itself must be a real directory; relative never empty here
        # (tool schemas enforce min_length).
        candidate = (self._root / relative).resolve(strict=False)
        if not candidate.is_relative_to(self._root):
            logger.error(
                "SECURITY: path traversal attempt blocked: %r -> %r (worktree %s)",
                relative,
                str(candidate),
                self._root,
            )
            raise PathTraversalError(relative, str(candidate))
        return candidate

    def read_file(self, relative_path: str) -> str:
        path = self._resolve(relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"not a file in worktree: {relative_path}")
        return path.read_text(encoding="utf-8", errors="replace")

    def write_file(self, relative_path: str, content: str) -> bool:
        """Write ``content`` to ``relative_path`` inside the root.

        Uses the EXACT same ``_resolve`` containment check as reads — a
        write can never escape the worktree any more than a read can. The
        target may be a new file (parents are created) or an existing one;
        returns True if the file already existed before the write.
        """
        path = self._resolve(relative_path)
        existed = path.is_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return existed

    def list_files(self, relative_dir: str = ".") -> list[str]:
        base = self._root if relative_dir in ("", ".") else self._resolve(relative_dir)
        if not base.is_dir():
            raise FileNotFoundError(f"not a directory in worktree: {relative_dir}")
        files: list[str] = []
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dirnames[:] = [d for d in dirnames if d != ".git"]
            for fname in filenames:
                if fname == ".git":  # worktrees store .git as a FILE
                    continue
                full = Path(dirpath) / fname
                # Re-verify the resolved path (defense-in-depth for symlinks).
                try:
                    self._resolve(str(full.relative_to(self._root)))
                except PathTraversalError:
                    continue  # symlink escape — excluded from listings
                files.append(str(full.relative_to(self._root)).replace(os.sep, "/"))
        return sorted(files)

    def search(self, query: str, glob: str | None = None) -> list[SearchMatch]:
        """Case-insensitive substring search over files inside the root."""
        needle = query.lower()
        matches: list[SearchMatch] = []
        for dirpath, dirnames, filenames in os.walk(self._root, followlinks=False):
            dirnames[:] = [d for d in dirnames if d != ".git"]
            for fname in filenames:
                if fname == ".git":  # worktrees store .git as a FILE
                    continue
                full = Path(dirpath) / fname
                rel = str(full.relative_to(self._root)).replace(os.sep, "/")
                if glob and not fnmatch.fnmatch(rel, glob):
                    continue
                try:
                    resolved = self._resolve(rel)
                except PathTraversalError:
                    continue  # symlink escape — never read outside the root
                try:
                    content = resolved.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for lineno, line in enumerate(content.splitlines(), 1):
                    if needle in line.lower():
                        matches.append(
                            SearchMatch(path=rel, line=lineno, snippet=line.strip()[:200])
                        )
                        if len(matches) >= MAX_SEARCH_RESULTS:
                            return matches
        return matches
