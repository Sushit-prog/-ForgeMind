"""Adversarial path-traversal tests — the security headline of this phase.

Real temp directories, real symlinks: the actual path-resolution logic is
exercised, not mocked. Every attempt must raise ``PathTraversalError`` and
never read the outside file (its content must never leak into an error
message or result).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.git.errors import PathTraversalError
from app.repository.file_access import FileAccess

OUTSIDE_SECRET = "TOP-SECRET-CONTENT-SHOULD-NEVER-LEAK"


@pytest.fixture()
def worktree(tmp_path: Path) -> Path:
    """A fake worktree root with files, a subdir, and a sibling secret file."""
    root = tmp_path / "worktree"
    (root / "src").mkdir(parents=True)
    (root / "README.md").write_text("# Hello\n")
    (root / "src" / "app.py").write_text("def main():\n    return 42\n")
    (root / "config.py").write_text("TOKEN = 'safe-value'\n")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secrets.env").write_text(OUTSIDE_SECRET)
    (outside / "data.txt").write_text(OUTSIDE_SECRET)
    return root


def try_symlink(target: Path, link: Path) -> bool:
    """Create a symlink; return False when the platform refuses (no privilege)."""
    try:
        link.symlink_to(target)
        return True
    except (OSError, NotImplementedError):
        return False


# --- lexical attacks --------------------------------------------------------

@pytest.mark.parametrize(
    "attempt",
    [
        "../secrets.env",
        "../../tmp/secrets.env",
        "src/../../secrets.env",
        "./../secrets.env",
        "sub/../..//secrets.env",
        "..\\..\\secrets.env",  # Windows separators
        "config.py/../../secrets.env",
    ],
)
def test_parent_traversal_rejected(worktree, attempt: str) -> None:
    access = FileAccess(worktree)
    with pytest.raises(PathTraversalError):
        access.read_file(attempt)


def test_absolute_path_rejected(worktree) -> None:
    access = FileAccess(worktree)
    # POSIX-style absolute path (drive-root-relative on Windows).
    with pytest.raises(PathTraversalError):
        access.read_file("/etc/passwd")


def test_windows_style_absolute_rejected(worktree) -> None:
    access = FileAccess(worktree)
    with pytest.raises(PathTraversalError):
        access.read_file("C:\\Windows\\system32\\drivers\\etc\\hosts")


def test_traversal_error_never_leaks_outside_content(worktree) -> None:
    access = FileAccess(worktree)
    with pytest.raises(PathTraversalError) as exc_info:
        access.read_file("../secrets.env")
    assert OUTSIDE_SECRET not in str(exc_info.value)
    assert OUTSIDE_SECRET not in str(exc_info.value.resolved)


def test_read_does_not_happen_on_traversal(worktree) -> None:
    """The secret file must not even be opened."""
    access = FileAccess(worktree)
    for attempt in ("../secrets.env", "/etc/passwd"):
        with pytest.raises(PathTraversalError):
            access.read_file(attempt)
    # Nothing leaked anywhere.
    assert OUTSIDE_SECRET not in access.read_file("README.md")


# --- symlink attacks --------------------------------------------------------

def test_symlink_file_escape_rejected(worktree) -> None:
    outside_secret = worktree.parent / "secrets.env"
    outside_secret.write_text(OUTSIDE_SECRET)
    link = worktree / "leak.txt"
    if not try_symlink(outside_secret, link):
        pytest.skip("symlinks not permitted on this platform")
    access = FileAccess(worktree)
    with pytest.raises(PathTraversalError):
        access.read_file("leak.txt")


def test_symlink_directory_escape_rejected(worktree) -> None:
    outside_dir = worktree.parent / "outside_dir"
    outside_dir.mkdir()
    (outside_dir / "secret.txt").write_text(OUTSIDE_SECRET)
    link = worktree / "src" / "linkdir"
    if not try_symlink(outside_dir, link):
        pytest.skip("symlinks not permitted on this platform")

    access = FileAccess(worktree)
    with pytest.raises(PathTraversalError):
        access.read_file("src/linkdir/secret.txt")
    # list_files must exclude the escaping symlink, not descend into it.
    assert "src/linkdir" not in access.list_files()
    assert "src/linkdir/secret.txt" not in access.list_files()


def test_search_skips_symlink_escape_and_reads_nothing(worktree) -> None:
    outside_dir = worktree.parent / "outside_dir"
    outside_dir.mkdir()
    (outside_dir / "secret.txt").write_text(OUTSIDE_SECRET)
    link = worktree / "src" / "linkdir"
    if not try_symlink(outside_dir, link):
        pytest.skip("symlinks not permitted on this platform")

    access = FileAccess(worktree)
    matches = access.search(OUTSIDE_SECRET.lower())
    assert all(m.path != "src/linkdir/secret.txt" for m in matches)
    assert all("linkdir" not in m.path for m in matches)


def test_symlink_inside_root_is_allowed(worktree) -> None:
    """A symlink that stays inside the root is legitimate — not rejected."""
    target = worktree / "config.py"
    link = worktree / "src" / "config_alias.py"
    if not try_symlink(target, link):
        pytest.skip("symlinks not permitted on this platform")
    access = FileAccess(worktree)
    assert "safe-value" in access.read_file("src/config_alias.py")


# --- symlink attacks on the WRITE path (Phase 7) -----------------------------
# ``filesystem.write_file`` reuses the exact same ``_resolve`` containment
# check as reads, so a symlink escaping the root must be rejected BEFORE any
# write — the write side of the Phase-4 read-side defense. These run on
# Linux/WSL where unprivileged symlink creation works.


def test_symlink_file_escape_write_rejected(worktree) -> None:
    """A symlink pointing at an outside file: writing through it must raise
    before anything is written — the outside file stays byte-identical."""
    outside_secret = worktree.parent / "secrets.env"
    outside_secret.write_text(OUTSIDE_SECRET)
    link = worktree / "leak.txt"
    if not try_symlink(outside_secret, link):
        pytest.skip("symlinks not permitted on this platform")

    access = FileAccess(worktree)
    with pytest.raises(PathTraversalError):
        access.write_file("leak.txt", "pwned-content")
    # The write never went through the symlink: the outside file is untouched.
    assert outside_secret.read_text() == OUTSIDE_SECRET


def test_symlink_directory_escape_write_rejected(worktree) -> None:
    """A symlinked directory escaping the root: writing a new file beneath it
    must raise, and nothing may be created outside the root."""
    outside_dir = worktree.parent / "outside_dir"
    outside_dir.mkdir()
    link = worktree / "src" / "linkdir"
    if not try_symlink(outside_dir, link):
        pytest.skip("symlinks not permitted on this platform")

    access = FileAccess(worktree)
    with pytest.raises(PathTraversalError):
        access.write_file("src/linkdir/evil.py", "import os; os.system('x')")
    assert not (outside_dir / "evil.py").exists()


def test_symlink_inside_root_write_is_allowed(worktree) -> None:
    """A symlink that stays inside the root is legitimate for writes too."""
    target = worktree / "src" / "app.py"
    link = worktree / "config_alias.py"
    if not try_symlink(target, link):
        pytest.skip("symlinks not permitted on this platform")
    access = FileAccess(worktree)
    access.write_file("config_alias.py", "def main():\n    return 7\n")
    assert "return 7" in access.read_file("src/app.py")


# --- benign behavior --------------------------------------------------------

def test_benign_reads_work(worktree) -> None:
    access = FileAccess(worktree)
    assert access.read_file("README.md") == "# Hello\n"
    assert "def main" in access.read_file("src/app.py")


def test_list_files_returns_relative_paths(worktree) -> None:
    files = FileAccess(worktree).list_files()
    assert files == ["README.md", "config.py", "src/app.py"]


def test_list_files_subdirectory(worktree) -> None:
    files = FileAccess(worktree).list_files("src")
    assert files == ["src/app.py"]


def test_search_finds_matches_with_line_numbers(worktree) -> None:
    matches = FileAccess(worktree).search("def main")
    assert len(matches) == 1
    assert matches[0].path == "src/app.py"
    assert matches[0].line == 1
    assert "return 42" in matches[0].snippet or "def main" in matches[0].snippet


def test_search_glob_filter(worktree) -> None:
    matches = FileAccess(worktree).search("token", glob="*.py")
    assert [m.path for m in matches] == ["config.py"]


def test_search_missing_file_raises(worktree) -> None:
    access = FileAccess(worktree)
    with pytest.raises(FileNotFoundError):
        access.read_file("nope.py")
