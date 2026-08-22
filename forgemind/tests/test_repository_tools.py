"""``repository.list_files`` output-bounding tests (hermetic).

Regression net for the first real-world failure: list_files on a 45k-file
monorepo serialized ~3 MB of paths into the researcher's context (~775k
tokens vs a 256k window -> provider HTTP 400 -> task ESCALATED). These tests
pin the contract that would have caught it:

- oversized trees are HARD-CAPPED at settings.list_files_max_entries
- truncated results END with an explicit sentinel the LLM cannot miss,
  naming both the shown and total counts
- ordering is depth-first (shallowest paths first) so the partial view is
  still a useful map of the repo, not an alphabetical slab
- typed ``truncated`` / ``total_entries`` fields carry the truth
- under-cap trees behave exactly as before (alphabetical, complete)
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

import app.tools.repository_tools as repo_tools
from app.config import get_settings
from app.tools.base import ExecutionContext
from app.tools.repository_tools import ListFilesInput, ListFilesTool


def run(coro):
    return asyncio.run(coro)


CTX = ExecutionContext(agent_type="researcher", task_id=uuid.uuid4())
WID = uuid.uuid4()
SENTINEL_PREFIX = "[truncated:"


@pytest.fixture()
def big_tree(tmp_path):
    """Synthetic worktree: 2 root files + 40 dirs x 30 deep files = 1202."""
    root = tmp_path / "wt"
    (root / "pkg00" / "sub" / "deep").mkdir(parents=True)
    (root / "README.md").write_text("x")
    (root / "main.py").write_text("x")
    for i in range(40):
        d = root / f"pkg{i:02d}" / "sub" / "deep"
        d.mkdir(parents=True, exist_ok=True)
        for j in range(30):
            (d / f"m{i:02d}_{j:02d}.py").write_text("x")
    return root


def _execute(monkeypatch, root):
    monkeypatch.setattr(repo_tools, "_root_for", lambda ctx, wid: root)
    return run(ListFilesTool().execute(ListFilesInput(worktree_id=WID), CTX))


# --- oversized tree: bounded + sentinel + depth-first ------------------------


def test_big_tree_is_capped_with_sentinel(monkeypatch, big_tree) -> None:
    out = _execute(monkeypatch, big_tree)

    cap = get_settings().list_files_max_entries
    assert out.truncated is True
    assert out.total_entries == 1202
    assert len(out.files) == cap + 1  # kept entries + sentinel element
    sentinel = out.files[-1]
    assert sentinel.startswith(SENTINEL_PREFIX)
    assert f"showing {cap} of {cap + 202} total entries" in sentinel
    assert "more specific path" in sentinel


def test_depth_first_ordering_keeps_shallow_entries(monkeypatch, big_tree) -> None:
    out = _execute(monkeypatch, big_tree)

    body = [f for f in out.files if not f.startswith(SENTINEL_PREFIX)]
    depths = [f.count("/") for f in body]
    assert depths == sorted(depths), "shallowest entries must come first"
    # Root-level files survive despite 1200 deeper competitors.
    assert "README.md" in body[:10]
    assert "main.py" in body[:10]
    # Exact-prefix pinning: the kept set is precisely the first `cap`
    # entries of the full depth-sorted listing (drops run deepest-last,
    # alphabetically within a depth tier).
    all_paths = [
        str(p.relative_to(big_tree)).replace("\\", "/")
        for p in big_tree.rglob("*")
        if p.is_file()
    ]
    all_paths.sort(key=lambda p: (p.count("/"), p))
    cap = get_settings().list_files_max_entries
    assert body == all_paths[:cap]


def test_tonights_scale_stays_bounded(monkeypatch, tmp_path) -> None:
    """45,000-entry tree (posthog-scale) — no disk writes, fake listing.

    This is the exact scenario that shipped: ~3.1 MB of tool output. The
    result must stay cap-sized regardless of how large the tree grows.
    """
    root = tmp_path / "wt"
    root.mkdir()

    class FakeFileAccess:
        def __init__(self, _root):
            pass

        def list_files(self, relative_dir: str = ".") -> list[str]:
            # Alphabetical slab dominated by one deep directory, as on a
            # real monorepo — the pathological input from tonight.
            return [f"aaa/pkg{i:05d}/sub/deep/file{i:05d}.py" for i in range(45_000)]

    monkeypatch.setattr(repo_tools, "_root_for", lambda ctx, wid: root)
    monkeypatch.setattr(repo_tools, "FileAccess", FakeFileAccess)

    out = run(ListFilesTool().execute(ListFilesInput(worktree_id=WID), CTX))

    cap = get_settings().list_files_max_entries
    assert out.truncated is True
    assert out.total_entries == 45_000
    assert len(out.files) == cap + 1
    assert f"showing {cap} of {45_000} total entries" in out.files[-1]


def test_env_override_respected(monkeypatch, big_tree) -> None:
    monkeypatch.setenv("LIST_FILES_MAX_ENTRIES", "50")
    get_settings.cache_clear()
    try:
        out = _execute(monkeypatch, big_tree)
        assert out.truncated is True
        assert len([f for f in out.files if not f.startswith(SENTINEL_PREFIX)]) == 50
        assert "showing 50 of 1202 total entries" in out.files[-1]
    finally:
        get_settings.cache_clear()


# --- under-cap tree: byte-for-byte unchanged behavior ------------------------


def test_small_tree_unchanged(monkeypatch, tmp_path) -> None:
    root = tmp_path / "wt"
    (root / "subdir").mkdir(parents=True)
    (root / "a.txt").write_text("x")
    (root / "subdir" / "b.txt").write_text("x")

    out = _execute(monkeypatch, root)

    assert out.truncated is False
    assert out.total_entries is None
    assert not any(f.startswith(SENTINEL_PREFIX) for f in out.files)
    # Alphabetical order preserved exactly as before this change.
    assert out.files == ["a.txt", "subdir/b.txt"]
