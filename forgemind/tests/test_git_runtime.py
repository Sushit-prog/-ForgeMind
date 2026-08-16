"""Git/Repository runtime tests against a REAL local git repository.

Covers: clone-once caching, per-task worktree isolation, read/list/search,
modify/diff/commit, discard + recreate identical state, and the invariant
that nothing ever touches the default branch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.git.errors import DirtyWorktreeError, GitOperationError, WorktreeNotFoundError
from app.git.operations import GitOperations
from app.git.runner import run_git
from app.git.worktree_manager import WorktreeManager
from app.models import Repository, Task
from app.repository.discovery import RepositoryDiscovery
from app.repository.file_access import FileAccess


def make_manager(db_session, tmp_path: Path) -> WorktreeManager:
    return WorktreeManager(db_session, cache_dir=tmp_path / "cache")


# --- lifecycle --------------------------------------------------------------

def test_full_lifecycle_clone_read_search_diff_commit(db_session, repo_task, tmp_path) -> None:
    repo, task = repo_task
    manager = make_manager(db_session, tmp_path)

    wt = manager.create(task.id, repo)
    wt_path = Path(wt.path)
    assert wt_path.is_dir()
    assert wt.branch_name == f"agent/task-{task.id}"
    assert wt.status == "active"

    # read / list / search
    access = FileAccess(wt_path)
    assert "VALUE = 1" in access.read_file("src/app.py")
    assert sorted(access.list_files()) == ["README.md", "src/app.py", "tests/test_app.py"]
    assert [m.path for m in access.search("assert True")] == ["tests/test_app.py"]

    # modify -> diff -> commit
    (wt_path / "src" / "app.py").write_text("VALUE = 2\n")
    ops = GitOperations(wt_path, base_commit=wt.base_commit)
    assert not ops.status().clean
    assert "VALUE = 2" in ops.diff()

    sha = ops.commit("fix: bump VALUE")
    assert sha
    assert run_git(wt_path, "log", "-1", "--format=%s").stdout.strip() == "fix: bump VALUE"
    assert run_git(wt_path, "rev-parse", "HEAD").stdout.strip() == sha
    assert ops.status().clean


def test_default_branch_head_used_as_base(db_session, repo_task, tmp_path) -> None:
    repo, task = repo_task
    manager = make_manager(db_session, tmp_path)
    wt = manager.create(task.id, repo)

    clone = Path(repo.local_clone_path)
    expected = run_git(clone, "rev-parse", "origin/main").stdout.strip()
    assert wt.base_commit == expected
    # The worktree's HEAD starts exactly at that commit.
    assert run_git(Path(wt.path), "rev-parse", "HEAD").stdout.strip() == expected


def test_clone_happens_once_not_per_task(db_session, repo_task, tmp_path) -> None:
    repo, task = repo_task
    manager = make_manager(db_session, tmp_path)
    manager.create(task.id, repo)
    clone = Path(repo.local_clone_path)
    assert clone.is_dir()

    # Second task, same repo: the clone is reused (no re-clone).
    task2 = Task(objective="second", repository_id=repo.id)
    db_session.add(task2)
    db_session.commit()
    manager.create(task2.id, repo)
    assert Path(repo.local_clone_path) == clone


# --- isolation --------------------------------------------------------------

def test_two_tasks_get_independent_worktrees(db_session, repo_task, tmp_path) -> None:
    repo, task = repo_task
    manager = make_manager(db_session, tmp_path)
    wt1 = manager.create(task.id, repo)

    task2 = Task(objective="second task", repository_id=repo.id)
    db_session.add(task2)
    db_session.commit()
    wt2 = manager.create(task2.id, repo)

    assert wt1.id != wt2.id
    assert Path(wt1.path) != Path(wt2.path)

    # Uncommitted changes in wt1 must not leak into wt2.
    (Path(wt1.path) / "src" / "app.py").write_text("VALUE = 99\n")
    assert "VALUE = 99" in FileAccess(Path(wt1.path)).read_file("src/app.py")
    assert "VALUE = 1" in FileAccess(Path(wt2.path)).read_file("src/app.py")


def test_commit_on_one_worktree_does_not_move_other(db_session, repo_task, tmp_path) -> None:
    repo, task = repo_task
    manager = make_manager(db_session, tmp_path)
    wt1 = manager.create(task.id, repo)

    task2 = Task(objective="second", repository_id=repo.id)
    db_session.add(task2)
    db_session.commit()
    wt2 = manager.create(task2.id, repo)

    (Path(wt1.path) / "src" / "app.py").write_text("VALUE = 7\n")
    GitOperations(Path(wt1.path)).commit("fix: seven")
    assert GitOperations(Path(wt2.path)).status().clean  # wt2 untouched
    assert run_git(Path(wt2.path), "rev-parse", "HEAD").stdout.strip() == wt2.base_commit


# --- never touch main -------------------------------------------------------

def test_commit_never_touches_default_branch(db_session, repo_task, tmp_path) -> None:
    repo, task = repo_task
    manager = make_manager(db_session, tmp_path)
    wt = manager.create(task.id, repo)
    clone = Path(repo.local_clone_path)
    main_before = run_git(clone, "rev-parse", "origin/main").stdout.strip()

    (Path(wt.path) / "src" / "app.py").write_text("VALUE = 5\n")
    GitOperations(Path(wt.path), base_commit=wt.base_commit).commit("fix: five")

    main_after = run_git(clone, "rev-parse", "origin/main").stdout.strip()
    assert main_after == main_before
    # The commit lives on the agent branch only.
    assert run_git(clone, "rev-parse", wt.branch_name).stdout.strip() == run_git(
        Path(wt.path), "rev-parse", "HEAD"
    ).stdout.strip()


def test_create_branch_starts_at_base_commit_not_main_head(db_session, repo_task, tmp_path) -> None:
    repo, task = repo_task
    manager = make_manager(db_session, tmp_path)
    wt = manager.create(task.id, repo)
    clone = Path(repo.local_clone_path)

    ops = GitOperations(Path(wt.path), base_commit=wt.base_commit)
    ops.create_branch("feature/xyz")
    assert run_git(Path(wt.path), "rev-parse", "feature/xyz").stdout.strip() == wt.base_commit
    # Branch creation must not switch the current branch or move main.
    assert run_git(Path(wt.path), "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == wt.branch_name
    assert run_git(clone, "rev-parse", "origin/main").stdout.strip() == wt.base_commit


def test_create_branch_duplicate_name_fails(db_session, repo_task, tmp_path) -> None:
    repo, task = repo_task
    manager = make_manager(db_session, tmp_path)
    wt = manager.create(task.id, repo)
    ops = GitOperations(Path(wt.path), base_commit=wt.base_commit)
    ops.create_branch("feature/dup")
    with pytest.raises(GitOperationError):
        ops.create_branch("feature/dup")


# --- commit edge cases ------------------------------------------------------

def test_commit_with_no_changes_fails(db_session, repo_task, tmp_path) -> None:
    repo, task = repo_task
    manager = make_manager(db_session, tmp_path)
    wt = manager.create(task.id, repo)
    with pytest.raises(GitOperationError, match="nothing to commit"):
        GitOperations(Path(wt.path)).commit("empty commit should fail")


def test_commit_requires_message(db_session, repo_task, tmp_path) -> None:
    repo, task = repo_task
    manager = make_manager(db_session, tmp_path)
    wt = manager.create(task.id, repo)
    (Path(wt.path) / "src" / "app.py").write_text("VALUE = 3\n")
    with pytest.raises(GitOperationError, match="message"):
        GitOperations(Path(wt.path)).commit("   ")


def test_commit_uses_fixed_identity(db_session, repo_task, tmp_path) -> None:
    repo, task = repo_task
    manager = make_manager(db_session, tmp_path)
    wt = manager.create(task.id, repo)
    (Path(wt.path) / "src" / "app.py").write_text("VALUE = 8\n")
    GitOperations(Path(wt.path)).commit("fix: identity")
    author = run_git(Path(wt.path), "log", "-1", "--format=%an <%ae>").stdout.strip()
    assert author == "ForgeMind Agent <agent@forgemind.local>"


# --- discard / recovery -----------------------------------------------------

def test_discard_then_path_for_raises(db_session, repo_task, tmp_path) -> None:
    repo, task = repo_task
    manager = make_manager(db_session, tmp_path)
    wt = manager.create(task.id, repo)
    manager.discard(wt.id)
    assert wt.status == "discarded"
    with pytest.raises(WorktreeNotFoundError):
        manager.path_for(wt.id)


def test_discard_and_recreate_identical_starting_state(db_session, repo_task, tmp_path) -> None:
    repo, task = repo_task
    manager = make_manager(db_session, tmp_path)
    wt1 = manager.create(task.id, repo)

    state_before = {
        "app": FileAccess(Path(wt1.path)).read_file("src/app.py"),
        "readme": FileAccess(Path(wt1.path)).read_file("README.md"),
        "head": run_git(Path(wt1.path), "rev-parse", "HEAD").stdout.strip(),
    }

    # Dirty the worktree, then discard + recreate (Section J recovery path).
    (Path(wt1.path) / "src" / "app.py").write_text("VALUE = 999\n")
    manager.discard(wt1.id)
    wt2 = manager.create(task.id, repo)

    assert Path(wt2.path).is_dir()
    assert FileAccess(Path(wt2.path)).read_file("src/app.py") == state_before["app"]
    assert FileAccess(Path(wt2.path)).read_file("README.md") == state_before["readme"]
    assert run_git(Path(wt2.path), "rev-parse", "HEAD").stdout.strip() == state_before["head"]


def test_manually_deleted_worktree_detected(db_session, repo_task, tmp_path) -> None:
    repo, task = repo_task
    manager = make_manager(db_session, tmp_path)
    wt = manager.create(task.id, repo)
    import shutil

    shutil.rmtree(Path(wt.path))  # external corruption
    with pytest.raises(WorktreeNotFoundError):
        manager.path_for(wt.id)


def test_create_twice_same_task_rejected(db_session, repo_task, tmp_path) -> None:
    repo, task = repo_task
    manager = make_manager(db_session, tmp_path)
    manager.create(task.id, repo)
    with pytest.raises(DirtyWorktreeError):
        manager.create(task.id, repo)


# --- discovery edge cases ---------------------------------------------------

def test_clone_failure_bad_url_reported_at_discovery(db_session, tmp_path) -> None:
    repo = Repository(url=str(tmp_path / "does-not-exist"), default_branch="main")
    db_session.add(repo)
    db_session.commit()
    discovery = RepositoryDiscovery(cache_dir=tmp_path / "cache")
    with pytest.raises(GitOperationError, match="clone"):
        discovery.clone_if_absent(repo)
    assert repo.local_clone_path is None  # not half-cached
