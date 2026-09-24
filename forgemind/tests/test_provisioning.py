"""Dependency provisioning tests (Phase 13) — hermetic.

The provisioner's ONLY subprocess seam is ``_run_subprocess`` (module-level),
which the conftest autouse fixture replaces with a fake so no network is ever
touched. These tests probe the REAL provisioner logic — venv path derivation,
skip-on-no-command, pip argv construction, failure/timeout handling — by
pointing the seam at more specific fakes via ``monkeypatch``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.git.runner import run_git
from app.git.worktree_manager import WorktreeManager
from app.models import Repository, Task, Worktree
from app.shell import provision
from app.shell.install_policy import InstallCommandError
from app.shell.provision import install_dependencies, venv_path_for


def make_repo(tmp_path, *, with_pyproject: bool) -> Path:
    repo = tmp_path / "provision-repo"
    repo.mkdir()
    run_git(repo, "init", "-b", "main")
    if with_pyproject:
        (repo / "pyproject.toml").write_text(
            "[tool.pytest.ini_options]\ntestpaths = ['tests']\n"
        )
    (repo / "README.md").write_text("deps here\n")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-m", "initial")
    return repo


def repo_and_task(
    db_session, repo_path: Path, *, set_install=None
) -> tuple[Repository, Task]:
    repo = Repository(url=str(repo_path), default_branch="main")
    db_session.add(repo)
    db_session.flush()
    if set_install is not None:
        repo.install_command = set_install
    task = Task(objective="provision", repository_id=repo.id)
    db_session.add(task)
    db_session.commit()
    db_session.refresh(repo)
    db_session.refresh(task)
    return repo, task


def make_worktree(db_session, repo: Repository, task: Task) -> Worktree:
    return WorktreeManager(db_session).create(task.id, repo)


def pending(db_session):
    db_session.expire_all()


def test_skip_when_no_install_command(db_session, tmp_path) -> None:
    """A repo with no detected install setup is a CLEAN SKIP — not a failure."""
    repo_path = make_repo(tmp_path, with_pyproject=False)
    repo, task = repo_and_task(db_session, repo_path)
    wt = make_worktree(db_session, repo, task)
    pending(db_session)

    result = install_dependencies(db_session, wt.id)

    assert result.installed is False
    assert result.skipped is True
    assert result.exit_code is None
    assert result.error is None
    # No venv was created for a skip.
    venv = venv_path_for(repo.id, task.id)
    assert not venv.exists()


def test_success_creates_sibling_venv_and_runs_pip(
    db_session, tmp_path, monkeypatch
) -> None:
    repo_path = make_repo(tmp_path, with_pyproject=True)
    repo, task = repo_and_task(db_session, repo_path)
    wt = make_worktree(db_session, repo, task)
    pending(db_session)
    assert repo.install_command == 'pip install -e ".[dev]"'

    captured: list[tuple[list[str], Path]] = []

    def recording_run(argv, *, cwd, timeout_seconds):
        captured.append(([str(a) for a in argv], Path(cwd)))
        if "venv" in argv:
            Path(argv[-1]).mkdir(parents=True, exist_ok=True)
            return 0, "(fake) venv created", False, 1, None
        return 0, "(fake) pip ok", False, 2, None

    monkeypatch.setattr(provision, "_run_subprocess", recording_run)
    result = install_dependencies(db_session, wt.id)

    assert result.installed is True
    assert result.skipped is False
    assert result.exit_code == 0

    # venv was created as a SIBLING of the worktree (never inside the git
    # tree — provisioning must not pollute git status), under cache/venvs.
    venv = venv_path_for(repo.id, task.id)
    assert venv.is_dir()
    assert venv.parent.name == "venvs"
    wt_dir = Path(wt.path).resolve()
    assert not str(venv.resolve()).startswith(str(wt_dir))

    # pip ran as the venv's OWN binary with the validated token body, cwd =
    # the worktree (so `pip install -e .` resolves the project).
    venv = venv_path_for(repo.id, task.id)
    pip_calls = [c for c in captured if Path(c[0][0]).name == "pip"]
    assert pip_calls
    argv, cwd = pip_calls[0]
    assert argv[0] == str(venv / "bin" / "pip")
    assert argv[1:] == ["install", "-e", ".[dev]"]
    assert cwd == Path(wt.path)


def test_pip_failure_reports_failure(db_session, tmp_path, monkeypatch) -> None:
    repo_path = make_repo(tmp_path, with_pyproject=True)
    repo, task = repo_and_task(db_session, repo_path)
    wt = make_worktree(db_session, repo, task)
    pending(db_session)

    def failing_run(argv, *, cwd, timeout_seconds):
        if "venv" in argv:
            Path(argv[-1]).mkdir(parents=True, exist_ok=True)
            return 0, "", False, 1, None
        return 1, "ERROR: No matching distribution found for litellm", False, 3, None

    monkeypatch.setattr(provision, "_run_subprocess", failing_run)
    result = install_dependencies(db_session, wt.id)

    assert result.installed is False
    assert result.skipped is False
    assert result.exit_code == 1
    assert "No matching distribution" in result.output


def test_timeout_is_a_failure_not_a_skip(db_session, tmp_path, monkeypatch) -> None:
    repo_path = make_repo(tmp_path, with_pyproject=True)
    repo, task = repo_and_task(db_session, repo_path)
    wt = make_worktree(db_session, repo, task)
    pending(db_session)

    def hanging_run(argv, *, cwd, timeout_seconds):
        if "venv" in argv:
            Path(argv[-1]).mkdir(parents=True, exist_ok=True)
            return 0, "", False, 1, None
        # Simulate subprocess.TimeoutExpired: partial output, no exit code.
        return None, "partial pip log", True, 150_000, None

    monkeypatch.setattr(provision, "_run_subprocess", hanging_run)
    result = install_dependencies(db_session, wt.id)

    assert result.installed is False
    assert result.timed_out is True
    assert result.exit_code is None


def test_mis_stored_install_command_refused(db_session, tmp_path) -> None:
    """A legacy/mis-stored install_command (impossible via discovery — it
    never stores a value that fails validation) is REFUSED at invocation,
    not executed: the same guarantee command_policy gives shell.run_test."""
    repo_path = make_repo(tmp_path, with_pyproject=True)
    repo, task = repo_and_task(
        db_session, repo_path, set_install="pip install ; rm -rf /"
    )
    wt = make_worktree(db_session, repo, task)
    pending(db_session)

    with pytest.raises(InstallCommandError):
        install_dependencies(db_session, wt.id)


def test_discard_removes_the_venv(db_session, tmp_path) -> None:
    """Section-J discard-and-recreate resets dependencies too: WorktreeManager
    discard removes the sibling venv alongside the worktree."""
    repo_path = make_repo(tmp_path, with_pyproject=True)
    repo, task = repo_and_task(db_session, repo_path)
    wt = make_worktree(db_session, repo, task)
    pending(db_session)

    install_dependencies(db_session, wt.id)
    venv = venv_path_for(repo.id, task.id)
    assert venv.is_dir()

    WorktreeManager(db_session).discard(wt.id)

    assert not venv.exists()
    discarded = db_session.get(Worktree, wt.id)
    assert discarded.status == "discarded"


def test_venv_is_per_task(db_session, tmp_path) -> None:
    """Two tasks on the same repo get SEPARATE venv paths — isolation per task,
    never a shared environment."""
    repo_path = make_repo(tmp_path, with_pyproject=True)
    repo, task1 = repo_and_task(db_session, repo_path)
    task2 = Task(objective="provision too", repository_id=repo.id)
    db_session.add(task2)
    db_session.commit()
    db_session.refresh(task2)

    wt1 = make_worktree(db_session, repo, task1)
    pending(db_session)
    wt2 = make_worktree(db_session, repo, task2)
    assert wt1.id != wt2.id

    v1 = venv_path_for(repo.id, task1.id)
    v2 = venv_path_for(repo.id, task2.id)
    assert v1 != v2
    assert not v1.exists() and not v2.exists()  # venvs are lazy — first install only


def test_venv_path_is_a_cache_sibling(db_session, tmp_path) -> None:
    """The venv path is derived deterministically from repository+task, so
    teardown/recovery can always recompute it — no persisted path needed."""
    repo_path = make_repo(tmp_path, with_pyproject=True)
    repo, task = repo_and_task(db_session, repo_path)
    wt = make_worktree(db_session, repo, task)

    assert venv_path_for(repo.id, task.id) == (
        Path(wt.path).parent.parent / "venvs" / f"{repo.id}-{task.id}"
    )