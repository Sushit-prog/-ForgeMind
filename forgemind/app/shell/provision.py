"""Dependency provisioning for ``shell.install_deps`` (architecture doc
section F, Phase 13).

The ONLY dependency-install path in the system. Each task gets its own venv
ALONGSIDE its worktree (``venvs/{repository.id}-{task.id}``, a deterministic
sibling of ``worktrees/{repository.id}-{task.id}``):

- the venv is created by ``sys.executable -m venv`` (argument list, never
  shell) the first time the task's install runs;
- the install command comes EXCLUSIVELY from ``repositories.install_command``
  (server-side, detected + validated at discovery) and is re-validated here
  via ``install_policy`` before execution — there is no agent-input path into
  the subprocess at all, the same guarantee ``command_policy`` gives
  ``shell.run_test``;
- the command body runs in the worktree directory as the venv's own
  ``pip`` (so ``pip install -e .`` resolves the project), under a hard
  timeout (``install_timeout_seconds``) with captured output.

Everything runs as plain blocking ``subprocess.run`` — callers (the tool)
must invoke through ``asyncio.to_thread``, exactly like ``CommandRunner``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import get_settings
from app.shell.install_policy import validate_install_command

logger = logging.getLogger(__name__)

# Upper bound on captured install output (defense against a package whose
# build logs flood stdout). Aligned with the test runner's cap.
MAX_CAPTURED_OUTPUT = 500_000


class ProvisionResult:
    """The raw provisioning outcome — pre-policy, pre-LLM ground truth."""

    def __init__(
        self,
        *,
        installed: bool,
        skipped: bool = False,
        exit_code: int | None = None,
        output: str = "",
        timed_out: bool = False,
        duration_ms: int = 0,
        error: str | None = None,
    ) -> None:
        self.installed = installed
        self.skipped = skipped
        self.exit_code = exit_code
        self.output = output[:MAX_CAPTURED_OUTPUT]
        self.timed_out = timed_out
        self.duration_ms = duration_ms
        self.error = error


def _run_subprocess(
    argv: list[str], *, cwd: Path, timeout_seconds: float
) -> tuple[int | None, str, bool, int, str | None]:
    """One argument-list ``subprocess.run`` with a hard timeout.

    Returns ``(exit_code, output, timed_out, duration_ms, error)``. This is
    the ONLY subprocess seam in the module — tests shim it for hermeticity.
    """
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        duration = int((time.perf_counter() - started) * 1000)
        partial = (exc.stdout or "") + (exc.stderr or "")
        return None, partial, True, duration, None
    except OSError as exc:
        duration = int((time.perf_counter() - started) * 1000)
        return None, "", False, duration, f"failed to run command: {exc}"
    duration = int((time.perf_counter() - started) * 1000)
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output, False, duration, None


def venv_path_for(
    repository_id: uuid.UUID, task_id: uuid.UUID, cache_dir: Path | None = None
) -> Path:
    """The task's venv path — a deterministic sibling of its worktree.

    Derived, never persisted: teardown (``WorktreeManager.discard``) and
    crash recovery can always recompute where a task's venv lives. Kept OUT
    of the git worktree so provisioned files never pollute ``git status``.
    """
    root = Path(cache_dir or get_settings().repo_cache_dir).resolve()
    return root / "venvs" / f"{repository_id}-{task_id}"


def install_dependencies(
    db: Session, worktree_id: uuid.UUID
) -> ProvisionResult:
    """Provision the task's venv and install its declared dependencies.

    Mirrors ``CommandRunner`` in posture: only ACTIVE worktrees resolve (via
    ``WorktreeManager.path_for``), a missing directory raises rather than
    provisioning into a phantom path, and a mis-stored/legacy
    ``install_command`` is refused with a clear error instead of executed.

    A repository with no ``install_command`` (nothing detected at discovery)
    is a clean SKIP — ``installed=False, skipped=True`` — NOT a failure: the
    suite is expected to run on the worker environment in that case.
    """
    from app.git.worktree_manager import WorktreeManager
    from app.models import Repository, Worktree

    manager = WorktreeManager(db)
    path = manager.path_for(worktree_id)
    wt = db.get(Worktree, worktree_id)
    if wt is None:
        raise RuntimeError(f"worktree row missing for {worktree_id}")
    repository = db.get(Repository, wt.repository_id)
    if repository is None:
        raise RuntimeError(f"repository row missing for worktree {worktree_id}")

    venv = venv_path_for(repository.id, wt.task_id, manager.cache_dir)
    timeout = get_settings().install_timeout_seconds

    install_command = (repository.install_command or "").strip()
    if not install_command:
        # Nothing to provision — clean skip, no venv created, no subprocess.
        # The suite is expected to run on the worker environment in this case.
        logger.info(
            "Task %s has no install_command — skipping provisioning", wt.task_id
        )
        return ProvisionResult(installed=False, skipped=True)

    if venv.exists() and not venv.is_dir():
        # Stale leftover (e.g. a crash mid-provision wrote a file): reset.
        shutil.rmtree(venv, ignore_errors=True)

    if not venv.is_dir():
        started = time.perf_counter()
        exit_code, output, timed_out, duration_ms, error = _run_subprocess(
            [sys.executable, "-m", "venv", str(venv)],
            cwd=path,
            timeout_seconds=timeout,
        )
        if error is not None:
            logger.error("venv creation failed for task %s: %s", wt.task_id, error)
            return ProvisionResult(
                installed=False, duration_ms=duration_ms, error=error
            )
        if exit_code != 0:
            logger.error(
                "venv creation failed for task %s (exit %s): %s",
                wt.task_id,
                exit_code,
                output[:500],
            )
            return ProvisionResult(
                installed=False,
                exit_code=exit_code,
                output=output,
                timed_out=timed_out,
                duration_ms=duration_ms,
            )

    # Re-validate the STORED value (not agent input — there is no agent
    # input): a legacy/mis-stored command must never reach pip.
    tokens = validate_install_command(install_command)

    # First token is the validated binary name; run it as the venv's own
    # binary (argv[0] rewritten), never PATH — deterministic, no lookup.
    pip_bin = venv / "bin" / "pip"
    argv = [str(pip_bin), *tokens[1:]]

    exit_code, output, timed_out, duration_ms, error = _run_subprocess(
        argv, cwd=path, timeout_seconds=timeout
    )
    if error is not None:
        return ProvisionResult(
            installed=False, duration_ms=duration_ms, error=error
        )
    if exit_code != 0:
        logger.error(
            "install command failed for task %s (exit %s): %s",
            wt.task_id,
            exit_code,
            output[:500],
        )
        return ProvisionResult(
            installed=False,
            exit_code=exit_code,
            output=output,
            timed_out=timed_out,
            duration_ms=duration_ms,
        )

    logger.info("Dependencies installed for task %s in %dms", wt.task_id, duration_ms)
    return ProvisionResult(
        installed=True,
        exit_code=0,
        output=output,
        duration_ms=duration_ms,
    )