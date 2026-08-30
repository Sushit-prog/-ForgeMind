"""Worker job-timeout ceiling + orphan-recovery tests (hermetic).

Attempt #5 orphaned a RESEARCHING task: arq's hardcoded 300s default
job_timeout killed the job mid-flight while bounded retries + fallback hops
legitimately needed longer, and nothing converted the kill into a task-row
update — the row sat stuck forever. The ceiling is now configurable
(WORKER_JOB_TIMEOUT_SECONDS, default 900) and the worker enforces an INNER
deadline slightly below it that converts hangs into explicit
FAILED(job_timeout), routing through the standard recovery path.
"""

from __future__ import annotations

import importlib
import uuid


from app.config import get_settings
from app.models import ExecutionEvent, TaskStatus


# --- configurable ceiling ----------------------------------------------------


def test_default_ceiling_is_900() -> None:
    assert get_settings().worker_job_timeout_seconds == 900
    from app.worker.worker import WorkerSettings

    assert WorkerSettings.job_timeout == 900


def test_env_override_rewires_ceiling(monkeypatch) -> None:
    monkeypatch.setenv("WORKER_JOB_TIMEOUT_SECONDS", "1234")
    get_settings.cache_clear()
    import app.worker.worker as worker_module

    try:
        reloaded = importlib.reload(worker_module)
        assert reloaded.WorkerSettings.job_timeout == 1234
    finally:
        # delenv BEFORE the restore-reload, or the reload would re-read the
        # overridden value (cache_clear alone cannot help while env persists).
        monkeypatch.delenv("WORKER_JOB_TIMEOUT_SECONDS", raising=False)
        get_settings.cache_clear()
        importlib.reload(worker_module)  # restore defaults for later tests

    assert worker_module.WorkerSettings.job_timeout == 900


def test_startup_logs_effective_ceiling(monkeypatch, caplog) -> None:
    """The startup hook must surface the effective value (auditability)."""
    import asyncio
    import logging

    from app.worker.worker import _on_startup

    monkeypatch.setenv("WORKER_SWEEP_ENABLED", "false")
    get_settings.cache_clear()
    try:
        with caplog.at_level(logging.INFO, logger="app.worker.worker"):
            asyncio.run(_on_startup({}))
    finally:
        get_settings.cache_clear()

    assert any(
        "job timeout ceiling" in r.getMessage() and "900" in r.getMessage()
        for r in caplog.records
    )


# --- orphan conversion -------------------------------------------------------


def _latest_event(db_session, task_id):
    return (
        db_session.query(ExecutionEvent)
        .filter(ExecutionEvent.task_id == task_id)
        .order_by(ExecutionEvent.created_at.desc())
        .first()
    )


def test_orphaned_researching_becomes_failed_job_timeout(db_session, repo_task) -> None:
    from app.runtime.task_lifecycle import next_status
    from app.worker.jobs.advance_task import (
        JOB_TIMEOUT_REASON,
        _mark_job_timeout_failure,
    )

    repo, task = repo_task
    task.status = TaskStatus.RESEARCHING.value
    db_session.commit()

    out = _mark_job_timeout_failure(db_session, task.id)

    db_session.refresh(task)
    assert out is TaskStatus.FAILED
    assert task.status == TaskStatus.FAILED.value
    event = _latest_event(db_session, task.id)
    assert event.to_status == TaskStatus.FAILED.value
    assert event.reason == JOB_TIMEOUT_REASON
    # The whole point: FAILED(job_timeout) routes through recovery, unlike
    # user_cancelled which stays terminal.
    assert (
        next_status(
            TaskStatus.FAILED,
            replan_count=0,
            max_replans=None,
            last_reason=JOB_TIMEOUT_REASON,
        )
        is TaskStatus.RECOVERING
    )


def test_already_failed_row_untouched(db_session, repo_task) -> None:
    from app.worker.jobs.advance_task import _mark_job_timeout_failure

    repo, task = repo_task
    task.status = TaskStatus.FAILED.value
    db_session.commit()
    before = (
        db_session.query(ExecutionEvent)
        .filter(ExecutionEvent.task_id == task.id)
        .count()
    )

    out = _mark_job_timeout_failure(db_session, task.id)

    assert out is None
    db_session.refresh(task)
    assert task.status == TaskStatus.FAILED.value
    after = (
        db_session.query(ExecutionEvent)
        .filter(ExecutionEvent.task_id == task.id)
        .count()
    )
    assert after == before, "no new event may be recorded for a skipped row"


def test_terminal_row_untouched(db_session, repo_task) -> None:
    from app.worker.jobs.advance_task import _mark_job_timeout_failure

    repo, task = repo_task
    task.status = TaskStatus.COMPLETED.value
    db_session.commit()

    out = _mark_job_timeout_failure(db_session, uuid.UUID(str(task.id)))

    db_session.refresh(task)
    assert out is None
    assert task.status == TaskStatus.COMPLETED.value
