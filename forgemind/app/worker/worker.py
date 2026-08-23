"""arq worker entrypoint.

Run with either:

    arq app.worker.worker.WorkerSettings
    python -m app.worker.worker

On startup the worker sweeps the database and re-enqueues any task in a
non-terminal state — this is what makes crash recovery work (Section J:
discard nothing, resume from the last persisted checkpoint). The sweep is
safe under concurrency because every transition is guarded by a row lock and
the state machine, so duplicate jobs are no-ops or single steps, never
double-applications.

The startup sweep alone cannot heal an enqueue lost while workers are
ALREADY RUNNING (POST /tasks commits the row, Redis drops the job): no
future startup happens to notice it. A cron sweep therefore runs about
every minute over tasks stuck in CREATED past a staleness threshold and
re-enqueues them — bounded by ``enqueue_attempts`` so a task that keeps
losing its enqueues escalates to FAILED(enqueue_lost) with an audit entry
instead of sweeping forever.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

from arq import Worker
from arq.cron import cron
from sqlalchemy import select

from app.config import get_settings
from app.database.session import SessionLocal
from app.logging import setup_logging
from app.models import AuditLog, Task, TaskStatus
from app.models.base import utcnow
from app.runtime.state_machine import TERMINAL_STATES
from app.runtime.task_lifecycle import transition_task
from app.worker.jobs.advance_task import advance_task
from app.worker.queue import JOB_ADVANCE_TASK, get_redis_settings

logger = logging.getLogger(__name__)

ENQUEUE_LOST_REASON = "enqueue_lost"
SWEEP_ACTOR = "worker-sweep"


async def _sweep_pending_tasks(ctx: dict) -> None:
    """Re-enqueue non-terminal tasks left behind by a crashed worker."""
    if not get_settings().worker_sweep_enabled:
        return
    db = SessionLocal()
    try:
        task_ids = db.scalars(
            select(Task.id).where(
                Task.status.not_in([s.value for s in TERMINAL_STATES])
            )
        ).all()
        for task_id in task_ids:
            await ctx["redis"].enqueue_job(JOB_ADVANCE_TASK, str(task_id))
        if task_ids:
            logger.info("Sweep re-enqueued %d task(s) for recovery", len(task_ids))
    finally:
        db.close()


async def sweep_stale_created_once(
    redis,
    *,
    threshold_seconds: int,
    max_attempts: int,
) -> int:
    """One pass of the periodic stale-CREATED sweep.

    Re-enqueues CREATED tasks older than ``threshold_seconds`` whose original
    enqueue was lost after the row was committed (workers already running —
    no future startup sweep would ever see them). Each recovery increments
    ``enqueue_attempts``; once a task would exceed ``max_attempts`` it is
    instead escalated to FAILED(enqueue_lost) with an audit-log entry —
    bounded retries, never an infinite loop.

    Returns the number of tasks acted on (recovered + escalated). Duplicate
    deliveries stay safe: advance_task is a row-locked no-op on stale jobs.
    """
    db = SessionLocal()
    acted = 0
    try:
        candidates = db.scalars(
            select(Task).where(Task.status == TaskStatus.CREATED.value)
        ).all()
        # Normalize to naive-UTC for the comparison (SQLite returns naive
        # datetimes; Postgres aware ones) — same trick as
        # task_lifecycle._next_event_created_at.
        now_naive = utcnow().replace(tzinfo=None)
        cutoff = now_naive - timedelta(seconds=threshold_seconds)
        for task in candidates:
            if task.created_at.replace(tzinfo=None) > cutoff:
                continue  # too young to distinguish from a healthy CREATED task
            attempts = task.enqueue_attempts + 1
            if attempts > max_attempts:
                transition_task(db, task, TaskStatus.FAILED, reason=ENQUEUE_LOST_REASON)
                db.add(
                    AuditLog(
                        task_id=task.id,
                        actor=SWEEP_ACTOR,
                        action="task.sweep_escalated",
                        entity_type="task",
                        entity_id=str(task.id),
                        details={
                            "reason": ENQUEUE_LOST_REASON,
                            "attempts": task.enqueue_attempts,
                            "max_attempts": max_attempts,
                        },
                    )
                )
                db.commit()
                logger.error(
                    "Task %s still CREATED after %d sweep recoveries — "
                    "escalated to FAILED(%s)",
                    task.id,
                    max_attempts,
                    ENQUEUE_LOST_REASON,
                )
                acted += 1
                continue
            task.enqueue_attempts = attempts
            db.commit()
            await redis.enqueue_job(JOB_ADVANCE_TASK, str(task.id))
            logger.warning(
                "Swept stale CREATED task %s (attempt %d/%d) — re-enqueued",
                task.id,
                attempts,
                max_attempts,
            )
            acted += 1
    finally:
        db.close()
    return acted


async def _sweep_stale_created(ctx: dict) -> None:
    """Cron wrapper: settings-driven pass of the stale-CREATED sweep."""
    if not get_settings().worker_sweep_enabled:
        return
    settings = get_settings()
    await sweep_stale_created_once(
        ctx["redis"],
        threshold_seconds=settings.sweep_stale_created_seconds,
        max_attempts=settings.sweep_stale_created_max_attempts,
    )


async def _on_startup(ctx: dict) -> None:
    logger.info(
        "arq job timeout ceiling: %ss (settings.worker_job_timeout_seconds)",
        get_settings().worker_job_timeout_seconds,
    )
    await _sweep_pending_tasks(ctx)


async def _on_shutdown(ctx: dict) -> None:  # noqa: ARG001
    logger.info("Worker shutting down")


WORKER_FUNCTIONS = [advance_task]
# Generous retries: SQLite (tests) can emit transient "database is locked"
# under concurrent writers; the row lock serializes on Postgres.
MAX_TRIES = 10

# Periodic self-heal (~every minute, wall-clock second 0). unique=True keeps
# N concurrent workers from double-running the same tick.
CRON_JOBS = [cron(_sweep_stale_created, second=0, unique=True, run_at_startup=False)]


class WorkerSettings:
    """arq WorkerSettings — consumed by `arq app.worker.worker.WorkerSettings`."""

    functions = WORKER_FUNCTIONS
    cron_jobs = CRON_JOBS
    on_startup = _on_startup
    on_shutdown = _on_shutdown
    redis_settings = get_redis_settings()
    max_tries = MAX_TRIES
    job_timeout = get_settings().worker_job_timeout_seconds


if __name__ == "__main__":
    setup_logging()
    # Direct construction (arq's Worker takes `functions` positionally; the
    # WorkerSettings class above is for the `arq` CLI).
    asyncio.run(
        Worker(
            WORKER_FUNCTIONS,
            redis_settings=get_redis_settings(),
            cron_jobs=CRON_JOBS,
            on_startup=_on_startup,
            on_shutdown=_on_shutdown,
            max_tries=MAX_TRIES,
            job_timeout=get_settings().worker_job_timeout_seconds,
        ).run()
    )
