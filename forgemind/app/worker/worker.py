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
"""

from __future__ import annotations

import asyncio
import logging

from arq import Worker
from sqlalchemy import select

from app.config import get_settings
from app.database.session import SessionLocal
from app.logging import setup_logging
from app.models import Task
from app.runtime.state_machine import TERMINAL_STATES
from app.worker.jobs.advance_task import advance_task
from app.worker.queue import JOB_ADVANCE_TASK, get_redis_settings

logger = logging.getLogger(__name__)


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


async def _on_startup(ctx: dict) -> None:
    await _sweep_pending_tasks(ctx)


async def _on_shutdown(ctx: dict) -> None:  # noqa: ARG001
    logger.info("Worker shutting down")


WORKER_FUNCTIONS = [advance_task]
# Generous retries: SQLite (tests) can emit transient "database is locked"
# under concurrent writers; the row lock serializes on Postgres.
MAX_TRIES = 10


class WorkerSettings:
    """arq WorkerSettings — consumed by `arq app.worker.worker.WorkerSettings`."""

    functions = WORKER_FUNCTIONS
    on_startup = _on_startup
    on_shutdown = _on_shutdown
    redis_settings = get_redis_settings()
    max_tries = MAX_TRIES


if __name__ == "__main__":
    setup_logging()
    # Direct construction (arq's Worker takes `functions` positionally; the
    # WorkerSettings class above is for the `arq` CLI).
    asyncio.run(
        Worker(
            WORKER_FUNCTIONS,
            redis_settings=get_redis_settings(),
            on_startup=_on_startup,
            on_shutdown=_on_shutdown,
            max_tries=MAX_TRIES,
        ).run()
    )
