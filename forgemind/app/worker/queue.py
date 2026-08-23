"""arq queue plumbing: Redis settings + pool + enqueue helper.

The pool is created lazily and reused for the process lifetime. The URL
comes from the same env-driven config as everything else (no new secret
surface — Phase 1 security posture).
"""

from __future__ import annotations

import logging
import uuid

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from app.config import get_settings

logger = logging.getLogger(__name__)

_pool: ArqRedis | None = None

JOB_ADVANCE_TASK = "advance_task"


def advance_job_id(task_id: str | uuid.UUID, target_status: str | None = None) -> str:
    """Deterministic arq job id for one stage of a task's advance job.

    arq dedupes on this id: while a job with the same id is queued OR running
    (its ``job_key`` survives until the job finishes), a second
    ``enqueue_job(_job_id=same)`` returns None instead of inserting a
    duplicate. Ids are therefore suffixed with the TARGET STATUS:

    - POST /tasks and the recovery sweeps both target ``PLANNING``/the
      task's current status — racing duplicates collapse into one job,
      which kills the duplicate-delivery deadlock at the source;
    - the worker's self-chain enqueues ``<next_status>`` AFTER committing
      that transition, so the id always differs from the finishing job's
      own id and the chain can never dedupe itself away.

    Pairs with ``WorkerSettings.keep_result = 0``: without it, the result
    key written under the same id on completion would block legitimate
    reuse of that stage's id for keep_result's TTL.
    """
    if target_status:
        return f"advance:{task_id}:{target_status}"
    return f"advance:{task_id}"


def get_redis_settings() -> RedisSettings:
    """arq RedisSettings derived from ``settings.redis_url``."""
    return RedisSettings.from_dsn(get_settings().redis_url)


async def get_pool() -> ArqRedis:
    """Lazily-created, process-wide arq pool."""
    global _pool
    if _pool is None:
        _pool = await create_pool(get_redis_settings())
    return _pool


async def enqueue_advance_task(
    task_id: uuid.UUID, target_status: str | None = None
) -> None:
    """Push an ``advance_task`` job for ``task_id`` onto the arq queue.

    No-op when the queue is disabled (hermetic tests); the worker's startup
    sweep would collect the task later if a queue existed.
    """
    if not get_settings().queue_enabled:
        return
    pool = await get_pool()
    await pool.enqueue_job(
        JOB_ADVANCE_TASK,
        str(task_id),
        _job_id=advance_job_id(task_id, target_status),
    )
    logger.debug("Enqueued %s for task %s", JOB_ADVANCE_TASK, task_id)
