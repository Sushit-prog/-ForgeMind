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


def get_redis_settings() -> RedisSettings:
    """arq RedisSettings derived from ``settings.redis_url``."""
    return RedisSettings.from_dsn(get_settings().redis_url)


async def get_pool() -> ArqRedis:
    """Lazily-created, process-wide arq pool."""
    global _pool
    if _pool is None:
        _pool = await create_pool(get_redis_settings())
    return _pool


async def enqueue_advance_task(task_id: uuid.UUID) -> None:
    """Push an ``advance_task`` job for ``task_id`` onto the arq queue.

    No-op when the queue is disabled (hermetic tests); the worker's startup
    sweep would collect the task later if a queue existed.
    """
    if not get_settings().queue_enabled:
        return
    pool = await get_pool()
    await pool.enqueue_job(JOB_ADVANCE_TASK, str(task_id))
    logger.debug("Enqueued %s for task %s", JOB_ADVANCE_TASK, task_id)
