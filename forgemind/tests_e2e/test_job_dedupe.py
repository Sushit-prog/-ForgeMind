"""Fix 1 proof on the real stack: duplicate advance jobs collapse to one.

arq semantics under test (connections.py:158): an enqueue whose _job_id
matches an existing job_key or result key returns None instead of inserting.
POST /tasks and the startup sweep racing for the same CREATED task must
therefore produce exactly ONE queued job.
"""

from __future__ import annotations

import asyncio
import uuid

import redis.asyncio as aioredis
from arq import create_pool

from app.config import get_settings
from app.worker.queue import (
    JOB_ADVANCE_TASK,
    advance_job_id,
    enqueue_advance_task,
    get_redis_settings,
)


def test_second_enqueue_same_id_is_dropped() -> None:
    async def scenario() -> None:
        pool = await create_pool(get_redis_settings())
        try:
            tid = f"test-{uuid.uuid4()}"
            first = await pool.enqueue_job(
                JOB_ADVANCE_TASK, "0" * 32, _job_id=advance_job_id(tid, "PLANNING")
            )
            second = await pool.enqueue_job(
                JOB_ADVANCE_TASK, "0" * 32, _job_id=advance_job_id(tid, "PLANNING")
            )
            assert first is not None, "first insert should win"
            assert second is None, "duplicate must be collapsed by arq"
            # Distinct stage ids are NOT swallowed — the self-chain stays alive.
            other = await pool.enqueue_job(
                JOB_ADVANCE_TASK, "0" * 32, _job_id=advance_job_id(tid, "RESEARCHING")
            )
            assert other is not None
        finally:
            keys = [k async for k in pool.scan_iter(match="advance:test-*")]
            if keys:
                await pool.delete(*keys)
            await pool.aclose()

    asyncio.run(scenario())


def test_enqueue_advance_task_uses_deterministic_stage_id() -> None:
    """POST /tasks path targets :PLANNING; a racing sweep targeting
    :PLANNING collapses into it — exactly one queued job."""

    async def scenario() -> None:
        task_id = uuid.uuid4()
        await enqueue_advance_task(task_id, target_status="PLANNING")
        await enqueue_advance_task(task_id, target_status="PLANNING")  # racing sweep

        r = aioredis.from_url(get_settings().redis_url)
        try:
            queued = await r.zrangebyscore("arq:queue", "-inf", "+inf")
            matching = [
                j for j in queued if j.decode().startswith(f"advance:{task_id}")
            ]
            assert len(matching) == 1, f"expected one job, saw {matching}"
        finally:
            await r.delete(advance_job_id(task_id, "PLANNING"))
            await r.aclose()

    asyncio.run(scenario())
