"""E2E regression — duplicate delivery racing an IN-FLIGHT job (the deadlock).

The Phase-9 deadlock: duplicate ``advance_task`` deliveries — a second app
enqueue AND the startup sweep both re-delivering a step a worker was already
executing — raced the task row. The fix was two-sided:
  (1) deterministic, stage-suffixed job ids (``advance_job_id(task_id,
      status)``) so arq collapses any enqueue whose id matches a job that is
      QUEUED or RUNNING (``job_key`` survives until the job finishes);
  (2) ``keep_result=0`` so no stale result key ever blocks legitimate future
      reuse of the same stage id.

``tests_e2e/test_job_dedupe.py`` proves the WHILE-QUEUED case only (two
enqueues, no worker present). This test proves the WHILE-RUNNING case on the
real stack: a windowed worker is observed mid-job (its ``arq:job:`` key is
present in Redis), and BOTH historical sources are then re-played against
that in-flight id —
    * an app-level duplicate enqueue (the ``enqueue_advance_task`` path that
      POST /tasks uses), and
    * the startup sweep's re-enqueue (``_sweep_pending_tasks``).
Both must be swallowed by arq (return None / insert nothing), and the task
must then walk the EXACT pipeline to COMPLETED — one transition per step, no
``IllegalTransitionError``, no double-processing — proving the dedup holds
under the real trigger condition, not just under two clean sequential runs.
"""

from __future__ import annotations

import asyncio
import uuid

from arq import create_pool
from sqlalchemy import select

from app.models import ExecutionEvent, Task, TaskStatus
from app.runtime.task_lifecycle import AUTO_PIPELINE
from app.worker.worker import _sweep_pending_tasks
from tests_e2e.conftest import approve_task, spawn_worker

EXPECTED_STATUSES = [s.value for s in AUTO_PIPELINE][1:]

_JOB_KEY_PREFIX = "arq:job:advance:"


async def _wait_for_running_job(pool, task_id: str, timeout: float = 20.0) -> list:
    """Poll Redis until a job for ``task_id`` is mid-execution (job_key set).

    arq holds ``arq:job:<id>`` only while a job is queued-as-retry or
    RUNNING — presence means a worker has pulled it and the dedup window is
    genuinely open. Returns the observed keys (decoded).
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        keys = []
        async for raw in pool.scan_iter(match=f"{_JOB_KEY_PREFIX}{task_id}:*"):
            keys.append(raw.decode())
        if keys:
            return keys
        if asyncio.get_event_loop().time() >= deadline:
            return []
        await asyncio.sleep(0.05)


async def _inject_raced_deliveries(pool, task_id: str, db_session) -> None:
    """Re-play the two historical duplicate sources against the in-flight job.

    Observed mid-flight, so the running job's id encodes the status it is
    ABOUT to transition out of: ``advance:{task_id}:{FROM}``. Both the
    app-level duplicate enqueue and the startup sweep target that same id —
    exactly the race that used to deadlock. Either one breaking the dedup is
    asserted loudly; the pipeline's later completion is the behavioral proof.
    """
    import app.worker.queue as q

    keys = await _wait_for_running_job(pool, task_id)
    assert keys, "worker never reached an in-flight job window"

    observed_live = keys[0]
    from_status = observed_live.split(":")[-1]

    # Sample the DB in the same window: the observed status must equal the
    # task's committed status, i.e. the job holding it is at that step.
    db_session.expire_all()
    current = TaskStatus(db_session.get(Task, uuid.UUID(task_id)).status)
    assert current.value == from_status, (
        f"observed in-flight status {from_status} != DB {current.value} — "
        "the observed job could not have been holding this step"
    )

    # 1) App-level duplicate: the exact path POST /tasks uses.
    dup = await pool.enqueue_job(
        q.JOB_ADVANCE_TASK, task_id, _job_id=q.advance_job_id(task_id, from_status)
    )
    assert dup is None, (
        f"duplicate enqueue of in-flight id advance:{task_id}:{from_status} "
        "was NOT deduped — this is the duplicate-delivery bug"
    )

    # 2) The same enqueue_advance_task wrapper POST /tasks runs.
    await q.enqueue_advance_task(uuid.UUID(task_id), target_status=from_status)

    # 3) Startup sweep re-enqueuing every non-terminal task — the second
    # half of the historical deadlock.
    await _sweep_pending_tasks({"redis": pool})


def test_inflight_duplicate_and_sweep_reenqueue_are_swallowed(
    client, db_session, source_repo
) -> None:
    # Windowed worker: every step sleeps at job start, so the running
    # job_key stays alive long enough to re-play the raced deliveries.
    proc = spawn_worker({"FORGEMIND_STEP_DELAY_MS": "2000"})
    pool = None
    try:
        created = client.post(
            "/tasks",
            json={
                "objective": "fix a bug",
                # A real clonable repo: RESEARCHING (Phase 6) needs one.
                "repository_url": "file:///" + str(source_repo).replace("\\", "/"),
                "fork_url": "https://github.com/fork-owner/forgemind-e2e-fork",
            },
        ).json()
        task_id = str(created["id"])

        pool = asyncio.run(create_pool(get_redis_settings()))
        asyncio.run(_inject_raced_deliveries(pool, task_id, db_session))

        # The worker kept the ORIGINAL job: the pipeline must still complete
        # with one exact transition per step — no double-processing.
        approve_task(client, task_id, timeout=150)

        events = db_session.scalars(
            select(ExecutionEvent)
            .where(ExecutionEvent.task_id == uuid.UUID(task_id))
            .order_by(ExecutionEvent.created_at, ExecutionEvent.id)
        ).all()
        assert [e.to_status for e in events] == EXPECTED_STATUSES, (
            f"task {task_id}: event trail was not the exact pipeline after "
            f"in-flight duplicate + sweep re-enqueue "
            f"(got {[e.to_status for e in events]})"
        )
        for e in events:
            assert e.from_status != e.to_status

        db_session.expire_all()
        task = db_session.get(Task, uuid.UUID(task_id))
        assert task.replan_count == 0
    finally:
        if pool is not None:
            asyncio.run(_flush_redis(pool))
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)


async def _flush_redis(pool) -> None:
    """Best-effort queue flush so a stray injected job can't leak across tests."""
    try:
        await pool.flushdb()
    finally:
        await pool.aclose()