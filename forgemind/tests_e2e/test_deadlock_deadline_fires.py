"""Fix 3 proof: the inner deadline now fires DURING a blocked DB call.

Before Fix 3 this scenario hung FOREVER (root-caused with py-spy): Job B's
sync `FOR UPDATE` blocked on Job A's row lock ON the event-loop thread, so
the inner asyncio.timeout could never fire and the loop processed nothing.

Now the blocked acquire runs in a thread:
- t=0    victim's FOR UPDATE starts waiting on the holder's row lock
- t=0.5  a concurrent marker task completes — PROOF the loop stayed live
         (pre-Fix 3 this marker could not run until the lock cleared)
- t=4s   the 4s inner deadline fires -> TimeoutError branch
- t=6s   holder releases; the FAILED(job_timeout) CAS lands

Total ~6s, task FAILED(job_timeout). Before all three fixes: infinite hang,
task stuck in CREATED forever.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid

import pytest
from sqlalchemy import select

from app.models import ExecutionEvent, Repository, Task
from app.worker.jobs.advance_task import advance_task


class NoopRedis:
    def __init__(self) -> None:
        self.enqueued: list[tuple] = []

    async def enqueue_job(self, function: str, *args, **kwargs) -> None:
        self.enqueued.append((function, *(str(a) for a in args)))


@pytest.fixture()
def short_deadline(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("WORKER_INNER_DEADLINE_SECONDS", "4")
    get_settings.cache_clear()
    try:
        yield 4
    finally:
        monkeypatch.delenv("WORKER_INNER_DEADLINE_SECONDS", raising=False)
        get_settings.cache_clear()


def test_inner_deadline_fires_while_db_call_blocks(db_session, short_deadline) -> None:
    repo = Repository(url=f"https://github.com/org/{uuid.uuid4().hex}.git")
    db_session.add(repo)
    db_session.flush()
    task = Task(objective="deadlock proof", repository_id=repo.id, status="CREATED")
    db_session.add(task)
    db_session.commit()
    db_session.close()  # victim gets its own session inside advance_task

    SessionLocal_holding_lock(task.id, hold_seconds=6)

    ctx = {"redis": NoopRedis()}
    started = time.monotonic()

    async def scenario():
        marker_done_at: list[float] = []

        async def marker():
            await asyncio.sleep(0.5)
            marker_done_at.append(time.monotonic() - started)

        marker_task = asyncio.create_task(marker())
        status = await advance_task(ctx, str(task.id))
        await marker_task
        return status, marker_done_at

    status, marker_done_at = asyncio.run(scenario())
    elapsed = time.monotonic() - started

    # THE regression assertions:
    assert marker_done_at, "event loop froze — marker never ran"
    assert marker_done_at[0] < 1.5, (
        f"marker completed at {marker_done_at[0]:.2f}s — event loop was blocked "
        "during the victim's DB call (Fix 3 regressed)"
    )
    assert status is None or True  # job returns; task state checked below
    assert elapsed < 20, f"run took {elapsed:.1f}s — deadline did not bound it"

    db_session.expire_all()
    final = db_session.get(Task, task.id)
    assert final.status == "FAILED"

    events = db_session.scalars(
        select(ExecutionEvent).where(ExecutionEvent.task_id == task.id)
    ).all()
    reasons = [e.reason for e in events]
    assert "job_timeout" in reasons, f"expected job_timeout, got {reasons}"


def SessionLocal_holding_lock(task_id, *, hold_seconds: float):
    """Simulate 'Job A mid-planner-await': hold the task row lock, release."""
    from app.database.session import SessionLocal

    session = SessionLocal()

    def hold():
        try:
            session.execute(select(Task).where(Task.id == task_id).with_for_update())
            time.sleep(hold_seconds)
        finally:
            session.rollback()
            session.close()

    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    return thread
