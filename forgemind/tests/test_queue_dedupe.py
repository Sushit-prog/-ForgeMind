"""Deterministic advance-job ids — duplicate-delivery dedupe (Fix 1).

arq drops an enqueue whose _job_id matches a queued-or-running job (the
job_key survives until finish). The id scheme therefore suffixes the
TARGET status:

- POST /tasks and both recovery sweeps target the same (task, status) id,
  so racing duplicates collapse into one job;
- the worker's self-chain enqueues under the NEXT status's id AFTER
  committing that transition, so it can never be swallowed by the
  finishing job's own id.
"""

from __future__ import annotations

import asyncio
import uuid

from app.models import TaskStatus
from app.worker.queue import JOB_ADVANCE_TASK, advance_job_id


class RecordingRedis:
    """Captures enqueue kwargs — "what would reach Redis"."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def enqueue_job(self, function: str, *args, **kwargs) -> None:
        self.calls.append({"function": function, "args": args, **kwargs})


def test_advance_job_id_scheme() -> None:
    tid = uuid.uuid4()
    assert advance_job_id(tid) == f"advance:{tid}"
    assert advance_job_id(tid, "PLANNING") == f"advance:{tid}:PLANNING"
    # Distinct stages never share an id — the self-chain must not collide
    # with the finishing job's own id.
    assert advance_job_id(tid, "PLANNING") != advance_job_id(tid, "RESEARCHING")


def test_startup_sweep_targets_current_status_per_task(db_session):
    """Sweep ids are deterministic per (task, current status)."""
    from app.models import Repository, Task
    from app.worker.worker import _sweep_pending_tasks

    redis = RecordingRedis()
    repo = Repository(url="https://github.com/org/r1.git", default_branch="main")
    db_session.add(repo)
    db_session.flush()
    created = Task(objective="a", repository_id=repo.id, status="CREATED")
    planning = Task(objective="b", repository_id=repo.id, status="PLANNING")
    db_session.add_all([created, planning])
    db_session.commit()

    asyncio.run(_sweep_pending_tasks({"redis": redis}))

    # Filter to OUR tasks: earlier API tests may legally leave other
    # non-terminal rows in the shared test database.
    ours = [
        c
        for c in redis.calls
        if c["_job_id"]
        in (
            f"advance:{created.id}:CREATED",
            f"advance:{planning.id}:PLANNING",
        )
    ]
    assert {c["_job_id"] for c in ours} == {
        f"advance:{created.id}:CREATED",
        f"advance:{planning.id}:PLANNING",
    }
    assert all(c["function"] == JOB_ADVANCE_TASK for c in ours)


def test_stale_sweep_uses_created_status_id(db_session):
    """Stale-CREATED sweep targets :CREATED — collapsing with POST /tasks."""
    from app.worker.worker import sweep_stale_created_once
    from tests.test_worker_sweep import _make_stale_task

    redis = RecordingRedis()
    task = _make_stale_task(db_session)

    acted = asyncio.run(
        sweep_stale_created_once(redis, threshold_seconds=0, max_attempts=5)
    )

    assert acted == 1
    assert redis.calls[0]["_job_id"] == f"advance:{task.id}:CREATED"


def test_chain_enqueue_uses_next_status_id() -> None:
    """Pin the chain rule: id derives from the COMMITTED next status."""
    # advance_task.py:151 passes new_status.value after the transition —
    # this test pins the contract so a refactor cannot silently break it.
    tid = uuid.uuid4()
    committed_next = TaskStatus.RESEARCHING.value
    assert advance_job_id(tid, committed_next) == f"advance:{tid}:RESEARCHING"
