"""Periodic stale-CREATED sweep — healing a lost enqueue after commit.

POST /tasks commits the task row and THEN enqueues the advance_task job.
If that enqueue fails or is lost while workers are already running, the
startup sweep never sees it (no future startup happens): the task would sit
in CREATED forever. The cron sweep re-enqueues such tasks past a staleness
threshold, bounded by ``enqueue_attempts`` — exhausting the cap escalates
to FAILED(enqueue_lost) with an audit entry instead of looping forever.

These tests drive ``sweep_stale_created_once`` directly against SQLite with
a recording redis stub — a lost enqueue is exactly "row exists, job never
created", so the stub IS the ground truth for what the queue received.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import select

from app.models import AuditLog, ExecutionEvent, Repository, Task
from app.worker.queue import JOB_ADVANCE_TASK, advance_job_id
from app.worker.worker import (
    ENQUEUE_LOST_REASON,
    SWEEP_ACTOR,
    WorkerSettings,
    _sweep_stale_created,
    sweep_stale_created_once,
)


class FakeRedis:
    """Records enqueued jobs — "what actually reached the queue"."""

    def __init__(self) -> None:
        self.enqueued: list[tuple] = []

    async def enqueue_job(self, function: str, *args, **kwargs) -> None:
        self.enqueued.append((function, *(str(a) for a in args), kwargs.get("_job_id")))


@pytest.fixture()
def fake_redis() -> FakeRedis:
    return FakeRedis()


def _make_stale_task(db_session, *, status: str = "CREATED") -> Task:
    # repositories.url is UNIQUE — suffix a uuid so multi-task tests don't
    # collide on the fixture row.
    url = f"https://github.com/org/{uuid.uuid4().hex}.git"
    repo = Repository(url=url, default_branch="main")
    db_session.add(repo)
    db_session.flush()
    task = Task(objective="fix a bug", repository_id=repo.id, status=status)
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)
    return task


async def _one_pass(redis, **kwargs) -> int:
    return await sweep_stale_created_once(
        redis,
        threshold_seconds=kwargs.get("threshold_seconds", 0),
        max_attempts=kwargs.get("max_attempts", 5),
    )


def test_young_created_task_is_left_alone(db_session, fake_redis) -> None:
    """A freshly-created task must not be double-enqueued by the sweep."""
    _make_stale_task(db_session)

    acted = asyncio.run(
        sweep_stale_created_once(fake_redis, threshold_seconds=3600, max_attempts=5)
    )

    assert acted == 0
    assert fake_redis.enqueued == []
    task = db_session.scalar(select(Task))
    assert task.status == "CREATED"
    assert task.enqueue_attempts == 0


def test_stale_created_task_is_recovered(db_session, fake_redis) -> None:
    """Lost enqueue: row exists, no job ever ran — the sweep re-enqueues it."""
    task = _make_stale_task(db_session)

    acted = asyncio.run(_one_pass(fake_redis))

    assert acted == 1
    assert fake_redis.enqueued == [
        (JOB_ADVANCE_TASK, str(task.id), advance_job_id(task.id, "CREATED"))
    ]
    db_session.expire_all()
    task = db_session.get(Task, task.id)
    # Still CREATED — recovery is a re-enqueue, not a transition. The next
    # real worker pickup advances it through the normal pipeline.
    assert task.status == "CREATED"
    assert task.enqueue_attempts == 1


def test_recovery_is_bounded_and_escalates_to_failed(db_session, fake_redis) -> None:
    """Attempts at the cap: escalate to FAILED(enqueue_lost), never loop."""
    task = _make_stale_task(db_session)
    task.enqueue_attempts = 5  # max_attempts reached already
    db_session.add(task)
    db_session.commit()

    acted = asyncio.run(_one_pass(fake_redis, max_attempts=5))

    assert acted == 1
    assert fake_redis.enqueued == []  # escalation is NOT another enqueue

    db_session.expire_all()
    escalated = db_session.get(Task, task.id)
    assert escalated.status == "FAILED"

    events = db_session.scalars(
        select(ExecutionEvent).where(ExecutionEvent.task_id == task.id)
    ).all()
    assert len(events) == 1
    assert events[0].from_status == "CREATED"
    assert events[0].to_status == "FAILED"
    assert events[0].reason == ENQUEUE_LOST_REASON

    audit = db_session.scalar(select(AuditLog).where(AuditLog.task_id == task.id))
    assert audit is not None
    assert audit.actor == SWEEP_ACTOR
    assert audit.action == "task.sweep_escalated"
    assert audit.details["reason"] == ENQUEUE_LOST_REASON
    assert audit.details["attempts"] == 5
    assert audit.details["max_attempts"] == 5


def test_boundary_attempt_recovers_then_next_sweep_escalates(
    db_session, fake_redis
) -> None:
    """Attempt N-1 still recovers; only exceeding the cap escalates."""
    task = _make_stale_task(db_session)
    task.enqueue_attempts = 4  # one recovery left before the cap of 5
    db_session.add(task)
    db_session.commit()

    acted = asyncio.run(_one_pass(fake_redis, max_attempts=5))
    assert acted == 1
    assert len(fake_redis.enqueued) == 1

    db_session.expire_all()
    task = db_session.get(Task, task.id)
    assert task.status == "CREATED"
    assert task.enqueue_attempts == 5

    # The NEXT pass crosses the cap: escalation, no further enqueue.
    acted = asyncio.run(_one_pass(fake_redis, max_attempts=5))
    assert acted == 1
    assert len(fake_redis.enqueued) == 1  # unchanged
    db_session.expire_all()
    assert db_session.get(Task, task.id).status == "FAILED"


def test_non_created_states_are_ignored(db_session, fake_redis) -> None:
    """Only stuck-CREATED tasks are sweep targets — never mid-pipeline ones."""
    planning = _make_stale_task(db_session, status="PLANNING")
    failed = _make_stale_task(db_session, status="FAILED")

    acted = asyncio.run(_one_pass(fake_redis))

    assert acted == 0
    assert fake_redis.enqueued == []
    db_session.expire_all()
    assert db_session.get(Task, planning.id).status == "PLANNING"
    assert db_session.get(Task, failed.id).status == "FAILED"


def test_cron_sweep_is_registered_on_the_worker() -> None:
    """WorkerSettings carries exactly one cron job: the stale-CREATED sweep."""
    cron_jobs = WorkerSettings.cron_jobs
    assert len(cron_jobs) == 1
    assert cron_jobs[0].coroutine is _sweep_stale_created
    # unique=True: N workers share one Redis — only one runs each tick.
    assert cron_jobs[0].unique is True
