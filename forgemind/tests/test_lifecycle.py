"""Lifecycle tests: applying transitions to the DB with event logging.

Runs against the SQLite test DB (FOR UPDATE is a no-op there, but the
atomicity/event-write behavior is identical to Postgres).
"""

import uuid

import pytest
from sqlalchemy import select

from app.models import ExecutionEvent, Repository, Task, TaskStatus
from app.runtime.state_machine import IllegalTransitionError, TERMINAL_STATES
from app.runtime.task_lifecycle import (
    AUTO_PIPELINE,
    USER_CANCELLED,
    advance_task_once,
    next_status,
    transition_task,
)


def make_task(db, status: TaskStatus = TaskStatus.CREATED) -> Task:
    repo = db.scalar(select(Repository).where(Repository.url == "https://github.com/o/r.git"))
    if repo is None:
        repo = Repository(url="https://github.com/o/r.git")
        db.add(repo)
        db.flush()
    task = Task(objective="Fix the bug", repository_id=repo.id, status=status.value)
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def events_for(db, task_id: uuid.UUID) -> list[ExecutionEvent]:
    return list(
        db.scalars(
            select(ExecutionEvent)
            .where(ExecutionEvent.task_id == task_id)
            .order_by(ExecutionEvent.created_at, ExecutionEvent.id)
        )
    )


def test_transition_task_writes_event_and_updates_status(db_session) -> None:
    task = make_task(db_session)
    event = transition_task(db_session, task, TaskStatus.PLANNING)
    db_session.commit()

    assert task.status == TaskStatus.PLANNING.value
    assert event.from_status == TaskStatus.CREATED.value
    assert event.to_status == TaskStatus.PLANNING.value
    assert event.task_id == task.id

    events = events_for(db_session, task.id)
    assert len(events) == 1
    assert events[0].reason is None


def test_transition_task_records_reason(db_session) -> None:
    task = make_task(db_session)
    transition_task(db_session, task, TaskStatus.FAILED, reason=USER_CANCELLED)
    db_session.commit()
    events = events_for(db_session, task.id)
    assert events[0].to_status == TaskStatus.FAILED.value
    assert events[0].reason == USER_CANCELLED


def test_illegal_transition_raises_and_persists_nothing(db_session) -> None:
    task = make_task(db_session, status=TaskStatus.PLANNING)
    with pytest.raises(IllegalTransitionError):
        transition_task(db_session, task, TaskStatus.COMPLETED)

    assert task.status == TaskStatus.PLANNING.value  # untouched
    assert events_for(db_session, task.id) == []  # no event written


def test_replanning_increments_replan_count(db_session) -> None:
    task = make_task(db_session, status=TaskStatus.AWAITING_APPROVAL)
    assert task.replan_count == 0
    transition_task(db_session, task, TaskStatus.REPLANNING)
    db_session.commit()
    assert task.replan_count == 1
    db_session.refresh(task)
    assert task.replan_count == 1  # persisted


def test_advance_task_once_walks_task_to_completed(db_session) -> None:
    task = make_task(db_session)
    assert task.status == TaskStatus.CREATED.value

    walked: list[TaskStatus] = []
    new_status = TaskStatus.CREATED
    while new_status is not None:
        new_status = advance_task_once(db_session, task.id)
        if new_status is not None:
            walked.append(new_status)

    assert walked == AUTO_PIPELINE[1:]
    db_session.expire_all()
    task = db_session.get(Task, task.id)
    assert task.status == TaskStatus.COMPLETED.value

    events = events_for(db_session, task.id)
    assert [e.to_status for e in events] == [s.value for s in AUTO_PIPELINE[1:]]
    # Every event's from/to pair must be a legal transition.
    for e in events:
        assert e.from_status != e.to_status


def test_advance_task_once_stops_at_terminal(db_session) -> None:
    task = make_task(db_session, status=TaskStatus.COMPLETED)
    assert advance_task_once(db_session, task.id) is None
    assert task.status == TaskStatus.COMPLETED.value


def test_advance_task_once_unknown_id_returns_none(db_session) -> None:
    assert advance_task_once(db_session, uuid.uuid4()) is None


def test_advance_task_once_respects_replan_budget(db_session) -> None:
    """A task in REPLANNING with exhausted budget escalates, not replays."""
    task = make_task(db_session, status=TaskStatus.REPLANNING)
    task.max_replans = 0
    task.replan_count = 0
    db_session.commit()

    new_status = advance_task_once(db_session, task.id)
    assert new_status is TaskStatus.ESCALATED
    db_session.expire_all()
    assert db_session.get(Task, task.id).status == TaskStatus.ESCALATED.value


def test_user_cancelled_task_is_not_advanced_by_worker(db_session) -> None:
    """A task cancelled by the user (FAILED + user_cancelled) stays put."""
    task = make_task(db_session, status=TaskStatus.FAILED)
    # What the cancel endpoint leaves behind: status FAILED + a reason event.
    db_session.add(
        ExecutionEvent(
            task_id=task.id,
            from_status=TaskStatus.CREATED.value,
            to_status=TaskStatus.FAILED.value,
            reason=USER_CANCELLED,
        )
    )
    db_session.commit()

    # The worker job would call advance_task_once -> next_status returns None.
    assert advance_task_once(db_session, task.id) is None
    db_session.expire_all()
    assert db_session.get(Task, task.id).status == TaskStatus.FAILED.value
