"""Task lifecycle: applying transitions to the DB and driving the stub pipeline.

- ``transition_task`` applies ONE legal transition and writes the matching
  ``execution_event`` in the same transaction — a task can never be observed
  in a half-written state (Section J: crash between transitions leaves the
  last *committed* status).
- ``advance_task_once`` is the worker's unit of work: SELECT ... FOR UPDATE,
  compute the next stub transition, apply it, commit. Row-level locking
  serializes concurrent workers (Section D / edge cases).
- ``next_status`` is the *stub* pipeline driver for this milestone: no agents
  exist yet, so transitions are instant and deterministic. A future phase
  replaces it with agent-driven transition decisions.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ExecutionEvent, Task, TaskStatus
from app.models.base import utcnow
from app.runtime.state_machine import TERMINAL_STATES, state_machine

logger = logging.getLogger(__name__)

USER_CANCELLED = "user_cancelled"

# Stub happy-path pipeline (section D): the worker walks this end to end.
AUTO_PIPELINE: list[TaskStatus] = [
    TaskStatus.CREATED,
    TaskStatus.PLANNING,
    TaskStatus.RESEARCHING,
    TaskStatus.IMPLEMENTING,
    TaskStatus.TESTING,
    TaskStatus.REVIEWING,
    TaskStatus.SECURITY_REVIEW,
    TaskStatus.VERIFICATION,
    TaskStatus.PR_CREATION,
    TaskStatus.AWAITING_APPROVAL,
    TaskStatus.COMPLETED,
]


def _next_event_created_at(db: Session, task_id: uuid.UUID) -> datetime:
    """Strictly-increasing timestamp for this task's next event.

    ``datetime.utcnow`` is only clock-tick precise on some platforms (Windows
    ticks at ~15.6ms), so back-to-back transitions could share a timestamp and
    the ``order by created_at, id`` tiebreak (random UUID) would scramble the
    trail. Bump past the task's last event instead — deterministic ordering on
    every platform, no schema change.
    """
    last = db.scalar(
        select(ExecutionEvent.created_at)
        .where(ExecutionEvent.task_id == task_id)
        .order_by(ExecutionEvent.created_at.desc())
        .limit(1)
    )
    now = utcnow()
    # SQLite returns naive datetimes; Postgres returns aware ones. Compare in
    # the same space (both are UTC).
    now_naive = now.replace(tzinfo=None)
    last_naive = last.replace(tzinfo=None) if last is not None else None
    if last_naive is not None and now_naive <= last_naive:
        return last + timedelta(microseconds=1)
    return now


def transition_task(
    db: Session,
    task: Task,
    target: TaskStatus,
    *,
    reason: str | None = None,
) -> ExecutionEvent:
    """Apply ``task.status -> target`` and record the event.

    Raises ``IllegalTransitionError`` if the transition is not in the
    Section-D table. The caller owns the transaction (commit/rollback).
    """
    current = TaskStatus(task.status)
    state_machine.transition(current, target)  # deterministic guard, no LLM
    task.status = target.value
    if target is TaskStatus.REPLANNING:
        task.replan_count += 1
    event = ExecutionEvent(
        task_id=task.id,
        from_status=current.value,
        to_status=target.value,
        reason=reason,
        created_at=_next_event_created_at(db, task.id),
    )
    db.add(event)
    # Flush so a later transition in the SAME transaction sees this event's
    # timestamp (the ordering query can't see unflushed rows). The caller
    # still owns commit/rollback.
    db.flush()
    return event


def next_status(
    current: TaskStatus,
    *,
    replan_count: int,
    max_replans: int | None,
    last_reason: str | None,
) -> TaskStatus | None:
    """Compute the next stub transition for ``current`` (or None = stop).

    Mirrors Section D: happy path walks ``AUTO_PIPELINE``; failures recover
    FAILED -> RECOVERING -> REPLANNING -> RESEARCHING; exhausted replan
    budget escalates; user-cancelled failures stay put.
    """
    if current in TERMINAL_STATES:
        return None
    if current is TaskStatus.FAILED:
        # A user-cancelled task is intentionally terminal; anything else
        # (real failures, future phases) recovers through the failure path.
        return None if last_reason == USER_CANCELLED else TaskStatus.RECOVERING
    if current is TaskStatus.RECOVERING:
        return TaskStatus.REPLANNING
    if current is TaskStatus.REPLANNING:
        if max_replans is not None and replan_count >= max_replans:
            return TaskStatus.ESCALATED
        return TaskStatus.RESEARCHING
    if current is TaskStatus.DEBUGGING:
        return TaskStatus.IMPLEMENTING
    try:
        idx = AUTO_PIPELINE.index(current)
    except ValueError:
        logger.error("Stub pipeline has no next step for %s", current)
        return None
    return AUTO_PIPELINE[idx + 1] if idx + 1 < len(AUTO_PIPELINE) else None


def advance_task_once(db: Session, task_id: uuid.UUID) -> TaskStatus | None:
    """Load the task FOR UPDATE, apply the next legal transition, commit.

    Returns the new status, or None when there is nothing to do (terminal
    task, cancelled task, or unknown id). Illegal transitions raise
    ``IllegalTransitionError`` and never silently update ``tasks.status``.

    This is the STUB driver (Phase 2): PLANNING -> RESEARCHING happens
    without a real plan. The worker uses ``advance_task_with_planner``
    instead, which routes PLANNING through the real PlanningAgent.
    """
    task = db.execute(
        select(Task).where(Task.id == task_id).with_for_update()
    ).scalar_one_or_none()
    if task is None:
        logger.warning("advance_task_once: task %s not found", task_id)
        return None

    last_event = db.execute(
        select(ExecutionEvent)
        .where(ExecutionEvent.task_id == task_id)
        .order_by(ExecutionEvent.created_at.desc(), ExecutionEvent.id.desc())
        .limit(1)
    ).scalar_one_or_none()

    target = next_status(
        TaskStatus(task.status),
        replan_count=task.replan_count,
        max_replans=task.max_replans,
        last_reason=last_event.reason if last_event else None,
    )
    if target is None:
        return None

    previous = TaskStatus(task.status)
    transition_task(db, task, target)
    db.commit()
    logger.info("Task %s: %s -> %s", task_id, previous.value, target.value)
    return target


async def advance_task_with_planner(
    db: Session, task_id: uuid.UUID, planner
) -> TaskStatus | None:
    """Worker unit of work (Phase 5): PLANNING runs the real PlanningAgent.

    - PLANNING: call ``planner.run``; on success persist happened inside the
      planner and we transition PLANNING -> RESEARCHING. On failure the task
      goes FAILED (never ESCALATED — that is reserved for replan-budget
      exhaustion), with the reason recorded on the execution event.
    - every other state: identical to the stub ``advance_task_once``.

    ``planner`` may be None (no provider configured) — PLANNING then fails
    the task cleanly with ``no_llm_provider`` instead of hanging.
    """
    task = db.execute(
        select(Task).where(Task.id == task_id).with_for_update()
    ).scalar_one_or_none()
    if task is None:
        logger.warning("advance_task_with_planner: task %s not found", task_id)
        return None

    if TaskStatus(task.status) is not TaskStatus.PLANNING:
        return advance_task_once(db, task_id)

    if planner is None:
        logger.error("Task %s in PLANNING but no LLM provider configured", task_id)
        transition_task(db, task, TaskStatus.FAILED, reason="no_llm_provider")
        db.commit()
        return TaskStatus.FAILED

    from app.agents.planner.schema import PlanValidationError
    from app.llm.errors import LLMMalformedOutputError, LLMTimeoutError
    from app.tools.base import ExecutionContext

    ctx = ExecutionContext(task_id=task.id, agent_type="planner", db=db)
    previous = TaskStatus(task.status)
    try:
        await planner.run(task, ctx)
    except PlanValidationError as exc:
        logger.error("Task %s plan invalid after retry: %s", task_id, exc)
        transition_task(db, task, TaskStatus.FAILED, reason="plan_validation_failed")
    except (LLMTimeoutError, LLMMalformedOutputError) as exc:
        logger.error("Task %s planner LLM error: %s", task_id, exc)
        transition_task(db, task, TaskStatus.FAILED, reason="llm_error")
    except Exception:  # noqa: BLE001 — any planner failure fails the task, never hangs
        logger.exception("Task %s planner crashed", task_id)
        transition_task(db, task, TaskStatus.FAILED, reason="planning_error")
    else:
        # A real, persisted, schema-validated plan exists — only now may the
        # task leave PLANNING.
        transition_task(db, task, TaskStatus.RESEARCHING, reason="plan_persisted")
        logger.info("Task %s: %s -> RESEARCHING (real plan persisted)", task_id, previous.value)
    db.commit()
    return TaskStatus(task.status)
