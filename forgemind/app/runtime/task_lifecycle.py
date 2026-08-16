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

    This is the STUB driver (Phase 2): no agent work. The worker uses
    ``advance_task_with_agents`` instead, which routes PLANNING and
    RESEARCHING through the real Planning/Research agents.
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


async def advance_task_with_agents(
    db: Session,
    task_id: uuid.UUID,
    planner=None,
    researcher=None,
) -> TaskStatus | None:
    """Worker unit of work (Phases 5/6): the LLM states run the real agents.

    - PLANNING: run ``planner.run``; RESEARCHING: run ``researcher.run``.
      Each persists its artifact, and the transition to the next state fires
      only after persistence. On failure the task goes FAILED (never
      ESCALATED — reserved for replan-budget exhaustion), with the reason on
      the execution event.
    - every other state: identical to the stub ``advance_task_once``.

    ``planner``/``researcher`` may be None (no provider configured) — the
    task then fails cleanly instead of hanging.
    """
    task = db.execute(
        select(Task).where(Task.id == task_id).with_for_update()
    ).scalar_one_or_none()
    if task is None:
        logger.warning("advance_task_with_agents: task %s not found", task_id)
        return None

    current = TaskStatus(task.status)
    if current is TaskStatus.PLANNING:
        return await _run_planning(db, task, planner)
    if current is TaskStatus.RESEARCHING:
        return await _run_researching(db, task, researcher)
    return advance_task_once(db, task_id)


def _cas_transition(
    db: Session,
    task_id: uuid.UUID,
    expected: TaskStatus,
    target: TaskStatus,
    reason: str | None,
) -> TaskStatus | None:
    """Re-lock the task row and transition ONLY if still in ``expected``.

    Agent runs commit internally (plan/artifact persistence), which RELEASES
    the FOR UPDATE lock held by ``advance_task_with_agents``. A second worker
    can then read stale state and try to apply the same transition. This
    compare-and-swap re-acquires the lock and refuses to transition if
    another worker already advanced the task — the Section-D guarantee that
    one step fires exactly once, even when two workers run the same agent
    step concurrently.
    """
    # populate_existing: the session's identity map may already hold the
    # Task (read at job start); FOR UPDATE alone does not refresh it, so
    # without this the check would compare against STALE in-memory state.
    locked = db.execute(
        select(Task)
        .where(Task.id == task_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    ).scalar_one_or_none()
    if locked is None:
        return None
    if TaskStatus(locked.status) is not expected:
        logger.info(
            "Task %s already at %s (expected %s) — skipping %s transition",
            task_id, locked.status, expected.value, target.value,
        )
        return TaskStatus(locked.status)
    transition_task(db, locked, target, reason=reason)
    db.commit()
    return target


async def _run_planning(db: Session, task: Task, planner) -> TaskStatus:
    """PLANNING: real planner or a clean FAILED (no provider)."""
    if planner is None:
        logger.error("Task %s in PLANNING but no LLM provider configured", task.id)
        return _cas_transition(
            db, task.id, TaskStatus.PLANNING, TaskStatus.FAILED, "no_llm_provider"
        )

    from app.agents.planner.schema import PlanValidationError
    from app.llm.errors import LLMMalformedOutputError, LLMTimeoutError
    from app.tools.base import ExecutionContext

    ctx = ExecutionContext(task_id=task.id, agent_type="planner", db=db)
    try:
        await planner.run(task, ctx)
    except PlanValidationError as exc:
        logger.error("Task %s plan invalid after retry: %s", task.id, exc)
        return _cas_transition(
            db, task.id, TaskStatus.PLANNING, TaskStatus.FAILED, "plan_validation_failed"
        )
    except (LLMTimeoutError, LLMMalformedOutputError) as exc:
        logger.error("Task %s planner LLM error: %s", task.id, exc)
        return _cas_transition(
            db, task.id, TaskStatus.PLANNING, TaskStatus.FAILED, "llm_error"
        )
    except Exception:  # noqa: BLE001 — any planner failure fails the task, never hangs
        logger.exception("Task %s planner crashed", task.id)
        return _cas_transition(
            db, task.id, TaskStatus.PLANNING, TaskStatus.FAILED, "planning_error"
        )
    logger.info("Task %s -> RESEARCHING (real plan persisted)", task.id)
    return _cas_transition(
        db, task.id, TaskStatus.PLANNING, TaskStatus.RESEARCHING, "plan_persisted"
    )


async def _run_researching(db: Session, task: Task, researcher) -> TaskStatus:
    """RESEARCHING: real research agent or a clean FAILED (no agent)."""
    if researcher is None:
        logger.error("Task %s in RESEARCHING but no research agent", task.id)
        return _cas_transition(
            db, task.id, TaskStatus.RESEARCHING, TaskStatus.FAILED, "no_research_agent"
        )

    from app.agents.researcher.agent import ResearchError
    from app.llm.errors import LLMMalformedOutputError, LLMTimeoutError
    from app.models import Plan as PlanRow
    from app.models import PlanStep as PlanStepRow
    from app.tools.base import ExecutionContext

    plan = db.scalar(
        select(PlanRow).where(
            PlanRow.task_id == task.id, PlanRow.status == "ACTIVE"
        ).order_by(PlanRow.created_at.desc()).limit(1)
    )
    research_step = None
    if plan is not None:
        research_step = db.scalar(
            select(PlanStepRow).where(
                PlanStepRow.plan_id == plan.id, PlanStepRow.step_type == "research"
            ).order_by(PlanStepRow.sequence).limit(1)
        )
    if research_step is None:
        logger.error("Task %s in RESEARCHING but its active plan has no research step", task.id)
        return _cas_transition(
            db, task.id, TaskStatus.RESEARCHING, TaskStatus.FAILED, "no_research_step"
        )

    ctx = ExecutionContext(task_id=task.id, agent_type="researcher", db=db)
    try:
        await researcher.run(task, research_step, ctx)
    except ResearchError as exc:
        logger.error("Task %s research failed: %s", task.id, exc)
        return _cas_transition(
            db, task.id, TaskStatus.RESEARCHING, TaskStatus.FAILED, "research_failed"
        )
    except (LLMTimeoutError, LLMMalformedOutputError) as exc:
        logger.error("Task %s research LLM error: %s", task.id, exc)
        return _cas_transition(
            db, task.id, TaskStatus.RESEARCHING, TaskStatus.FAILED, "research_llm_error"
        )
    except Exception:  # noqa: BLE001 — never hang, never silently continue
        logger.exception("Task %s researcher crashed", task.id)
        return _cas_transition(
            db, task.id, TaskStatus.RESEARCHING, TaskStatus.FAILED, "research_error"
        )
    logger.info("Task %s -> IMPLEMENTING (research artifact persisted)", task.id)
    return _cas_transition(
        db, task.id, TaskStatus.RESEARCHING, TaskStatus.IMPLEMENTING, "artifact_persisted"
    )
