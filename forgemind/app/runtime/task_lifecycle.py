"""Task lifecycle: applying transitions to the DB and driving the pipeline.

- ``transition_task`` applies ONE legal transition and writes the matching
  ``execution_event`` in the same transaction — a task can never be observed
  in a half-written state (Section J: crash between transitions leaves the
  last *committed* status).
- ``advance_task_once`` is the worker's unit of work: SELECT ... FOR UPDATE,
  compute the next stub transition, apply it, commit. Row-level locking
  serializes concurrent workers (Section D / edge cases).
- ``advance_task_with_agents`` routes PLANNING/RESEARCHING/IMPLEMENTING/
  TESTING/DEBUGGING through the real Planning/Research/Developer/Test/
  Debugger agents; every other state falls through to the stub driver.
- ``next_status`` remains the *stub* decision for the remaining non-agent
  states only. TESTING's pass/fail branching and DEBUGGING's
  fixable/flaky/unfixable branching are REAL (Phase 8): TESTING branches
  passed -> REVIEWING / failed|error -> DEBUGGING; DEBUGGING branches
  flaky -> REVIEWING / unfixable -> FAILED / fixable -> IMPLEMENTING with
  the replan budget enforced at the transition.
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
    developer=None,
    tester=None,
    debugger=None,
    reviewer=None,
    security=None,
) -> TaskStatus | None:
    """Worker unit of work (Phases 5-9): the real states run the real agents.

    - PLANNING -> ``planner``; RESEARCHING -> ``researcher``; IMPLEMENTING
      -> ``developer`` (real commit + grounded summary); TESTING -> the
      deterministic ``tester`` (real subprocess run of the configured test
      command, Section 41: no LLM judgment); DEBUGGING -> ``debugger``
      (read-only investigation + classification, with the flakiness re-run);
      REVIEWING -> ``reviewer`` (diff + test result only — independent of
      the developer's summary); SECURITY_REVIEW -> ``security`` (diff only,
      blind to the Reviewer's verdict); VERIFICATION -> plain code (Phase 9
      staleness check, no LLM at all). Each persists its artifact and the
      transition fires only after persistence. On failure the task goes
      FAILED (never ESCALATED — reserved for replan-budget exhaustion),
      with the reason on the event.
    - TESTING branches for real: passed -> REVIEWING; failed/error ->
      DEBUGGING. DEBUGGING branches for real: flaky -> REVIEWING;
      unfixable -> FAILED with the category; fixable -> IMPLEMENTING.
      REVIEWING branches for real: APPROVE -> SECURITY_REVIEW;
      REQUEST_CHANGES/REJECT -> IMPLEMENTING. SECURITY_REVIEW branches:
      PASS -> VERIFICATION; FAIL -> IMPLEMENTING. VERIFICATION: the
      reviewed commit must still be worktree HEAD and the last test run
      must still be passed — otherwise back to TESTING (stale review). All
      replans draw from the ONE shared max_replans budget at the transition
      (exhausted -> ESCALATED).
    - every other state: identical to the stub ``advance_task_once``.

    Any agent may be None (no provider configured) — the task then fails
    cleanly instead of hanging.
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
    if current is TaskStatus.IMPLEMENTING:
        return await _run_implementing(db, task, developer)
    if current is TaskStatus.TESTING:
        return await _run_testing(db, task, tester)
    if current is TaskStatus.DEBUGGING:
        return await _run_debugging(db, task, debugger)
    if current is TaskStatus.REVIEWING:
        return await _run_reviewing(db, task, reviewer)
    if current is TaskStatus.SECURITY_REVIEW:
        return await _run_security_review(db, task, security)
    if current is TaskStatus.VERIFICATION:
        return await _run_verification(db, task)
    return advance_task_once(db, task_id)


def _latest_fix_instruction(db: Session, task_id: uuid.UUID) -> str | None:
    """The most recent fix instruction from ANY replan source (Phase 9).

    Debugger classifications, Reviewer verdicts, and Security findings all
    route back to IMPLEMENTING; the developer must receive the LATEST one,
    clearly labeled with its source — never a merged/ambiguous instruction
    (e.g. Reviewer approved but Security then failed: the developer must
    see that it is Security's finding, not a mix of both). One shared replan
    budget (``max_replans``) governs all three sources (Section 42): the
    budget is counted on the task row, not per source.
    """
    from app.models import FailureClassification as ClassificationRow
    from app.models import ReviewResult as ReviewRow
    from app.models import SecurityResult as SecurityRow

    candidates: list[tuple] = []  # (created_at, source, text)
    classification = db.scalar(
        select(ClassificationRow).where(ClassificationRow.task_id == task_id)
        .order_by(ClassificationRow.created_at.desc(), ClassificationRow.id.desc())
        .limit(1)
    )
    if classification is not None and classification.fix_instruction:
        candidates.append(
            (classification.created_at, "DEBUGGER", classification.fix_instruction)
        )
    review = db.scalar(
        select(ReviewRow).where(ReviewRow.task_id == task_id)
        .order_by(ReviewRow.created_at.desc(), ReviewRow.id.desc())
        .limit(1)
    )
    if review is not None and review.decision != "APPROVE" and review.issues:
        text = "; ".join(
            f"{i.get('severity', 'medium')}: {i.get('description', '')} "
            f"({i.get('file', '?')}:{i.get('line', '?')})"
            for i in review.issues
        )
        candidates.append((review.created_at, f"REVIEWER {review.decision}", text))
    security = db.scalar(
        select(SecurityRow).where(SecurityRow.task_id == task_id)
        .order_by(SecurityRow.created_at.desc(), SecurityRow.id.desc())
        .limit(1)
    )
    if security is not None and security.decision == "FAIL" and security.findings:
        text = "; ".join(
            f"{f.get('category', 'OTHER')}: {f.get('description', '')} "
            f"({f.get('file', '?')}:{f.get('line', '?')})"
            for f in security.findings
        )
        candidates.append((security.created_at, "SECURITY", text))
    if not candidates:
        return None
    _, source, text = max(candidates, key=lambda c: c[0])
    return f"[{source}] {text}"


def _replan_to_implementing(
    db: Session,
    task_id: uuid.UUID,
    *,
    expected: TaskStatus,
    reason: str,
) -> TaskStatus | None:
    """The ONE shared replan path (Phase 8/9): lock the task, enforce the
    shared replan budget at the transition, increment ``replan_count``, and
    route to IMPLEMENTING — or ESCALATED when the budget is exhausted.

    Used by DEBUGGING (debugger_replan), REVIEWING (review_requested_changes
    / review_rejected), and SECURITY_REVIEW (security_failed). The developer
    never sees the budget; the gate is this transition. The expected-status
    check (populate_existing under the lock) makes concurrent workers skip
    instead of double-replanning.
    """
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
            "Task %s already at %s (expected %s) — skipping %s replan",
            task_id, locked.status, expected.value, reason,
        )
        return TaskStatus(locked.status)
    if locked.max_replans is not None and locked.replan_count >= locked.max_replans:
        logger.warning(
            "Task %s replan budget exhausted (%d >= %d) — ESCALATED",
            task_id, locked.replan_count, locked.max_replans,
        )
        transition_task(
            db, locked, TaskStatus.ESCALATED, reason="replan_budget_exhausted"
        )
        db.commit()
        return TaskStatus.ESCALATED
    locked.replan_count += 1
    transition_task(db, locked, TaskStatus.IMPLEMENTING, reason=reason)
    db.commit()
    logger.info(
        "Task %s -> IMPLEMENTING (%s replan #%d)", task_id, reason, locked.replan_count
    )
    return TaskStatus.IMPLEMENTING


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


async def _run_implementing(db: Session, task: Task, developer) -> TaskStatus:
    """IMPLEMENTING: real Developer Agent or a clean FAILED (no agent).

    The developer receives the ACTIVE plan's implement step and the task's
    most recent research artifact; it must persist a real commit + grounded
    ImplementationSummary before the transition to TESTING fires. TESTING is
    the state machine's ONLY legal success successor from IMPLEMENTING, so
    routing there on success is deterministic, not a stub "happy-path"
    choice (the real decision stub is TESTING's branching, which arrives
    with the Test Agent in Phase 8).
    """
    if developer is None:
        logger.error("Task %s in IMPLEMENTING but no developer agent", task.id)
        return _cas_transition(
            db, task.id, TaskStatus.IMPLEMENTING, TaskStatus.FAILED, "no_developer_agent"
        )

    from app.agents.developer.agent import DeveloperError
    from app.llm.errors import LLMMalformedOutputError, LLMTimeoutError
    from app.models import Plan as PlanRow
    from app.models import PlanStep as PlanStepRow
    from app.models import ResearchArtifact as ArtifactRow
    from app.tools.base import ExecutionContext

    plan = db.scalar(
        select(PlanRow).where(
            PlanRow.task_id == task.id, PlanRow.status == "ACTIVE"
        ).order_by(PlanRow.created_at.desc()).limit(1)
    )
    implement_step = None
    if plan is not None:
        implement_step = db.scalar(
            select(PlanStepRow).where(
                PlanStepRow.plan_id == plan.id, PlanStepRow.step_type == "implement"
            ).order_by(PlanStepRow.sequence).limit(1)
        )
    if implement_step is None:
        logger.error("Task %s in IMPLEMENTING but its active plan has no implement step", task.id)
        return _cas_transition(
            db, task.id, TaskStatus.IMPLEMENTING, TaskStatus.FAILED, "no_implement_step"
        )

    artifact = db.scalar(
        select(ArtifactRow).where(ArtifactRow.task_id == task.id)
        .order_by(ArtifactRow.created_at.desc()).limit(1)
    )
    if artifact is None:
        logger.error("Task %s in IMPLEMENTING but no research artifact exists", task.id)
        return _cas_transition(
            db, task.id, TaskStatus.IMPLEMENTING, TaskStatus.FAILED, "no_research_artifact"
        )

    # A replan after debugging/review/security carries that source's CONCRETE
    # fix instruction (Phase 8/9) — the developer receives it as DATA for its
    # next run, labeled with which gate rejected it (never merged/ambiguous:
    # if the Reviewer approved but Security then failed, the developer sees
    # the SECURITY finding, not a mix). The latest of the three sources wins.
    fix_instruction = _latest_fix_instruction(db, task.id)

    ctx = ExecutionContext(task_id=task.id, agent_type="developer", db=db)
    try:
        await developer.run(
            task, implement_step, artifact, ctx, fix_instruction=fix_instruction
        )
    except DeveloperError as exc:
        logger.error("Task %s implementation failed: %s", task.id, exc)
        return _cas_transition(
            db, task.id, TaskStatus.IMPLEMENTING, TaskStatus.FAILED, "developer_failed"
        )
    except (LLMTimeoutError, LLMMalformedOutputError) as exc:
        logger.error("Task %s developer LLM error: %s", task.id, exc)
        return _cas_transition(
            db, task.id, TaskStatus.IMPLEMENTING, TaskStatus.FAILED, "developer_llm_error"
        )
    except Exception:  # noqa: BLE001 — never hang, never silently continue
        logger.exception("Task %s developer crashed", task.id)
        return _cas_transition(
            db, task.id, TaskStatus.IMPLEMENTING, TaskStatus.FAILED, "developer_error"
        )
    logger.info("Task %s -> TESTING (implementation summary persisted)", task.id)
    return _cas_transition(
        db, task.id, TaskStatus.IMPLEMENTING, TaskStatus.TESTING, "implementation_persisted"
    )


async def _run_testing(db: Session, task: Task, tester) -> TaskStatus:
    """TESTING (Phase 8): run the real test command, branch for real.

    The Test Agent is deterministic (Section 41): one ``shell.run_test``
    against the repository's configured test command, exit code + structured
    parser — no LLM judgment call. Branching is now REAL, not a stub:

    - ``passed`` -> REVIEWING (the state machine's only legal success
      successor from TESTING).
    - ``failed`` / ``error`` -> DEBUGGING. ``error`` (timeout, no tests
      collected, no configured test_command) is deliberately distinct from
      ``failed`` so the Debugger can tell a hung suite from a clean failing
      exit code.
    """
    if tester is None:
        logger.error("Task %s in TESTING but no tester agent", task.id)
        return _cas_transition(
            db, task.id, TaskStatus.TESTING, TaskStatus.FAILED, "no_tester_agent"
        )

    from app.agents.tester.agent import TestError
    from app.git.worktree_manager import WorktreeManager
    from app.tools.base import ExecutionContext

    ctx = ExecutionContext(task_id=task.id, agent_type="tester", db=db)
    try:
        worktree = WorktreeManager(db).get_or_create_for_task(task)
        result = await tester.run(task, worktree, ctx)
    except TestError as exc:
        logger.error("Task %s testing failed: %s", task.id, exc)
        return _cas_transition(
            db, task.id, TaskStatus.TESTING, TaskStatus.FAILED, "testing_failed"
        )
    except Exception:  # noqa: BLE001 — never hang, never silently continue
        logger.exception("Task %s tester crashed", task.id)
        return _cas_transition(
            db, task.id, TaskStatus.TESTING, TaskStatus.FAILED, "testing_error"
        )

    if result.status == "passed":
        logger.info(
            "Task %s -> REVIEWING (tests passed: %d passed, %dms)",
            task.id, result.passed, result.duration_ms,
        )
        return _cas_transition(
            db, task.id, TaskStatus.TESTING, TaskStatus.REVIEWING, "tests_passed"
        )
    logger.info(
        "Task %s -> DEBUGGING (tests %s: %d failed)",
        task.id, result.status, result.failed,
    )
    return _cas_transition(
        db, task.id, TaskStatus.TESTING, TaskStatus.DEBUGGING, f"tests_{result.status}"
    )


async def _run_debugging(db: Session, task: Task, debugger) -> TaskStatus:
    """DEBUGGING (Phase 8): classify the failure, branch for real.

    The Debugger investigates the failing run (read-only), re-runs the
    suite ONCE via the Test Agent to OBSERVE flakiness rather than guess it,
    and produces a persisted ``FailureClassification``. Branching:

    - ``is_flaky`` -> REVIEWING — treated as if TESTING had passed (Section
      10); the flaky result stays on the trace, never swept away.
    - ``fixable == False`` -> FAILED with the category attached — an
      environment/dependency failure is not something re-running the
      Developer fixes.
    - fixable -> IMPLEMENTING with ``replan_count`` + 1, enforced at the
      TRANSITION (the Developer never learns about budgets) and checked
      against ``max_replans``: exhausted -> ESCALATED, not another attempt.
    """
    if debugger is None:
        logger.error("Task %s in DEBUGGING but no debugger agent", task.id)
        return _cas_transition(
            db, task.id, TaskStatus.DEBUGGING, TaskStatus.FAILED, "no_debugger_agent"
        )

    from app.agents.debugger.agent import DebuggerError
    from app.agents.tester.schema import result_from_row
    from app.llm.errors import LLMMalformedOutputError, LLMTimeoutError
    from app.models import ImplementationSummary as SummaryRow
    from app.models import TestRun
    from app.tools.base import ExecutionContext

    run = db.scalar(
        select(TestRun).where(TestRun.task_id == task.id)
        .order_by(TestRun.created_at.desc(), TestRun.id.desc()).limit(1)
    )
    if run is None:
        logger.error("Task %s in DEBUGGING but no test run exists", task.id)
        return _cas_transition(
            db, task.id, TaskStatus.DEBUGGING, TaskStatus.FAILED, "no_test_run"
        )
    summary = db.scalar(
        select(SummaryRow).where(SummaryRow.task_id == task.id)
        .order_by(SummaryRow.created_at.desc()).limit(1)
    )
    if summary is None:
        logger.error(
            "Task %s in DEBUGGING but no implementation summary exists", task.id
        )
        return _cas_transition(
            db, task.id, TaskStatus.DEBUGGING, TaskStatus.FAILED, "no_implementation_summary"
        )

    ctx = ExecutionContext(task_id=task.id, agent_type="debugger", db=db)
    try:
        classification = await debugger.run(
            task, result_from_row(run), summary, ctx
        )
    except DebuggerError as exc:
        logger.error("Task %s debugging failed: %s", task.id, exc)
        return _cas_transition(
            db, task.id, TaskStatus.DEBUGGING, TaskStatus.FAILED, "debugger_failed"
        )
    except (LLMTimeoutError, LLMMalformedOutputError) as exc:
        logger.error("Task %s debugger LLM error: %s", task.id, exc)
        return _cas_transition(
            db, task.id, TaskStatus.DEBUGGING, TaskStatus.FAILED, "debugger_llm_error"
        )
    except Exception:  # noqa: BLE001 — never hang, never silently continue
        logger.exception("Task %s debugger crashed", task.id)
        return _cas_transition(
            db, task.id, TaskStatus.DEBUGGING, TaskStatus.FAILED, "debugger_error"
        )

    if classification.is_flaky:
        logger.info(
            "Task %s -> REVIEWING (flaky test detected, never blocks the pipeline)",
            task.id,
        )
        return _cas_transition(
            db, task.id, TaskStatus.DEBUGGING, TaskStatus.REVIEWING, "flaky_test"
        )
    if not classification.fixable:
        logger.info(
            "Task %s -> FAILED (unfixable %s: %s)",
            task.id, classification.category, classification.root_cause[:200],
        )
        return _cas_transition(
            db, task.id, TaskStatus.DEBUGGING, TaskStatus.FAILED,
            f"unfixable:{classification.category}",
        )

    # Fixable: bounded replan through the ONE shared replan path (Phase 9 —
    # Debugger/Reviewer/Security all draw from the same max_replans budget,
    # Section 42). The budget is enforced HERE at the transition (the
    # Developer never sees it), under a fresh row lock — the debugger
    # committed internally, releasing the job's original FOR UPDATE lock.
    return _replan_to_implementing(
        db, task.id, expected=TaskStatus.DEBUGGING, reason="debugger_replan"
    )


async def _run_reviewing(db: Session, task: Task, reviewer) -> TaskStatus:
    """REVIEWING (Phase 9): the Reviewer Agent critiques the commit, branch
    for real.

    The Reviewer sees the commit diff + the test result ONLY — the
    ImplementationSummary is deliberately NOT passed (the agent's run
    signature has no summary parameter; the prompt builder has no summary
    field). Branching:

    - APPROVE -> SECURITY_REVIEW (the state machine's only legal success
      successor from REVIEWING).
    - REQUEST_CHANGES / REJECT -> IMPLEMENTING through the shared replan
      path, the issues labeled as the developer's fix instruction
      (``_latest_fix_instruction`` picks them up on the next run), budget
      enforced at the transition.
    """
    if reviewer is None:
        logger.error("Task %s in REVIEWING but no reviewer agent", task.id)
        return _cas_transition(
            db, task.id, TaskStatus.REVIEWING, TaskStatus.FAILED, "no_reviewer_agent"
        )

    from app.agents.reviewer.agent import ReviewerError
    from app.agents.tester.schema import result_from_row
    from app.llm.errors import LLMMalformedOutputError, LLMTimeoutError
    from app.models import ImplementationSummary as SummaryRow
    from app.models import TestRun
    from app.tools.base import ExecutionContext

    summary = db.scalar(
        select(SummaryRow).where(SummaryRow.task_id == task.id)
        .order_by(SummaryRow.created_at.desc(), SummaryRow.id.desc()).limit(1)
    )
    if summary is None or not summary.commit_sha:
        logger.error(
            "Task %s in REVIEWING but no implementation summary with a commit", task.id
        )
        return _cas_transition(
            db, task.id, TaskStatus.REVIEWING, TaskStatus.FAILED,
            "no_implementation_summary",
        )
    run = db.scalar(
        select(TestRun).where(TestRun.task_id == task.id)
        .order_by(TestRun.created_at.desc(), TestRun.id.desc()).limit(1)
    )
    if run is None:
        logger.error("Task %s in REVIEWING but no test run exists", task.id)
        return _cas_transition(
            db, task.id, TaskStatus.REVIEWING, TaskStatus.FAILED, "no_test_run"
        )

    ctx = ExecutionContext(task_id=task.id, agent_type="reviewer", db=db)
    try:
        review = await reviewer.run(task, summary.commit_sha, result_from_row(run), ctx)
    except ReviewerError as exc:
        logger.error("Task %s review failed: %s", task.id, exc)
        return _cas_transition(
            db, task.id, TaskStatus.REVIEWING, TaskStatus.FAILED, "review_failed"
        )
    except (LLMTimeoutError, LLMMalformedOutputError) as exc:
        logger.error("Task %s reviewer LLM error: %s", task.id, exc)
        return _cas_transition(
            db, task.id, TaskStatus.REVIEWING, TaskStatus.FAILED, "review_llm_error"
        )
    except Exception:  # noqa: BLE001 — never hang, never silently continue
        logger.exception("Task %s reviewer crashed", task.id)
        return _cas_transition(
            db, task.id, TaskStatus.REVIEWING, TaskStatus.FAILED, "review_error"
        )

    if review.decision == "APPROVE":
        logger.info("Task %s -> SECURITY_REVIEW (review approved)", task.id)
        return _cas_transition(
            db, task.id, TaskStatus.REVIEWING, TaskStatus.SECURITY_REVIEW,
            "review_approved",
        )
    reason = (
        "review_requested_changes"
        if review.decision == "REQUEST_CHANGES"
        else "review_rejected"
    )
    logger.info(
        "Task %s -> IMPLEMENTING (review %s)", task.id, review.decision
    )
    return _replan_to_implementing(
        db, task.id, expected=TaskStatus.REVIEWING, reason=reason
    )


async def _run_security_review(db: Session, task: Task, security) -> TaskStatus:
    """SECURITY_REVIEW (Phase 9): the Security Agent runs its checklist on
    the same commit, blind to the Reviewer's verdict.

    - PASS -> VERIFICATION (the state machine's only legal success
      successor from SECURITY_REVIEW).
    - FAIL -> IMPLEMENTING through the shared replan path, the findings
      labeled as the fix instruction, budget enforced at the transition.
    """
    if security is None:
        logger.error("Task %s in SECURITY_REVIEW but no security agent", task.id)
        return _cas_transition(
            db, task.id, TaskStatus.SECURITY_REVIEW, TaskStatus.FAILED,
            "no_security_agent",
        )

    from app.agents.security.agent import SecurityError
    from app.llm.errors import LLMMalformedOutputError, LLMTimeoutError
    from app.models import ImplementationSummary as SummaryRow
    from app.tools.base import ExecutionContext

    summary = db.scalar(
        select(SummaryRow).where(SummaryRow.task_id == task.id)
        .order_by(SummaryRow.created_at.desc(), SummaryRow.id.desc()).limit(1)
    )
    if summary is None or not summary.commit_sha:
        logger.error(
            "Task %s in SECURITY_REVIEW but no implementation summary with a commit",
            task.id,
        )
        return _cas_transition(
            db, task.id, TaskStatus.SECURITY_REVIEW, TaskStatus.FAILED,
            "no_implementation_summary",
        )

    ctx = ExecutionContext(task_id=task.id, agent_type="security", db=db)
    try:
        result = await security.run(task, summary.commit_sha, ctx)
    except SecurityError as exc:
        logger.error("Task %s security review failed: %s", task.id, exc)
        return _cas_transition(
            db, task.id, TaskStatus.SECURITY_REVIEW, TaskStatus.FAILED,
            "security_failed",
        )
    except (LLMTimeoutError, LLMMalformedOutputError) as exc:
        logger.error("Task %s security LLM error: %s", task.id, exc)
        return _cas_transition(
            db, task.id, TaskStatus.SECURITY_REVIEW, TaskStatus.FAILED,
            "security_llm_error",
        )
    except Exception:  # noqa: BLE001 — never hang, never silently continue
        logger.exception("Task %s security agent crashed", task.id)
        return _cas_transition(
            db, task.id, TaskStatus.SECURITY_REVIEW, TaskStatus.FAILED,
            "security_error",
        )

    if result.decision == "PASS":
        logger.info("Task %s -> VERIFICATION (security passed)", task.id)
        return _cas_transition(
            db, task.id, TaskStatus.SECURITY_REVIEW, TaskStatus.VERIFICATION,
            "security_passed",
        )
    logger.info(
        "Task %s -> IMPLEMENTING (security failed: %d findings)",
        task.id, len(result.findings),
    )
    return _replan_to_implementing(
        db, task.id, expected=TaskStatus.SECURITY_REVIEW, reason="security_failed"
    )


async def _run_verification(db: Session, task: Task) -> TaskStatus:
    """VERIFICATION (Phase 9): a thin, code-only staleness check — no LLM.

    Confirms the approval is still valid before PR_CREATION:

    1. The commit that was reviewed is still the current HEAD of the
       worktree branch (nothing changed between review and now — a real
       possibility once replan loops exist).
    2. The last test run still holds a "passed" status for that commit.

    Either check failing = the review is STALE -> back to TESTING (re-run
    the current state) rather than proceeding on an invalidated approval.
    Both passing -> PR_CREATION.
    """
    from app.git.operations import GitOperations
    from app.git.worktree_manager import WorktreeManager
    from app.models import ImplementationSummary as SummaryRow
    from app.models import TestRun

    summary = db.scalar(
        select(SummaryRow).where(SummaryRow.task_id == task.id)
        .order_by(SummaryRow.created_at.desc(), SummaryRow.id.desc()).limit(1)
    )
    if summary is None or not summary.commit_sha:
        logger.error(
            "Task %s in VERIFICATION but no implementation summary with a commit",
            task.id,
        )
        return _cas_transition(
            db, task.id, TaskStatus.VERIFICATION, TaskStatus.FAILED,
            "no_implementation_summary",
        )

    run = db.scalar(
        select(TestRun).where(TestRun.task_id == task.id)
        .order_by(TestRun.created_at.desc(), TestRun.id.desc()).limit(1)
    )
    tests_still_pass = run is not None and run.status == "passed"
    if not tests_still_pass:
        logger.warning(
            "Task %s verification stale: last test run is %s, not passed — "
            "back to TESTING", task.id, getattr(run, "status", None),
        )
        return _cas_transition(
            db, task.id, TaskStatus.VERIFICATION, TaskStatus.TESTING, "stale_review"
        )

    try:
        worktree = WorktreeManager(db).get_or_create_for_task(task)
        head = GitOperations(worktree.path).head_sha()
    except Exception as exc:  # noqa: BLE001 — a missing worktree is stale, not a crash
        logger.warning(
            "Task %s verification stale: cannot read worktree HEAD (%s)", task.id, exc
        )
        return _cas_transition(
            db, task.id, TaskStatus.VERIFICATION, TaskStatus.TESTING, "stale_review"
        )
    if head != summary.commit_sha:
        logger.warning(
            "Task %s verification stale: HEAD %s != reviewed commit %s — "
            "back to TESTING", task.id, head, summary.commit_sha,
        )
        return _cas_transition(
            db, task.id, TaskStatus.VERIFICATION, TaskStatus.TESTING, "stale_review"
        )

    logger.info("Task %s -> PR_CREATION (verification passed)", task.id)
    return _cas_transition(
        db, task.id, TaskStatus.VERIFICATION, TaskStatus.PR_CREATION,
        "verification_passed",
    )
