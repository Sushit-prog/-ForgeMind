"""``advance_task`` arq job — the worker's unit of work.

One job = at most one state transition. PLANNING runs the real Planning
Agent (LLM call, persisted plan); RESEARCHING runs the real Research
Agent (bounded tool-use loop, persisted artifact); IMPLEMENTING runs the
real Developer Agent (bounded tool-use loop, one commit, persisted
summary); TESTING runs the deterministic Test Agent (real subprocess run
of the configured test command — Section 41, no LLM call); DEBUGGING runs
the real Debugger Agent (read-only investigation + classification with the
flakiness re-run); every other state uses the stub driver. The transition
is applied atomically under a row lock; if it succeeds the job re-enqueues
itself so the pipeline keeps moving. Illegal transitions are caught,
logged, and never silently applied. A crash between the commit and the
re-enqueue is healed by the worker's startup sweep (Section J).

Agents are built PER JOB, never cached: the stub LLM provider's canned
script is per-task state (each agent run consumes its proposal queue), so
reusing one agent instance across tasks would exhaust the script and leave
later tasks with no proposals — fatal for the developer, whose zero-commit
path is a hard failure. Real providers are stateless, so per-job
construction costs nothing and is the honest model.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid

from app.config import get_settings
from app.database.session import SessionLocal
from app.models import Task, TaskStatus
from app.runtime.state_machine import TERMINAL_STATES, IllegalTransitionError
from app.runtime.task_lifecycle import advance_task_with_agents, transition_task
from app.worker.queue import JOB_ADVANCE_TASK, advance_job_id

logger = logging.getLogger(__name__)

# The worker enforces its OWN inner deadline slightly below arq's outer
# job_timeout ceiling (settings.worker_job_timeout_seconds): a hung round-trip
# then converts into an explicit FAILED(job_timeout) — routed through the
# standard FAILED -> RECOVERING -> REPLANNING recovery — instead of arq
# killing the job outright and silently orphaning the row (run #1 and
# attempt #5 both failed this way before this existed).
JOB_TIMEOUT_REASON = "job_timeout"
_JOB_TIMEOUT_MARGIN_SECONDS = 30
MIN_INNER_BUDGET_SECONDS = 60


def _mark_job_timeout_failure(db, task_id: uuid.UUID) -> TaskStatus | None:
    """CAS the stuck task to FAILED(job_timeout); None when not applicable.

    Skips terminal rows and rows already FAILED — a user_cancelled row is
    intentionally terminal and must never be resurrected by this path.
    """
    locked = db.get(Task, task_id)
    if locked is None:
        return None
    current = TaskStatus(locked.status)
    if current in TERMINAL_STATES or current is TaskStatus.FAILED:
        return None
    transition_task(db, locked, TaskStatus.FAILED, reason=JOB_TIMEOUT_REASON)
    db.commit()
    return TaskStatus.FAILED


def _build_agent(build_fn, label: str):
    """Construct one agent fresh (real OpenRouter or stub). None when
    unconfigured — its state's tasks then fail cleanly instead of hanging."""
    try:
        return build_fn()
    except Exception as exc:  # noqa: BLE001 — unconfigured provider
        logger.warning("%s unavailable (%s); its tasks will fail", label, exc)
        return None


def _test_knob_int(name: str, default: int = 0) -> int:
    """Read a test-only integer knob, honoring it ONLY in test mode.

    ``FORGEMIND_TEST_MODE=1`` (set by the e2e harness, never by prod config)
    must be present or every ``FORGEMIND_*`` test knob is ignored. A stray
    ``FORGEMIND_STEP_DELAY_MS`` or ``FORGEMIND_CRASH_AFTER_COMMIT`` in a prod
    .env is therefore inert instead of silently stalling or killing workers.
    """
    if os.environ.get("FORGEMIND_TEST_MODE") != "1":
        return default
    value = os.environ.get(name)
    return int(value) if value else default


async def advance_task(ctx: dict, task_id: str) -> None:
    """Load the task, apply the next legal transition, persist, re-enqueue."""
    # Test knob (test_mode-gated): simulate slow transitions so crash
    # windows are observable. asyncio.sleep, not time.sleep — even a
    # misconfigured test worker thread can't block the event loop.
    delay_ms = _test_knob_int("FORGEMIND_STEP_DELAY_MS")
    if delay_ms > 0:
        await asyncio.sleep(delay_ms / 1000)

    task_uuid = uuid.UUID(task_id)
    db = SessionLocal()
    new_status: TaskStatus | None = None
    try:
        from app.agents.debugger.agent import build_debugger
        from app.agents.developer.agent import build_developer
        from app.agents.github_agent.agent import build_github
        from app.agents.planner.agent import build_planner
        from app.agents.researcher.agent import build_researcher
        from app.agents.reviewer.agent import build_reviewer
        from app.agents.security.agent import build_security
        from app.agents.tester.agent import build_tester

        # Fresh agents per job: the stub provider's proposal script must start
        # over for each task (see module docstring). The tester is
        # deterministic (no LLM provider at all) and every other agent's
        # provider is per-job like the rest.
        settings = get_settings()
        inner_budget = (
            settings.worker_inner_deadline_seconds
            if settings.worker_inner_deadline_seconds is not None
            else max(
                settings.worker_job_timeout_seconds - _JOB_TIMEOUT_MARGIN_SECONDS,
                MIN_INNER_BUDGET_SECONDS,
            )
        )
        async with asyncio.timeout(inner_budget):
            new_status = await advance_task_with_agents(
                db,
                task_uuid,
                _build_agent(build_planner, "Planner"),
                _build_agent(build_researcher, "Researcher"),
                _build_agent(build_developer, "Developer"),
                _build_agent(build_tester, "Tester"),
                _build_agent(build_debugger, "Debugger"),
                _build_agent(build_reviewer, "Reviewer"),
                _build_agent(build_security, "Security"),
                _build_agent(build_github, "GitHub"),
            )
    except TimeoutError:
        # OUR inner deadline fired before arq's outer ceiling.
        db.rollback()
        new_status = _mark_job_timeout_failure(db, task_uuid)
        logger.warning(
            "task %s exceeded %ss inner budget — marked %s (%s)",
            task_id,
            inner_budget,
            new_status,
            JOB_TIMEOUT_REASON,
        )
    except IllegalTransitionError as exc:
        # Deterministic guard fired: log loudly, never silently update status.
        db.rollback()
        logger.error("Illegal transition attempt for task %s: %s", task_id, exc)
        return
    except Exception:
        db.rollback()
        raise  # let arq retry (worker settings max_tries)
    finally:
        db.close()

    # Test hook (test_mode-gated): simulate a crash in the window between the
    # transition committing and the re-enqueue — exactly what the startup
    # sweep heals. Never honored outside FORGEMIND_TEST_MODE=1. Only fires
    # when a transition actually committed, so a stale job (task not found,
    # already advanced) cannot kill the worker before it processes the test's
    # own task.
    if (
        new_status is not None
        and os.environ.get("FORGEMIND_TEST_MODE") == "1"
        and os.environ.get("FORGEMIND_CRASH_AFTER_COMMIT") == "1"
    ):
        logger.warning(
            "FORGEMIND_CRASH_AFTER_COMMIT set — simulating crash after commit"
        )
        os._exit(1)

    if new_status is not None and new_status not in TERMINAL_STATES:
        await ctx["redis"].enqueue_job(
            JOB_ADVANCE_TASK,
            task_id,
            _job_id=advance_job_id(task_id, new_status.value),
        )

    if new_status in (TaskStatus.COMPLETED, TaskStatus.ESCALATED):
        logger.info("Task %s reached terminal state %s", task_id, new_status.value)
