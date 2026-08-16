"""``advance_task`` arq job — the worker's unit of work.

One job = at most one state transition. The transition is applied atomically
under a row lock (see ``advance_task_once``); if it succeeds the job
re-enqueues itself so the pipeline keeps moving. Illegal transitions are
caught, logged, and never silently applied. A crash between the commit and
the re-enqueue is healed by the worker's startup sweep (Section J).
"""

from __future__ import annotations

import logging
import os
import time
import uuid

from app.database.session import SessionLocal
from app.models import TaskStatus
from app.runtime.state_machine import TERMINAL_STATES, IllegalTransitionError
from app.runtime.task_lifecycle import advance_task_once
from app.worker.queue import JOB_ADVANCE_TASK

logger = logging.getLogger(__name__)


async def advance_task(ctx: dict, task_id: str) -> None:
    """Load the task, apply the next legal transition, persist, re-enqueue."""
    # Test/ops knob: simulate slow transitions so crash windows are observable.
    delay_ms = int(os.environ.get("FORGEMIND_STEP_DELAY_MS", "0") or 0)
    if delay_ms > 0:
        time.sleep(delay_ms / 1000)

    task_uuid = uuid.UUID(task_id)
    db = SessionLocal()
    try:
        new_status = advance_task_once(db, task_uuid)
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

    # Test hook: simulate a crash in the window between the transition
    # committing and the re-enqueue — exactly what the startup sweep heals.
    # Never set outside tests.
    if os.environ.get("FORGEMIND_CRASH_AFTER_COMMIT") == "1":
        logger.warning("FORGEMIND_CRASH_AFTER_COMMIT set — simulating crash after commit")
        os._exit(1)

    if new_status is not None and new_status not in TERMINAL_STATES:
        await ctx["redis"].enqueue_job(JOB_ADVANCE_TASK, task_id)

    if new_status in (TaskStatus.COMPLETED, TaskStatus.ESCALATED):
        logger.info("Task %s reached terminal state %s", task_id, new_status.value)
