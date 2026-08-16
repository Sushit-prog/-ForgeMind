"""Task API routes (Phase 1 + Phase 2 contracts).

POST /tasks                {objective, repository_url} -> 201, enqueues advance_task
GET  /tasks                -> list of tasks
GET  /tasks/{id}           -> full task record
POST /tasks/{id}/cancel    -> transition to FAILED ("user_cancelled"), enqueues nothing
GET  /tasks/{id}/events    -> execution_events, ordered by created_at
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models import AuditLog, ExecutionEvent, Repository, Task, TaskStatus
from app.runtime.state_machine import TERMINAL_STATES
from app.runtime.task_lifecycle import USER_CANCELLED, transition_task
from app.schemas import ExecutionEventRead, TaskCreate, TaskRead
from app.worker.queue import enqueue_advance_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


def _get_or_create_repository(db: Session, url: str) -> Repository:
    """Get the repository row for ``url``, creating it if unknown.

    ``repositories.url`` is unique, so a repeated POST for the same repo
    reuses the existing row instead of duplicating it.
    """
    repo = db.scalar(select(Repository).where(Repository.url == url))
    if repo is None:
        repo = Repository(url=url)
        db.add(repo)
        db.flush()  # assign id without committing — same transaction as the task
    return repo


@router.post("", response_model=TaskRead, status_code=status.HTTP_201_CREATED)
async def create_task(payload: TaskCreate, db: Session = Depends(get_db)) -> Task:
    """Create a task in CREATED state, record an audit entry, enqueue the worker."""
    repository = _get_or_create_repository(db, payload.repository_url)

    task = Task(
        objective=payload.objective,
        repository_id=repository.id,
        status=TaskStatus.CREATED.value,
    )
    db.add(task)
    db.flush()  # assign task id for the audit entry, same transaction

    db.add(
        AuditLog(
            task_id=task.id,
            actor="api",
            action="task.created",
            entity_type="task",
            entity_id=str(task.id),
            details={"repository_url": payload.repository_url},
        )
    )
    db.commit()
    db.refresh(task)
    logger.info("Task %s created (status=%s)", task.id, task.status)

    # Hand off to the worker — the API never drives transitions synchronously.
    # If the queue is unavailable the task stays CREATED and the worker's
    # startup sweep picks it up later.
    try:
        await enqueue_advance_task(task.id)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to enqueue advance_task for %s — will be swept later", task.id)
    return task


@router.get("", response_model=list[TaskRead])
def list_tasks(db: Session = Depends(get_db)) -> list[Task]:
    """List tasks, most recent first (id breaks ties within the same timestamp)."""
    return list(
        db.scalars(select(Task).order_by(Task.created_at.desc(), Task.id.desc()))
    )


@router.get("/{task_id}", response_model=TaskRead)
def get_task(task_id: uuid.UUID, db: Session = Depends(get_db)) -> Task:
    """Fetch a single task by id."""
    task = db.get(Task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.post("/{task_id}/cancel", response_model=TaskRead)
def cancel_task(task_id: uuid.UUID, db: Session = Depends(get_db)) -> Task:
    """Cancel a task: transition to FAILED with reason ``user_cancelled``.

    Row-locked so it cannot race a worker transition. Terminal tasks
    (COMPLETED/ESCALATED) and already-FAILED tasks return 409 — never a
    silent no-op.
    """
    task = db.execute(
        select(Task).where(Task.id == task_id).with_for_update()
    ).scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    current = TaskStatus(task.status)
    if current in TERMINAL_STATES or current is TaskStatus.FAILED:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel task in state {current.value}",
        )

    transition_task(db, task, TaskStatus.FAILED, reason=USER_CANCELLED)
    db.commit()
    db.refresh(task)
    logger.info("Task %s cancelled by user (%s -> FAILED)", task.id, current.value)
    return task


@router.get("/{task_id}/events", response_model=list[ExecutionEventRead])
def list_events(task_id: uuid.UUID, db: Session = Depends(get_db)) -> list[ExecutionEvent]:
    """Execution-event trail for a task, oldest first."""
    task = db.get(Task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return list(
        db.scalars(
            select(ExecutionEvent)
            .where(ExecutionEvent.task_id == task_id)
            .order_by(ExecutionEvent.created_at, ExecutionEvent.id)
        )
    )
