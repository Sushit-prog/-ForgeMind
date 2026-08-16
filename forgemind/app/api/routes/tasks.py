"""Task API routes (milestone Phase 1 contract).

POST /tasks        {objective, repository_url} -> 201 {id, status: "CREATED"}
GET  /tasks        -> list of tasks
GET  /tasks/{id}   -> full task record
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models import AuditLog, Repository, Task, TaskStatus
from app.schemas import TaskCreate, TaskRead

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
def create_task(payload: TaskCreate, db: Session = Depends(get_db)) -> Task:
    """Create a task in CREATED state and record an audit entry."""
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
