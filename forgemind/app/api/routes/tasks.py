"""Task API routes (Phase 1 + Phase 2 contracts, Phase 10.5 auth).

POST /tasks                {objective, repository_url} -> 201, enqueues advance_task  [auth]
GET  /tasks                -> list of tasks                                            [open]
GET  /tasks/{id}           -> full task record                                          [open]
POST /tasks/{id}/cancel    -> transition to FAILED ("user_cancelled"), enqueues nothing [auth]
POST /tasks/{id}/approve   -> COMPLETED (human yes at the checkpoint)                   [auth]
POST /tasks/{id}/reject    -> FAILED (human no at the checkpoint)                       [auth]
GET  /tasks/{id}/events    -> execution_events, ordered by created_at                   [open]

Every state-mutating route is gated by the shared bearer token
(``require_api_token``) and writes the authenticated identity ("token-holder")
to the audit trail — the one place that authorizes real-world side effects
must not have an author gap. Read routes stay open so a human can watch a task
walk the pipeline without the token.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps.auth import TOKEN_HOLDER, require_api_token
from app.database.session import get_db
from app.models import (
    Approval,
    AuditLog,
    ExecutionEvent,
    PullRequest,
    Repository,
    Task,
    TaskStatus,
)
from app.runtime.state_machine import TERMINAL_STATES
from app.runtime.task_lifecycle import USER_CANCELLED, transition_task
from app.schemas import ApprovalRequest, ExecutionEventRead, TaskCreate, TaskRead
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


@router.post(
    "",
    response_model=TaskRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_token)],
)
async def create_task(payload: TaskCreate, db: Session = Depends(get_db)) -> Task:
    """Create a task in CREATED state, record an audit entry, enqueue the worker.

    Bearer-token gated: creating a task triggers real LLM spend once the
    worker picks it up, so it is never an anonymous action.
    """
    repository = _get_or_create_repository(db, payload.repository_url)
    # Phase 10: the fork ForgeMind will push/PR against (stored on the repo
    # row and shared by its tasks). Unset fork_url means PR_CREATION fails
    # closed — there is no fallback to the upstream URL.
    if payload.fork_url:
        repository.fork_url = payload.fork_url

    task = Task(
        objective=payload.objective,
        repository_id=repository.id,
        status=TaskStatus.CREATED.value,
        issue_number=payload.issue_number,
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
            details={
                "repository_url": payload.repository_url,
                **(
                    {"fork_url": payload.fork_url}
                    if payload.fork_url is not None
                    else {}
                ),
                **(
                    {"issue_number": payload.issue_number}
                    if payload.issue_number is not None
                    else {}
                ),
            },
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
        logger.warning(
            "Failed to enqueue advance_task for %s — will be swept later", task.id
        )
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


@router.post(
    "/{task_id}/cancel",
    response_model=TaskRead,
)
def cancel_task(
    task_id: uuid.UUID,
    db: Session = Depends(get_db),
    actor: str = Depends(require_api_token),
) -> Task:
    """Cancel a task: transition to FAILED with reason ``user_cancelled``.

    Row-locked so it cannot race a worker transition. Terminal tasks
    (COMPLETED/ESCALATED) and already-FAILED tasks return 409 — never a
    silent no-op. The authenticated actor is recorded on the audit trail.
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
    db.add(
        AuditLog(
            task_id=task.id,
            actor=actor,
            action="task.cancelled",
            entity_type="task",
            entity_id=str(task.id),
            details={"reason": USER_CANCELLED},
        )
    )
    db.commit()
    db.refresh(task)
    logger.info("Task %s cancelled by user (%s -> FAILED)", task.id, current.value)
    return task


def _await_approval_lock(
    db: Session, task_id: uuid.UUID, endpoint: str
) -> tuple[Task, TaskStatus]:
    """Row-lock the task and enforce that it is actually awaiting approval.

    Returns ``(task, current)``; raises the HTTP 404/409 errors that the
    approve/reject endpoints share — a decision on a task that is NOT
    waiting for a human is a 409, never a silent no-op.
    """
    task = db.execute(
        select(Task).where(Task.id == task_id).with_for_update()
    ).scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    current = TaskStatus(task.status)
    if current is not TaskStatus.AWAITING_APPROVAL:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot {endpoint} task in state {current.value} — "
            "it must be AWAITING_APPROVAL",
        )
    return task, current


def _record_approval(
    db: Session,
    task: Task,
    action: str,
    reason: str | None,
    actor: str = TOKEN_HOLDER,
) -> None:
    """Insert the human decision row + match the PR row's status.

    ``actor`` is the authenticated identity from ``require_api_token`` — with
    the single shared token that is ``TOKEN_HOLDER``; the audit trail records
    *who* approved/rejected (the bearer holder) so the one action that
    authorizes real-world side effects has no author gap.
    """
    db.add(
        Approval(task_id=task.id, action=action, reason=reason),
    )
    db.add(
        AuditLog(
            task_id=task.id,
            actor=actor,
            action=f"task.{action}",
            entity_type="task",
            entity_id=str(task.id),
            details={"reason": reason} if reason else None,
        )
    )
    pr = db.scalar(
        select(PullRequest)
        .where(PullRequest.task_id == task.id)
        .order_by(PullRequest.created_at.desc(), PullRequest.id.desc())
        .limit(1)
    )
    if pr is not None:
        # The PR row mirrors the human decision (PR-status vocabulary).
        # It never claims a merge — nothing here merges anything.
        pr.status = "approved" if action == "approve" else "rejected"
    db.flush()


@router.post("/{task_id}/approve", response_model=TaskRead)
def approve_task(
    task_id: uuid.UUID,
    payload: ApprovalRequest | None = None,
    db: Session = Depends(get_db),
    actor: str = Depends(require_api_token),
) -> Task:
    """The human 'yes' at the AWAITING_APPROVAL checkpoint.

    Bearer-token gated. Records an ``approvals`` row (action=approve) and
    transitions the task to COMPLETED. Per section 18/13 this does NOT merge
    anything — merging remains a manual action on GitHub. Approval means
    "I reviewed ForgeMind's PR and consider the task done".

    SCOPE: the token is a SINGLE shared secret, so "who" is the token holder,
    not an individual user. That matches the single-operator MVP; multi-user
    authorization (per-account, per-task) is future work and is documented as
    such in the README.
    """
    task, current = _await_approval_lock(db, task_id, "approve")
    reason = payload.reason if payload else None
    _record_approval(db, task, "approve", reason, actor=actor)
    transition_task(db, task, TaskStatus.COMPLETED, reason="user_approved")
    db.commit()
    db.refresh(task)
    logger.info("Task %s approved by user (AWAITING_APPROVAL -> COMPLETED)", task.id)
    return task


@router.post("/{task_id}/reject", response_model=TaskRead)
def reject_task(
    task_id: uuid.UUID,
    payload: ApprovalRequest | None = None,
    db: Session = Depends(get_db),
    actor: str = Depends(require_api_token),
) -> Task:
    """The human 'no' at the AWAITING_APPROVAL checkpoint.

    Bearer-token gated. Records an ``approvals`` row (action=reject) and
    transitions the task to FAILED — a deliberate stop, NOT a replan. Reason
    preserved on the event and the approval row. Same single-token scope as
    ``approve``.
    """
    task, current = _await_approval_lock(db, task_id, "reject")
    reason = payload.reason if payload else None
    _record_approval(db, task, "reject", reason, actor=actor)
    transition_task(
        db,
        task,
        TaskStatus.FAILED,
        reason=("user_rejected" if reason is None else f"user_rejected: {reason}"),
    )
    db.commit()
    db.refresh(task)
    logger.info("Task %s rejected by user (AWAITING_APPROVAL -> FAILED)", task.id)
    return task


@router.get("/{task_id}/events", response_model=list[ExecutionEventRead])
def list_events(
    task_id: uuid.UUID, db: Session = Depends(get_db)
) -> list[ExecutionEvent]:
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
