"""Task API routes (Phase 1 + Phase 2 contracts, Phase 10.5 auth, Phase 12 merge).

POST /tasks                {objective, repository_url} -> 201, enqueues advance_task  [auth]
GET  /tasks                -> list of tasks                                            [open]
GET  /tasks/{id}           -> full task record (+ recommendation)                     [open]
POST /tasks/{id}/cancel    -> transition to FAILED ("user_cancelled"), enqueues nothing [auth]
POST /tasks/{id}/approve   -> COMPLETED (human yes at the checkpoint)                   [auth]
POST /tasks/{id}/reject    -> FAILED (human no at the checkpoint)                       [auth]
POST /tasks/{id}/merge     -> gated squash-merge of the approved PR                    [auth]
GET  /tasks/{id}/events    -> execution_events, ordered by created_at                   [open]

Every state-mutating route is gated by the shared bearer token
(``require_api_token``) and writes the authenticated identity ("token-holder")
to the audit trail — the one place that authorizes real-world side effects
must not have an author gap. Read routes stay open so a human can watch a task
walk the pipeline without the token.

The Phase 12 merge endpoint is deliberately separate from approve: approval
judges the diff/plan/verdicts; merge LANDs it. A task can be approved and
merged later, or approved and never merged.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps.auth import TOKEN_HOLDER, require_api_token
from app.capabilities import capabilities_for_agent
from app.database.session import get_db
from app.execution import ToolPipeline
from app.models import (
    Approval,
    AuditLog,
    ExecutionEvent,
    PullRequest,
    Repository,
    Task,
    TaskStatus,
)
from app.runtime.recommendation import recommendation_for
from app.runtime.state_machine import TERMINAL_STATES
from app.runtime.task_lifecycle import USER_CANCELLED, transition_task
from app.runtime.task_trace import list_execution_events
from app.schemas import (
    ApprovalRequest,
    ExecutionEventRead,
    MergeResult,
    TaskCreate,
    TaskRead,
)
from app.tools.base import ExecutionContext
from app.worker.queue import enqueue_advance_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


def _task_read(db: Session, task: Task) -> TaskRead:
    """Build a ``TaskRead`` with the Phase 12 recommendation signal populated.

    Single-task endpoints (GET /{id}, approve, reject, merge) enrich the
    response. ``list_tasks`` stays lean — the list view leaves both fields
    ``None`` to avoid N+1 queries at the operator scale this system targets.
    """
    rec_action, rec_reason = recommendation_for(db, task)
    return TaskRead(
        id=task.id,
        objective=task.objective,
        repository_id=task.repository_id,
        status=task.status,
        replan_count=task.replan_count,
        created_at=task.created_at,
        updated_at=task.updated_at,
        recommended_action=rec_action,
        recommendation_reason=rec_reason,
    )


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
    #
    # The bootstrap hop runs under the task's CURRENT status (CREATED), NOT
    # "PLANNING": its id (advance:<id>:CREATED) is what the startup sweep and
    # the stale-CREATED sweep also target, so racers collapse to one job (the
    # dedupe Fix). And it lets the worker's self-chain re-enqueue the PLANNING
    # stage under a DIFFERENT id (advance:<id>:PLANNING) after committing
    # CREATED -> PLANNING. If we enqueued "PLANNING" here, the bootstrap job
    # would re-enqueue advance:<id>:PLANNING while that very job is still
    # running, arq would dedupe it (job_key survives until finish), and the
    # task would strand at PLANNING with no continuation job.
    try:
        await enqueue_advance_task(task.id, target_status=task.status)
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
def get_task(task_id: uuid.UUID, db: Session = Depends(get_db)) -> TaskRead:
    """Fetch a single task by id (with the recommendation signal)."""
    task = db.get(Task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return _task_read(db, task)


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
) -> TaskRead:
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
    return _task_read(db, task)


@router.post("/{task_id}/reject", response_model=TaskRead)
def reject_task(
    task_id: uuid.UUID,
    payload: ApprovalRequest | None = None,
    db: Session = Depends(get_db),
    actor: str = Depends(require_api_token),
) -> TaskRead:
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
    return _task_read(db, task)


# -- Phase 12: gated squash merge --------------------------------------------


@router.post("/{task_id}/merge", response_model=MergeResult)
async def merge_task(
    task_id: uuid.UUID,
    db: Session = Depends(get_db),
    actor: str = Depends(require_api_token),
) -> MergeResult:
    """Gated squash-merge of the task's approved draft PR.

    Bearer-token gated. Returns ``merged=True`` + merge-commit SHA on
    success, or ``merged=False`` + a specific ``denial_reason`` for
    business-logic failures (not approved, repo not allowlisted, stale PR,
    already merged, or a GitHub API error).

    The ONLY precondition the endpoint enforces itself is approval: the task
    must have an ``approvals`` row with ``action="approve"`` (checked
    independently of the tool for defense in depth). All other gates — repo
    allowlist, staleness check, actual merge — are enforced inside the
    ``github.merge_pr`` tool through the pipeline, preserving the
    ``validate → capability → policy → execute → audit`` contract.

    The tool returns structured output for business denials (EXECUTED with
    ``merged=False``) rather than raising; the endpoint writes an
    ``AuditLog`` row in all three outcomes (merged / denied / failed) to
    keep the audit trail complete. HTTP status is always 200 except for
    the precondition error (409 when not approved), matching the pre-existing
    approve/reject pattern.
    """
    # 1. Row-lock + 404 — same pattern as approve/reject.
    task = db.execute(
        select(Task).where(Task.id == task_id).with_for_update()
    ).scalar_one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    # 2. Precondition: must already be approved (409 if not).
    latest_approval = db.scalar(
        select(Approval)
        .where(Approval.task_id == task.id)
        .order_by(Approval.created_at.desc(), Approval.id.desc())
        .limit(1)
    )
    if latest_approval is None or latest_approval.action != "approve":
        raise HTTPException(
            status_code=409,
            detail=("task not approved — call POST /tasks/{id}/approve first"),
        )

    # 3. Invoke github.merge_pr through the full pipeline (no shortcuts).
    ctx = ExecutionContext(task_id=task.id, agent_type="github", db=db)
    caps = set(capabilities_for_agent("github"))
    result = await ToolPipeline(db).invoke(
        "github.merge_pr",
        {"task_id": str(task.id)},
        caps,
        ctx,
    )

    if result.status == "EXECUTED" and result.output is not None:
        merged_flag = result.output.get("merged")
        merge_sha = result.output.get("merge_commit_sha")
        denial = result.output.get("denial_reason")
        pr_number = result.output.get("pr_number")
        repo = result.output.get("repo")

        if merged_flag:
            db.add(
                AuditLog(
                    task_id=task.id,
                    actor=actor,
                    action="task.merged",
                    entity_type="pull_request",
                    entity_id=str(pr_number) if pr_number else None,
                    details={
                        "merge_commit_sha": merge_sha,
                        "repo": repo,
                        "pr_number": pr_number,
                    },
                )
            )
        else:
            db.add(
                AuditLog(
                    task_id=task.id,
                    actor=actor,
                    action="task.merge_denied",
                    entity_type="task",
                    entity_id=str(task.id),
                    details={"reason": denial},
                )
            )
    else:
        # Pipeline FAILED (GitHub API error — raises → pipeline records FAILED).
        db.add(
            AuditLog(
                task_id=task.id,
                actor=actor,
                action="task.merge_failed",
                entity_type="task",
                entity_id=str(task.id),
                details={"error": result.error or "unknown"},
            )
        )

    db.commit()

    merged_flag = bool(result.output.get("merged") is True) if result.output else False
    return MergeResult(
        merged=merged_flag,
        merge_commit_sha=(
            result.output.get("merge_commit_sha") if result.output else None
        ),
        denial_reason=(
            (result.output.get("denial_reason") if result.output else None)
            or result.error
        ),
    )


@router.get("/{task_id}/events", response_model=list[ExecutionEventRead])
def list_events(
    task_id: uuid.UUID, db: Session = Depends(get_db)
) -> list[ExecutionEvent]:
    """Execution-event trail for a task, oldest first."""
    task = db.get(Task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    # Shared with the trace viewer (Phase 11) — one query implementation.
    return list_execution_events(db, task_id)
