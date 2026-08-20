"""Read-only trace assembly for the execution trace viewer (Phase 11).

The trace endpoint renders a human-readable journey for one task. Every
piece of data it shows already exists in the database — this module is the
single query/formatting layer, so the JSON events endpoint and the HTML
viewer never drift: ``list_execution_events`` is the ONE implementation of
the events query (used by ``GET /tasks/{id}/events``), and
``build_task_trace`` wraps it with everything the template needs.

``build_task_trace`` merges the events' transition spine with the artifacts
each phase persisted (plan DAG, research artifact, implementation summary,
tool calls, test runs, review/security verdicts, PR, human approval) into a
single chronological timeline. All values are passed as plain data; the
Jinja2 template autoescapes every string, so LLM-derived text can never be
interpreted as markup.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    Approval,
    ExecutionEvent,
    ImplementationSummary,
    Plan,
    PlanStep,
    PullRequest,
    Repository,
    ResearchArtifact,
    ReviewResult,
    SecurityResult,
    Task,
    TestRun,
    ToolCall,
)

# Terminal states never auto-advance — the trace stops auto-refreshing.
REFERENCE_TERMINAL = frozenset({"COMPLETED", "ESCALATED", "FAILED"})

_UB = datetime(1970, 1, 1, tzinfo=timezone.utc)


def list_execution_events(db: Session, task_id) -> list[ExecutionEvent]:
    """Execution-event trail for a task, oldest first.

    The single implementation of the events query — shared by the JSON
    ``GET /tasks/{id}/events`` route and the trace viewer.
    """
    return list(
        db.scalars(
            select(ExecutionEvent)
            .where(ExecutionEvent.task_id == task_id)
            .order_by(ExecutionEvent.created_at, ExecutionEvent.id)
        )
    )


def _ts(dt: datetime | None) -> str:
    """Human + machine readable timestamp (UTC, local render handled by CSS)."""
    return (dt or _UB).isoformat(timespec="seconds")


def _item(
    dt: datetime | None, kind: str, label: str, detail: str | None = None
) -> dict:
    return {
        "ts": _ts(dt),
        "kind": kind,
        "label": label,
        "detail": detail,
    }


def _safe_str(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def build_task_trace(db: Session, task: Task) -> dict:
    """Assemble everything the trace template needs for ``task``.

    Returns a plain-dict context (no ORM objects) so the route stays dumb
    and the template cannot reach back into the session.
    """
    task_id = task.id
    events = list_execution_events(db, task_id)

    repo = db.get(Repository, task.repository_id)
    pr = db.scalar(
        select(PullRequest)
        .where(PullRequest.task_id == task_id)
        .order_by(PullRequest.created_at.desc(), PullRequest.id.desc())
        .limit(1)
    )
    plan = db.scalar(
        select(Plan)
        .where(Plan.task_id == task_id)
        .order_by(Plan.created_at.desc(), Plan.id.desc())
        .limit(1)
    )

    timeline: list[dict] = [
        _item(e.created_at, "transition", f"{e.from_status} → {e.to_status}", e.reason)
        for e in events
    ]

    plan_card = None
    if plan is not None:
        steps = list(
            db.scalars(
                select(PlanStep)
                .where(PlanStep.plan_id == plan.id)
                .order_by(PlanStep.sequence)
            )
        )
        plan_card = {
            "status": plan.status,
            "steps": [
                {
                    "sequence": s.sequence,
                    "step_type": s.step_type,
                    "depends_on": str(s.depends_on) if s.depends_on else None,
                    "status": s.status,
                }
                for s in steps
            ],
        }
        timeline.append(
            _item(
                plan.created_at,
                "plan",
                "Plan created",
                f"{len(steps)} step(s); raw LLM output preserved for reproducibility.",
            )
        )

    for artifact in db.scalars(
        select(ResearchArtifact)
        .where(ResearchArtifact.task_id == task_id)
        .order_by(ResearchArtifact.created_at, ResearchArtifact.id)
    ):
        timeline.append(
            _item(
                artifact.created_at,
                "research",
                "Research complete",
                f"files: {', '.join(artifact.relevant_files or [])} — "
                f"confidence {artifact.confidence}.",
            )
        )

    for summary in db.scalars(
        select(ImplementationSummary)
        .where(ImplementationSummary.task_id == task_id)
        .order_by(ImplementationSummary.created_at, ImplementationSummary.id)
    ):
        files = ", ".join(summary.files_changed or [])
        detail = (
            f"commit {summary.commit_sha}" if summary.commit_sha else summary.status
        )
        timeline.append(
            _item(
                summary.created_at,
                "implement",
                "Implementation",
                f"{summary.status}: {detail}" + (f" — {files}" if files else ""),
            )
        )

    for call in db.scalars(
        select(ToolCall)
        .where(ToolCall.task_id == task_id)
        .order_by(ToolCall.created_at, ToolCall.id)
    ):
        detail = f"{call.status}"
        if call.status == "DENIED" and call.denial_reason:
            detail += f" — {call.denial_reason}"
        if call.risk:
            detail += f" (risk {call.risk})"
        timeline.append(
            _item(
                call.created_at,
                "tool",
                f"{call.agent_type or 'agent'}: {call.tool_name}",
                detail,
            )
        )

    for run in db.scalars(
        select(TestRun)
        .where(TestRun.task_id == task_id)
        .order_by(TestRun.created_at, TestRun.id)
    ):
        label = f"Test run: {run.status}"
        detail = (
            f"{run.passed} passed, {run.failed} failed, {run.duration_ms}ms "
            f"(exit {run.exit_code})" + (" — TIMED OUT" if run.timed_out else "")
        )
        timeline.append(_item(run.created_at, "test", label, detail))

    for review in db.scalars(
        select(ReviewResult)
        .where(ReviewResult.task_id == task_id)
        .order_by(ReviewResult.created_at, ReviewResult.id)
    ):
        timeline.append(
            _item(
                review.created_at,
                "review",
                f"Review: {review.decision}",
                f"severity {review.severity}; {len(review.issues or [])} issue(s).",
            )
        )

    for sec in db.scalars(
        select(SecurityResult)
        .where(SecurityResult.task_id == task_id)
        .order_by(SecurityResult.created_at, SecurityResult.id)
    ):
        timeline.append(
            _item(
                sec.created_at,
                "security",
                f"Security: {sec.decision}",
                f"{len(sec.findings or [])} finding(s).",
            )
        )

    if pr is not None:
        timeline.append(
            _item(
                pr.created_at,
                "pr",
                f"Draft PR #{pr.number} ({pr.status})",
                pr.url,
            )
        )

    for approval in db.scalars(
        select(Approval)
        .where(Approval.task_id == task_id)
        .order_by(Approval.created_at, Approval.id)
    ):
        timeline.append(
            _item(
                approval.created_at,
                "human",
                f"Human {approval.action}",
                approval.reason or None,
            )
        )

    timeline.sort(key=lambda item: (item["ts"], item["label"]))

    status = task.status
    safe = REFERENCE_TERMINAL
    return {
        "task_id": str(task.id),
        "objective": task.objective,
        "status": status,
        "created_at": _ts(task.created_at),
        "updated_at": _ts(task.updated_at),
        "repository_url": _safe_str(repo.url if repo else None),
        "issue_number": task.issue_number,
        "replan_count": task.replan_count,
        "pr": (
            {
                "url": pr.url,
                "number": pr.number,
                "status": pr.status,
            }
            if pr is not None
            else None
        ),
        "show_pr": pr is not None and status in {"AWAITING_APPROVAL", "COMPLETED"},
        "terminal_reason": (
            (events[-1].reason or events[-1].to_status)
            if events and status in safe
            else None
        ),
        "plan": plan_card,
        "timeline": timeline,
        "no_events": not timeline,
        "auto_refresh": status not in safe,
    }
