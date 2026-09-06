"""Derived recommendation signal for the operator UI (Phase 12).

Read-only derivation from the Reviewer + Security agents' PERSISTED verdicts
that already gate the pipeline — nothing new is invented here, there is no
LLM call, and no agent is involved. The verdicts' meaning:

- ``ReviewResult.decision`` — APPROVE / REQUEST_CHANGES / REJECT.
- ``SecurityResult.decision`` — PASS / FAIL.

``recommendation_for`` maps both latest verdicts to the simple
operator-facing signal ``("merge", reason)`` or ``("hold", reason)``. A task
whose Reviewer approved AND Security passed is recommended for merge;
anything else is a hold. The signal is advisory only — the approvals table
(GET via ``POST /tasks/{id}/approve``) and the gated merge tool remain the
authoritative gates.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ReviewResult, SecurityResult, Task


def recommendation_for(db: Session, task: Task) -> tuple[str, str]:
    """``(recommended_action, recommendation_reason)`` derived from verdicts.

    Returns ``("hold", ...)`` when either agent has not run yet (no verdict
    to trust) — fail closed on absence, same posture as the pipeline.
    """
    review = db.scalar(
        select(ReviewResult)
        .where(ReviewResult.task_id == task.id)
        .order_by(ReviewResult.created_at.desc(), ReviewResult.id.desc())
        .limit(1)
    )
    security = db.scalar(
        select(SecurityResult)
        .where(SecurityResult.task_id == task.id)
        .order_by(SecurityResult.created_at.desc(), SecurityResult.id.desc())
        .limit(1)
    )

    if review is None or security is None:
        missing = []
        if review is None:
            missing.append("no review verdict yet")
        if security is None:
            missing.append("no security verdict yet")
        return "hold", "; ".join(missing)

    review_line = (
        f"Reviewer {review.decision} (severity {review.severity}, "
        f"{len(review.issues or [])} issue(s))"
    )
    security_line = (
        f"Security {security.decision} ({len(security.findings or [])} finding(s))"
    )

    if review.decision == "APPROVE" and security.decision == "PASS":
        return "merge", f"{review_line}; {security_line}"
    return "hold", f"{review_line}; {security_line}"
