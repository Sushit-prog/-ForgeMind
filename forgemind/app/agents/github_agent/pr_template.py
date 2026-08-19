"""PR body assembly (Phase 10) — plain code, NO LLM.

The PR a human reads when deciding whether to approve must be a direct,
unembellished readout of what ForgeMind actually verified — not a fresh LLM
narrative that could drift from ground truth. Every section is assembled
from the PERSISTED artifacts of the phases that already ran (plan,
research, implementation, test, review, security). If an artifact is
absent the section is simply omitted; nothing is invented.

Sections:

1. Overview      — task objective (+ source-issue link, when present)
2. Plan          — the step descriptions of the ACTIVE plan
3. Research      — root-cause hypothesis + relevant files/tests + confidence
4. Implementation— the developer's summary + files changed + tests added
5. Tests         — the pass/fail counts of the last test run
6. Review        — the Reviewer's APPROVE decision + severity
7. Security      — the Security Agent's PASS decision
8. Note          — draft PR, awaiting human approval; merging is manual
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    ImplementationSummary,
    Plan,
    PlanStep,
    ResearchArtifact,
    ReviewResult,
    SecurityResult,
    TestRun,
    Task,
)

PR_TITLE_MAX = 72


def build_pr_title(task: Task) -> str:
    """``ForgeMind: <objective>`` truncated to a GitHub-friendly length."""
    text = " ".join(task.objective.split())
    clipped = text[:PR_TITLE_MAX]
    if len(text) > PR_TITLE_MAX:
        clipped = clipped.rstrip() + "…"
    return f"ForgeMind: {clipped}"


def build_pr_body(task: Task, db: Session) -> str:
    """Assemble the PR description from the persisted artifacts only."""
    sections: list[str] = []

    objective = " ".join(task.objective.split())
    if task.issue_number is not None:
        objective += f"\n\nSource issue: #{task.issue_number}"
    sections.append(f"## Overview\n\n{objective}")

    plan = _latest_active_plan(db, task.id)
    if plan is not None:
        steps = db.scalars(
            select(PlanStep)
            .where(PlanStep.plan_id == plan.id)
            .order_by(PlanStep.sequence)
        ).all()
        if steps:
            lines = [f"{i}. `{s.step_type}`" for i, s in enumerate(steps, 1)]
            sections.append("## Plan\n\n" + "\n".join(lines))

    artifact = db.scalar(
        select(ResearchArtifact)
        .where(ResearchArtifact.task_id == task.id)
        .order_by(ResearchArtifact.created_at.desc())
        .limit(1)
    )
    if artifact is not None:
        lines = [
            f"**Hypothesis:** {artifact.root_cause_hypothesis}",
            f"**Confidence:** {artifact.confidence}",
        ]
        if artifact.relevant_files:
            lines.append(f"**Relevant files:** {', '.join(artifact.relevant_files)}")
        if artifact.relevant_tests:
            lines.append(f"**Relevant tests:** {', '.join(artifact.relevant_tests)}")
        sections.append("## Research\n\n" + "\n".join(lines))

    summary = db.scalar(
        select(ImplementationSummary)
        .where(
            ImplementationSummary.task_id == task.id,
            ImplementationSummary.status == "COMPLETE",
        )
        .order_by(ImplementationSummary.created_at.desc())
        .limit(1)
    )
    if summary is not None:
        lines = [f"**Summary:** {summary.summary}"]
        if summary.files_changed:
            lines.append(f"**Files changed:** {', '.join(summary.files_changed)}")
        if summary.tests_added:
            lines.append(f"**Tests added:** {', '.join(summary.tests_added)}")
        if summary.deviations_from_research:
            lines.append(
                f"**Deviation from research:** {summary.deviations_from_research}"
            )
        sections.append("## Implementation\n\n" + "\n".join(lines))

    test_run = db.scalar(
        select(TestRun)
        .where(TestRun.task_id == task.id)
        .order_by(TestRun.created_at.desc(), TestRun.id.desc())
        .limit(1)
    )
    if test_run is not None:
        sections.append(
            "## Tests\n\n"
            f"Status: **{test_run.status}** "
            f"({test_run.passed} passed, {test_run.failed} failed, "
            f"{test_run.duration_ms}ms)"
        )

    review = db.scalar(
        select(ReviewResult)
        .where(ReviewResult.task_id == task.id)
        .order_by(ReviewResult.created_at.desc(), ReviewResult.id.desc())
        .limit(1)
    )
    if review is not None:
        sections.append(
            "## Review\n\n"
            f"Reviewer verdict: **{review.decision}** (severity: {review.severity}"
            f", {len(review.issues)} issue(s))"
        )

    security = db.scalar(
        select(SecurityResult)
        .where(SecurityResult.task_id == task.id)
        .order_by(SecurityResult.created_at.desc(), SecurityResult.id.desc())
        .limit(1)
    )
    if security is not None:
        sections.append(
            "## Security\n\n"
            f"Security verdict: **{security.decision}** "
            f"({len(security.findings)} finding(s))"
        )

    sections.append(
        "## Note\n\n"
        "This is a **draft** PR opened by ForgeMind. It is pending human "
        "approval; nothing has been merged and nothing will be merged by "
        "the system automatically — merging remains a manual action."
    )
    return "\n\n".join(sections)


def _latest_active_plan(db: Session, task_id) -> Plan | None:
    return db.scalar(
        select(Plan)
        .where(Plan.task_id == task_id, Plan.status == "ACTIVE")
        .order_by(Plan.created_at.desc())
        .limit(1)
    )


def build_pr_content(task: Task, db: Session) -> tuple[str, str]:
    """``(title, body)`` for the ``github.create_pr`` tool."""
    return build_pr_title(task), build_pr_body(task, db)
