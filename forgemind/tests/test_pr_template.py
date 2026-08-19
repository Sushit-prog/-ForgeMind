"""PR template assembly (Phase 10) — plain code from persisted artifacts, NO LLM."""

from __future__ import annotations

from app.agents.github_agent.pr_template import build_pr_body, build_pr_title
from app.models import (
    ImplementationSummary,
    Plan,
    PlanStep,
    ResearchArtifact,
    ReviewResult,
    SecurityResult,
    Task,
    TestRun,
)
from app.runtime.task_lifecycle import transition_task
from app.runtime.state_machine import TaskStatus


def make_task(db_session, repo_task) -> Task:
    repo, task = repo_task
    task.issue_number = 42
    db_session.commit()
    db_session.refresh(task)
    return task


def test_body_includes_every_persisted_artifact_section(db_session, repo_task) -> None:
    task = make_task(db_session, repo_task)

    plan = Plan(task_id=task.id, status="ACTIVE")
    db_session.add(plan)
    db_session.flush()
    db_session.add_all(
        [
            PlanStep(plan_id=plan.id, step_type="research", sequence=1),
            PlanStep(plan_id=plan.id, step_type="implement", sequence=2),
        ]
    )
    db_session.add(
        ResearchArtifact(
            task_id=task.id,
            root_cause_hypothesis="VALUE is wrong",
            relevant_files=["src/app.py"],
            relevant_tests=["tests/test_app.py"],
            evidence=["Searched for VALUE."],
            confidence=0.8,
        )
    )
    db_session.add(
        ImplementationSummary(
            task_id=task.id,
            commit_sha="a" * 40,
            files_changed=["src/app.py"],
            summary="Set VALUE = 2",
            tests_added=["tests/test_value.py"],
            status="COMPLETE",
        )
    )
    db_session.add(
        TestRun(
            task_id=task.id,
            status="passed",
            passed=5,
            failed=0,
            duration_ms=42,
            exit_code=0,
        )
    )
    db_session.add(
        ReviewResult(
            task_id=task.id,
            commit_sha="a" * 40,
            decision="APPROVE",
            severity="low",
            issues=[],
        )
    )
    db_session.add(
        SecurityResult(
            task_id=task.id, commit_sha="a" * 40, decision="PASS", findings=[]
        )
    )
    db_session.commit()

    body = build_pr_body(task, db_session)

    assert "## Overview" in body
    assert "#42" in body  # source-issue link
    assert "## Plan" in body
    assert "`research`" in body
    assert "`implement`" in body
    assert "## Research" in body
    assert "VALUE is wrong" in body
    assert "src/app.py" in body
    assert "## Implementation" in body
    assert "Set VALUE = 2" in body
    assert "## Tests" in body
    assert "5 passed, 0 failed" in body
    assert "## Review" in body
    assert "**APPROVE**" in body
    assert "## Security" in body
    assert "**PASS**" in body
    assert "## Note" in body
    assert "draft" in body  # the human-read note


def test_absent_artifacts_omit_their_sections(db_session, repo_task) -> None:
    task = make_task(db_session, repo_task)
    body = build_pr_body(task, db_session)
    assert "## Overview" in body
    for section in (
        "## Plan",
        "## Research",
        "## Implementation",
        "## Tests",
        "## Review",
        "## Security",
    ):
        assert section not in body
    assert "## Note" in body  # the note is always present


def test_title_is_prefixed_and_truncated(db_session, repo_task) -> None:
    task = make_task(db_session, repo_task)
    task.objective = "Fix a super long objective " * 20
    db_session.commit()
    title = build_pr_title(task)
    assert title.startswith("ForgeMind: ")
    # "ForgeMind: " (11) + PR_TITLE_MAX (72) + possible ellipsis (1).
    assert len(title) <= 84


def test_no_llm_is_ever_involved(db_session, repo_task) -> None:
    """The template is pure Python over rows — proven by importing only data
    modules and by the absence of any provider import in the module."""
    import app.agents.github_agent.pr_template as t

    source = open(t.__file__, encoding="utf-8").read()
    assert "LLMProvider" not in source
    assert "from app.llm" not in source
