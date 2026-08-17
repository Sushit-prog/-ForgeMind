"""Reviewer Agent tests (Phase 9, architecture doc sections 11/E).

The Reviewer critiques the developer's commit from the git.diff observation
+ the test result ONLY. The single most important test here is the
INDEPENDENCE test: a plausible-sounding ImplementationSummary must not bias
the review, because the summary is structurally absent from the reviewer's
context (its run signature has no summary parameter and its prompt builder
has no summary field) — the review can only be wrong for diff-level
reasons, never for story-level reasons.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from app.agents.reviewer.agent import ReviewerAgent
from app.agents.reviewer.schema import ReviewIssue, ReviewResult
from app.agents.tester.schema import TestResult
from app.git.runner import run_git
from app.git.worktree_manager import WorktreeManager
from app.llm import StubLLMProvider
from app.llm.mock import FINAL_PROPOSAL
from app.models import AuditLog
from app.models import ImplementationSummary, Repository, ReviewResult as ReviewRow
from app.models import Task, TestRun
from app.tools.base import ExecutionContext


def run(coro):
    return asyncio.run(coro)


def make_commit_repo(tmp_path, *, app_content: str) -> Path:
    """A repo whose worktree has ONE developer commit on top of main."""
    repo = tmp_path / "reviewrepo"
    repo.mkdir()
    run_git(repo, "init", "-b", "main")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("VALUE = 1\n")
    (repo / "README.md").write_text("# review fixture\n")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-m", "initial")
    return repo


def repo_task_worktree_with_commit(db_session, repo_path: Path, app_content: str):
    """Task + worktree + ONE developer commit. Returns the commit sha."""
    from app.models import Repository

    repo = Repository(url=str(repo_path), default_branch="main")
    db_session.add(repo)
    db_session.flush()
    task = Task(objective="implement the change", repository_id=repo.id)
    db_session.add(task)
    db_session.commit()
    db_session.refresh(repo)
    db_session.refresh(task)
    worktree = WorktreeManager(db_session).create(task.id, repo)
    wt_path = Path(worktree.path)
    (wt_path / "src" / "app.py").write_text(app_content)
    run_git(wt_path, "add", "-A")
    run_git(wt_path, "commit", "-m", "dev: the change")
    sha = run_git(wt_path, "rev-parse", "HEAD").stdout.strip()
    return repo, task, worktree, sha


def fake_test_result(status: str = "passed") -> TestResult:
    return TestResult(status=status, passed=1, failed=0, duration_ms=10, exit_code=0)


def reviewer_with(by_schema: dict) -> ReviewerAgent:
    return ReviewerAgent(StubLLMProvider(by_schema=by_schema))


def diff_proposal(sha: str) -> str:
    """Propose git.diff for the REAL commit sha under review — the mock's
    canned proposal can't know the sha, so tests build it dynamically."""
    return json.dumps({"tool_call": {"tool": "git.diff", "input": {"commit": sha}}})
APPROVE_RESPONSE = json.dumps(
    {"decision": "APPROVE", "issues": [], "severity": "low"}
)
REJECT_RESPONSE = json.dumps(
    {
        "decision": "REQUEST_CHANGES",
        "issues": [
            {
                "description": "VALUE is hardcoded instead of configurable.",
                "severity": "high",
                "file": "src/app.py",
                "line": 1,
            }
        ],
        "severity": "high",
    }
)


# --- schema ----------------------------------------------------------------


def test_review_result_schema_validation() -> None:
    approve = ReviewResult(decision="APPROVE", issues=[], severity="low")
    assert approve.decision == "APPROVE"
    with pytest.raises(ValidationError):
        ReviewResult(decision="APPROVE", issues=[ReviewIssue(
            description="x", severity="low", file="f", line=1
        )], severity="low")  # APPROVE with issues is invalid
    with pytest.raises(ValidationError):
        ReviewResult(decision="REJECT", issues=[], severity="high")  # REJECT needs issues
    # line must be >= 1 (or absent), never 0.
    with pytest.raises(ValidationError):
        ReviewResult(
            decision="REQUEST_CHANGES",
            issues=[ReviewIssue(description="x", severity="medium", file="f", line=0)],
            severity="medium",
        )


# --- integration: approve on a clean diff ----------------------------------


def test_reviewer_approves_clean_diff(db_session, tmp_path) -> None:
    """A clean, well-scoped diff -> APPROVE. The loop ran git.diff against
    the REAL developer commit and the verdict is grounded in the diff."""
    repo_path = make_commit_repo(tmp_path, app_content="VALUE = 1\n")
    repo, task, worktree, sha = repo_task_worktree_with_commit(
        db_session, repo_path, "VALUE = 2\n"
    )
    reviewer = reviewer_with(
        {
            "ToolCallProposal": [diff_proposal(sha), FINAL_PROPOSAL],
            "ReviewResult": [APPROVE_RESPONSE],
        }
    )

    result = run(
        reviewer.run(task, sha, fake_test_result(), ExecutionContext(
            task_id=task.id, agent_type="reviewer", db=db_session
        ))
    )

    assert result.decision == "APPROVE"
    rows = db_session.scalars(
        select(ReviewRow).where(ReviewRow.task_id == task.id)
    ).all()
    assert len(rows) == 1
    assert rows[0].decision == "APPROVE"
    assert rows[0].commit_sha == sha


# --- integration: reject a deliberately bad diff ---------------------------


def test_reviewer_rejects_bad_diff(db_session, tmp_path) -> None:
    """A diff with a real problem -> REQUEST_CHANGES with a file-anchored
    issue. The loop saw the REAL diff (the observation), not a summary."""
    repo_path = make_commit_repo(tmp_path, app_content="VALUE = 1\n")
    repo, task, worktree, sha = repo_task_worktree_with_commit(
        db_session, repo_path, "import os; os.system('rm -rf /')\nVALUE = 2\n"
    )
    reviewer = reviewer_with(
        {
            "ToolCallProposal": [diff_proposal(sha), FINAL_PROPOSAL],
            "ReviewResult": [REJECT_RESPONSE],
        }
    )

    result = run(
        reviewer.run(task, sha, fake_test_result(), ExecutionContext(
            task_id=task.id, agent_type="reviewer", db=db_session
        ))
    )

    assert result.decision == "REQUEST_CHANGES"
    assert result.issues[0].file == "src/app.py"
    rows = db_session.scalars(
        select(ReviewRow).where(ReviewRow.task_id == task.id)
    ).all()
    assert rows[0].decision == "REQUEST_CHANGES"


# --- THE INDEPENDENCE TEST -------------------------------------------------


def test_reviewer_is_blind_to_developer_summary(db_session, tmp_path) -> None:
    """The most important test of this phase: a plausible-sounding
    ImplementationSummary must NOT bias the review.

    The diff contains a real problem (an unsafe subprocess call). The
    summary claims everything is perfect. A reviewer who SAW the summary
    might be swayed; this one cannot be, because the summary is structurally
    absent from its context. We prove absence directly: the LLM's prompts
    contain the diff observation and the summary text / deviations appear
    NOWHERE in any message."""
    repo_path = make_commit_repo(tmp_path, app_content="VALUE = 1\n")
    repo, task, worktree, sha = repo_task_worktree_with_commit(
        db_session, repo_path, "import os; os.system('rm -rf /')\nVALUE = 2\n"
    )

    # A summary that would bias a human favorably...
    biased = ImplementationSummary(
        task_id=task.id,
        commit_sha=sha,
        files_changed=["src/app.py"],
        summary="The change is perfect: refactored for clarity, zero risk, "
        "all edge cases handled, no regressions, highly configurable.",
        tests_added=["tests/test_app.py"],
        deviations_from_research=None,
        status="COMPLETE",
    )
    db_session.add(biased)
    db_session.commit()

    provider = StubLLMProvider(
        by_schema={
            "ToolCallProposal": [diff_proposal(sha), FINAL_PROPOSAL],
            "ReviewResult": [REJECT_RESPONSE],
        }
    )
    reviewer = ReviewerAgent(provider)
    ctx = ExecutionContext(task_id=task.id, agent_type="reviewer", db=db_session)

    result = run(reviewer.run(task, sha, fake_test_result(), ctx))

    # The reviewer STILL rejects — driven by the diff, not the story.
    assert result.decision == "REQUEST_CHANGES"

    # Proof of structural blindness: the summary text and its deviations
    # appear in NONE of the messages sent to the LLM.
    all_text = "\n".join(
        m.content for msgs in provider.structured_calls + provider.generate_calls
        for m in msgs
    )
    assert "perfect" not in all_text
    assert "highly configurable" not in all_text
    assert "deviations_from_research" not in all_text
    # ...but the diff observation (with the unsafe call) IS present.
    assert "os.system" in all_text
    assert "VALUE = 2" in all_text


# --- adversarial: write proposal denied + audited ---------------------------


def test_reviewer_write_proposal_denied_and_audited(db_session, tmp_path) -> None:
    """A write proposal by the read-only Reviewer is DENIED and audited as
    reviewer.unexpected_denial (Developer's pattern — no legitimate reason to
    probe write access). The loop survives and still produces a verdict."""
    repo_path = make_commit_repo(tmp_path, app_content="VALUE = 1\n")
    repo, task, worktree, sha = repo_task_worktree_with_commit(
        db_session, repo_path, "VALUE = 2\n"
    )
    write_proposal = json.dumps(
        {
            "tool_call": {
                "tool": "filesystem.write_file",
                "input": {"path": "src/app.py", "content": "VALUE = 3\n"},
            }
        }
    )
    reviewer = reviewer_with(
        {
            "ToolCallProposal": [write_proposal, diff_proposal(sha), FINAL_PROPOSAL],
            "ReviewResult": [APPROVE_RESPONSE],
        }
    )

    result = run(
        reviewer.run(task, sha, fake_test_result(), ExecutionContext(
            task_id=task.id, agent_type="reviewer", db=db_session
        ))
    )

    assert result.decision == "APPROVE"
    audits = db_session.scalars(
        select(AuditLog).where(AuditLog.task_id == task.id)
    ).all()
    denial = [a for a in audits if a.action == "reviewer.unexpected_denial"]
    assert len(denial) == 1
    assert denial[0].details["tool"] == "filesystem.write_file"
    assert denial[0].details["surfaced_as"] == "denied"
