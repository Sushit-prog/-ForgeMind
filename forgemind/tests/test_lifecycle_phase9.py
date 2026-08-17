"""Phase 9 lifecycle tests: REVIEWING and SECURITY_REVIEW branch for real,
and VERIFICATION is a code-only staleness check.

Drives ``advance_task_with_agents`` with REAL agents (tester runs real
pytest subprocesses; reviewer + security use mocked LLM providers):

    REVIEWING:      APPROVE -> SECURITY_REVIEW
                    REQUEST_CHANGES/REJECT -> IMPLEMENTING (shared replan)
    SECURITY_REVIEW: PASS -> VERIFICATION | FAIL -> IMPLEMENTING (shared)
    VERIFICATION:   reviewed commit still HEAD + last run passed -> PR_CREATION
                    stale -> TESTING (no LLM, plain code)

The replan budget is ONE shared budget (Section 42): replan_count
accumulates across Debugger- and Reviewer/Security-triggered replans and is
enforced at the transition.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from sqlalchemy import select

from app.agents.debugger.agent import DebuggerAgent
from app.agents.developer.agent import DeveloperAgent
from app.agents.planner.agent import PlanningAgent
from app.agents.researcher.agent import ResearchAgent
from app.agents.reviewer.agent import ReviewerAgent
from app.agents.security.agent import SecurityAgent
from app.agents.tester.agent import TestAgent
from app.git.operations import GitOperations
from app.git.runner import run_git
from app.llm import StubLLMProvider
from app.llm.mock import FINAL_PROPOSAL, RESEARCH_ARTIFACT_RESPONSE, SEARCH_PROPOSAL
from app.models import (
    ExecutionEvent,
    FailureClassification as ClassificationRow,
    ReviewResult as ReviewRow,
    SecurityResult as SecurityRow,
    Task,
    TaskStatus,
    TestRun,
)
from app.runtime.task_lifecycle import advance_task_with_agents, transition_task

# --- canned fragments -------------------------------------------------------

WRITE_2 = json.dumps(
    {"tool_call": {"tool": "filesystem.write_file",
                   "input": {"path": "src/app.py", "content": "VALUE = 2\n"}}}
)
WRITE_3 = json.dumps(
    {"tool_call": {"tool": "filesystem.write_file",
                   "input": {"path": "src/app.py", "content": "VALUE = 3\n"}}}
)
COMMIT = json.dumps(
    {"tool_call": {"tool": "git.commit", "input": {"message": "fix"}}}
)
SUMMARY = json.dumps(
    {"files_changed": ["src/app.py"], "summary": "updated VALUE",
     "tests_added": [], "deviations_from_research": None}
)
READ_APP = json.dumps(
    {"tool_call": {"tool": "repository.read_file", "input": {"path": "src/app.py"}}}
)
APPROVE = json.dumps({"decision": "APPROVE", "issues": [], "severity": "low"})
REQUEST_CHANGES = json.dumps(
    {"decision": "REQUEST_CHANGES",
     "issues": [{"description": "Use VALUE = 3 instead of 2.",
                 "severity": "medium", "file": "src/app.py", "line": 1}],
     "severity": "medium"}
)
SECURITY_PASS = json.dumps({"decision": "PASS", "findings": []})
SECURITY_FAIL = json.dumps(
    {"decision": "FAIL",
     "findings": [{"category": "SECRETS", "file": "src/app.py", "line": 1,
                   "description": "Prefer VALUE = 3.",
                   "severity": "medium"}]}
)


def run(coro):
    return asyncio.run(coro)


def make_repo(tmp_path, *, accepts: tuple[str, ...] = ("VALUE = 2",)) -> Path:
    """A repo whose test passes when the file contains ANY of ``accepts``.

    Reviewer/security replans change the value (2 -> 3), so the test must
    accept both or the re-run would fail into DEBUGGING instead of reaching
    REVIEWING again.
    """
    repo = tmp_path / "phase9repo"
    repo.mkdir()
    run_git(repo, "init", "-b", "main")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("VALUE = 1\n")
    (repo / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = ['tests']\n"
    )
    (repo / "tests").mkdir()
    # The test passes when ANY accepted value is present (OR, not AND — the
    # point is that both the first commit and the post-replan commit pass).
    conditions = " or ".join(f"{a!r} in content" for a in accepts)
    (repo / "tests" / "test_suite.py").write_text(
        "from pathlib import Path\n\n"
        "def test_value():\n"
        "    content = Path('src/app.py').read_text()\n"
        f"    assert {conditions}\n"
    )
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-m", "initial")
    return repo


def make_task(db_session, repo_path: Path) -> Task:
    from app.models import Repository

    repo = Repository(url=str(repo_path), default_branch="main")
    db_session.add(repo)
    db_session.flush()
    task = Task(objective="fix the failing test", repository_id=repo.id)
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)
    return task


def agents(developer_queue=None, review_queue=None, security_queue=None):
    """Fresh agent set for one lifecycle drive. The developer's proposal
    queue defaults to a single write->commit->final cycle; reviewer/security
    default to read->final with their verdicts from the queues."""
    planner = PlanningAgent(StubLLMProvider())
    researcher = ResearchAgent(
        StubLLMProvider(
            by_schema={
                "ToolCallProposal": [SEARCH_PROPOSAL, FINAL_PROPOSAL],
                "ResearchArtifact": [RESEARCH_ARTIFACT_RESPONSE],
            }
        )
    )
    developer = DeveloperAgent(
        StubLLMProvider(
            by_schema={
                "ToolCallProposal": developer_queue
                or [WRITE_2, COMMIT, FINAL_PROPOSAL],
                "ImplementationSummaryDraft": [SUMMARY],
            }
        )
    )
    reviewer = ReviewerAgent(
        StubLLMProvider(
            by_schema={
                "ToolCallProposal": [READ_APP, FINAL_PROPOSAL],
                "ReviewResult": review_queue or [APPROVE],
            }
        )
    )
    security = SecurityAgent(
        StubLLMProvider(
            by_schema={
                "ToolCallProposal": [READ_APP, FINAL_PROPOSAL],
                "SecurityResult": security_queue or [SECURITY_PASS],
            }
        )
    )
    return planner, researcher, developer, TestAgent(), DebuggerAgent(
        StubLLMProvider(
            by_schema={
                "ToolCallProposal": [READ_APP, FINAL_PROPOSAL],
                "FailureClassification": [json.dumps(
                    {"category": "CODE_FAILURE",
                     "root_cause": "x", "fix_instruction": "use VALUE = 3",
                     "fixable": True}
                )],
            }
        )
    ), reviewer, security


def drive(db_session, task: Task, a) -> TaskStatus | None:
    planner, researcher, developer, tester, debugger, reviewer, security = a
    return run(
        advance_task_with_agents(
            db_session, task.id,
            planner=planner, researcher=researcher, developer=developer,
            tester=tester, debugger=debugger, reviewer=reviewer, security=security,
        )
    )


def events_for(db_session, task_id: uuid.UUID) -> list[tuple[str, str]]:
    events = db_session.scalars(
        select(ExecutionEvent).where(ExecutionEvent.task_id == task_id)
        .order_by(ExecutionEvent.created_at, ExecutionEvent.id)
    ).all()
    return [(e.from_status, e.to_status) for e in events]


def walk_to(db_session, task, a, target: TaskStatus) -> TaskStatus | None:
    status = None
    for _ in range(10):
        status = drive(db_session, task, a)
        db_session.expire_all()
        task = db_session.get(Task, task.id)
        if status is target or status in (TaskStatus.FAILED, TaskStatus.ESCALATED):
            break
    return status


# --- review reject-then-approve --------------------------------------------


def test_review_reject_then_approve_reaches_verification(db_session, tmp_path) -> None:
    """The full REVIEWING loop: developer commits VALUE=2, test passes,
    Reviewer REQUEST_CHANGES -> IMPLEMENTING (replan #1, issues labeled as
    the fix instruction), developer writes VALUE=3, test passes, Reviewer
    APPROVES -> SECURITY_REVIEW -> VERIFICATION -> PR_CREATION."""
    repo = make_repo(tmp_path, accepts=("VALUE = 2", "VALUE = 3"))
    task = make_task(db_session, repo)
    a = agents(
        developer_queue=[WRITE_2, COMMIT, FINAL_PROPOSAL, WRITE_3, COMMIT, FINAL_PROPOSAL],
        review_queue=[REQUEST_CHANGES, APPROVE],
    )

    status = None
    for _ in range(14):
        status = drive(db_session, task, a)
        db_session.expire_all()
        task = db_session.get(Task, task.id)
        if status is TaskStatus.PR_CREATION:
            break
    assert status == TaskStatus.PR_CREATION

    trail = events_for(db_session, task.id)
    assert ("REVIEWING", "IMPLEMENTING") in trail, trail
    assert ("REVIEWING", "SECURITY_REVIEW") in trail, trail
    assert ("SECURITY_REVIEW", "VERIFICATION") in trail, trail
    assert ("VERIFICATION", "PR_CREATION") in trail, trail

    db_session.expire_all()
    task = db_session.get(Task, task.id)
    assert task.replan_count == 1  # exactly one shared-budget replan

    # Two reviews persisted: REQUEST_CHANGES then APPROVE.
    reviews = db_session.scalars(
        select(ReviewRow).where(ReviewRow.task_id == task.id)
    ).all()
    assert [r.decision for r in reviews] == ["REQUEST_CHANGES", "APPROVE"]
    # The security verdict is PASS.
    securities = db_session.scalars(
        select(SecurityRow).where(SecurityRow.task_id == task.id)
    ).all()
    assert [s.decision for s in securities] == ["PASS"]


# --- security fail-then-pass -----------------------------------------------


def test_security_fail_then_pass_reaches_verification(db_session, tmp_path) -> None:
    """Reviewer approves, Security FAILs (planted finding) -> IMPLEMENTING
    replan with the SECURITY finding labeled as the fix instruction, then the
    fixed commit passes Security and verification."""
    repo = make_repo(tmp_path, accepts=("VALUE = 2", "VALUE = 3"))
    task = make_task(db_session, repo)
    a = agents(
        developer_queue=[WRITE_2, COMMIT, FINAL_PROPOSAL, WRITE_3, COMMIT, FINAL_PROPOSAL],
        security_queue=[SECURITY_FAIL, SECURITY_PASS],
    )

    from app.runtime.task_lifecycle import _latest_fix_instruction

    # Drive step-by-step; the moment the security FAIL replans, the latest
    # fix instruction must be the SECURITY finding (labeled, not merged).
    instr_after_fail = None
    status = None
    for _ in range(14):
        status = drive(db_session, task, a)
        db_session.expire_all()
        task = db_session.get(Task, task.id)
        if status is TaskStatus.IMPLEMENTING:
            instr_after_fail = _latest_fix_instruction(db_session, task.id)
        if status is TaskStatus.PR_CREATION:
            break
    assert status == TaskStatus.PR_CREATION

    trail = events_for(db_session, task.id)
    assert ("SECURITY_REVIEW", "IMPLEMENTING") in trail, trail
    assert ("SECURITY_REVIEW", "VERIFICATION") in trail, trail
    db_session.expire_all()
    task = db_session.get(Task, task.id)
    assert task.replan_count == 1

    securities = db_session.scalars(
        select(SecurityRow).where(SecurityRow.task_id == task.id)
    ).all()
    assert [s.decision for s in securities] == ["FAIL", "PASS"]
    # The fix instruction handed to the developer was labeled with SECURITY.
    assert instr_after_fail is not None, "no fix instruction captured after FAIL"
    assert instr_after_fail.startswith("[SECURITY]"), instr_after_fail


# --- shared replan budget across sources -----------------------------------


def test_replan_count_accumulates_across_sources(db_session, tmp_path) -> None:
    """A Reviewer replan followed by a Security replan: replan_count
    accumulates to 2 against the ONE shared budget — never reset between
    sources, never a second budget. (The Debugger draws from the SAME helper
    and budget — proven by Phase 8's full-loop test plus the shared
    ``_replan_to_implementing``; this test proves the Phase 9 sources share
    it too.)"""
    repo = make_repo(tmp_path, accepts=("VALUE = 2", "VALUE = 3"))
    task = make_task(db_session, repo)
    a = agents(
        developer_queue=[
            WRITE_2, COMMIT, FINAL_PROPOSAL,   # run 1 -> REVIEWING
            WRITE_3, COMMIT, FINAL_PROPOSAL,   # run 2 (reviewer replan)
            WRITE_2, COMMIT, FINAL_PROPOSAL,   # run 3 -> REVIEWING again
            WRITE_3, COMMIT, FINAL_PROPOSAL,   # run 4 (security replan)
            WRITE_2, COMMIT, FINAL_PROPOSAL,   # run 5 -> REVIEWING
            WRITE_3, COMMIT, FINAL_PROPOSAL,   # run 6 -> security passes
        ],
        review_queue=[REQUEST_CHANGES, APPROVE],
        security_queue=[SECURITY_FAIL, SECURITY_PASS],
    )

    status = None
    for _ in range(18):
        status = drive(db_session, task, a)
        db_session.expire_all()
        task = db_session.get(Task, task.id)
        if status is TaskStatus.PR_CREATION:
            break
    assert status == TaskStatus.PR_CREATION

    db_session.expire_all()
    task = db_session.get(Task, task.id)
    assert task.replan_count == 2  # reviewer + security, one budget
    trail = events_for(db_session, task.id)
    assert ("REVIEWING", "IMPLEMENTING") in trail, trail
    assert ("SECURITY_REVIEW", "IMPLEMENTING") in trail, trail
    assert ("VERIFICATION", "PR_CREATION") in trail, trail


# --- budget exhaustion at the transition ------------------------------------


def test_review_replan_budget_exhausted_escalates(db_session, tmp_path) -> None:
    """max_replans = 0: a REQUEST_CHANGES from the Reviewer escalates at the
    transition instead of replanning — the shared budget is enforced for the
    Reviewer exactly as for the Debugger (Phase 8)."""
    repo = make_repo(tmp_path, accepts=("VALUE = 2",))
    task = make_task(db_session, repo)
    task.max_replans = 0
    db_session.commit()
    a = agents(review_queue=[REQUEST_CHANGES])

    status = None
    for _ in range(8):
        status = drive(db_session, task, a)
        db_session.expire_all()
        task = db_session.get(Task, task.id)
        if status in (TaskStatus.ESCALATED, TaskStatus.FAILED):
            break
    assert status == TaskStatus.ESCALATED

    events = db_session.scalars(
        select(ExecutionEvent).where(ExecutionEvent.task_id == task.id)
    ).all()
    reviewing_events = [e for e in events if e.from_status == "REVIEWING"]
    assert reviewing_events[-1].to_status == "ESCALATED"
    assert reviewing_events[-1].reason == "replan_budget_exhausted"
    db_session.expire_all()
    task = db_session.get(Task, task.id)
    assert task.replan_count == 0


# --- VERIFICATION staleness check -------------------------------------------


def test_verification_passes_through_when_current(db_session, tmp_path) -> None:
    """The happy case: the reviewed commit IS still the worktree HEAD and the
    last test run passed -> PR_CREATION. Plain code, no LLM."""
    repo = make_repo(tmp_path, accepts=("VALUE = 2",))
    task = make_task(db_session, repo)
    a = agents()

    status = None
    for _ in range(10):
        status = drive(db_session, task, a)
        db_session.expire_all()
        task = db_session.get(Task, task.id)
        if status is TaskStatus.PR_CREATION:
            break
    assert status == TaskStatus.PR_CREATION
    events = db_session.scalars(
        select(ExecutionEvent).where(ExecutionEvent.task_id == task.id)
    ).all()
    verif = [e for e in events if e.from_status == "VERIFICATION"]
    assert verif[-1].to_status == "PR_CREATION"
    assert verif[-1].reason == "verification_passed"


def test_verification_catches_stale_commit(db_session, tmp_path) -> None:
    """The staleness check is real: a NEW commit lands on the worktree AFTER
    the review (simulating a replan racing in), so the reviewed commit is no
    longer HEAD. VERIFICATION routes back to TESTING — the approval is stale,
    never silently carried forward."""
    repo = make_repo(tmp_path, accepts=("VALUE = 2",))
    task = make_task(db_session, repo)
    # Walk legally to VERIFICATION (bypassing agent semantics — this test
    # only proves the code-only staleness check).
    for target in (
        TaskStatus.PLANNING,
        TaskStatus.RESEARCHING,
        TaskStatus.IMPLEMENTING,
        TaskStatus.TESTING,
        TaskStatus.REVIEWING,
        TaskStatus.SECURITY_REVIEW,
        TaskStatus.VERIFICATION,
    ):
        transition_task(db_session, task, target)
    db_session.commit()

    # A real implementation summary + passed test run exist (what the review
    # approved), then a NEW commit lands on the worktree.
    from app.git.worktree_manager import WorktreeManager
    from app.models import ImplementationSummary

    worktree = WorktreeManager(db_session).get_or_create_for_task(task)
    wt_path = Path(worktree.path)
    (wt_path / "src" / "app.py").write_text("VALUE = 2\n")
    run_git(wt_path, "add", "-A")
    run_git(wt_path, "commit", "-m", "dev: the change")
    reviewed_sha = GitOperations(wt_path).head_sha()
    db_session.add(
        ImplementationSummary(
            task_id=task.id, worktree_id=worktree.id, commit_sha=reviewed_sha,
            files_changed=["src/app.py"], summary="s", tests_added=[],
            deviations_from_research=None, status="COMPLETE",
        )
    )
    db_session.add(
        TestRun(task_id=task.id, worktree_id=worktree.id, status="passed",
                passed=1, failed=0, duration_ms=10, exit_code=0)
    )
    db_session.commit()

    # A stale commit lands AFTER the approval (HEAD moves past reviewed_sha).
    (wt_path / "src" / "app.py").write_text("VALUE = 3\n")
    run_git(wt_path, "add", "-A")
    run_git(wt_path, "commit", "-m", "unreviewed: landed after review")

    status = drive(db_session, task, agents())
    assert status == TaskStatus.TESTING  # stale -> re-test, never PR_CREATION

    events = db_session.scalars(
        select(ExecutionEvent).where(ExecutionEvent.task_id == task.id)
    ).all()
    verif = [e for e in events if e.from_status == "VERIFICATION"]
    assert verif[-1].to_status == "TESTING"
    assert verif[-1].reason == "stale_review"


def test_verification_catches_stale_test_result(db_session, tmp_path) -> None:
    """HEAD is still the reviewed commit, but the last test run is no longer
    passed -> stale -> TESTING (the passing approval was invalidated)."""
    repo = make_repo(tmp_path, accepts=("VALUE = 2",))
    task = make_task(db_session, repo)
    for target in (
        TaskStatus.PLANNING,
        TaskStatus.RESEARCHING,
        TaskStatus.IMPLEMENTING,
        TaskStatus.TESTING,
        TaskStatus.REVIEWING,
        TaskStatus.SECURITY_REVIEW,
        TaskStatus.VERIFICATION,
    ):
        transition_task(db_session, task, target)
    db_session.commit()

    from app.git.worktree_manager import WorktreeManager
    from app.models import ImplementationSummary

    worktree = WorktreeManager(db_session).get_or_create_for_task(task)
    wt_path = Path(worktree.path)
    (wt_path / "src" / "app.py").write_text("VALUE = 2\n")
    run_git(wt_path, "add", "-A")
    run_git(wt_path, "commit", "-m", "dev: the change")
    reviewed_sha = GitOperations(wt_path).head_sha()
    db_session.add(
        ImplementationSummary(
            task_id=task.id, worktree_id=worktree.id, commit_sha=reviewed_sha,
            files_changed=["src/app.py"], summary="s", tests_added=[],
            deviations_from_research=None, status="COMPLETE",
        )
    )
    # The LAST test run is a failure (a new run invalidated the passing one).
    db_session.add(
        TestRun(task_id=task.id, worktree_id=worktree.id, status="failed",
                passed=0, failed=1, duration_ms=10, exit_code=1)
    )
    db_session.commit()

    status = drive(db_session, task, agents())
    assert status == TaskStatus.TESTING
    events = db_session.scalars(
        select(ExecutionEvent).where(ExecutionEvent.task_id == task.id)
    ).all()
    verif = [e for e in events if e.from_status == "VERIFICATION"]
    assert verif[-1].reason == "stale_review"
