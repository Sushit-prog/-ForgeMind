"""Phase 8 lifecycle tests: TESTING and DEBUGGING branch for real.

Drives ``advance_task_with_agents`` with REAL agents (tester runs real
pytest subprocesses; debugger + developer use mocked LLM providers) and
verifies the branching the Test/Debugger phases make real:

    TESTING:  passed -> REVIEWING | failed/error -> DEBUGGING
    DEBUGGING: flaky -> REVIEWING | unfixable -> FAILED
              | fixable -> IMPLEMENTING (replan_count+1, budget checked)

Including the full realistic loop IMPLEMENTING -> TESTING -> DEBUGGING ->
IMPLEMENTING -> TESTING -> REVIEWING (fail once, debug, fix, pass).
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
from app.agents.tester.agent import TestAgent
from app.git.runner import run_git
from app.llm import StubLLMProvider
from app.llm.mock import (
    DEFAULT_PLAN_RESPONSE,
    FINAL_PROPOSAL,
    RESEARCH_ARTIFACT_RESPONSE,
    SEARCH_PROPOSAL,
)
from app.models import (
    FailureClassification as ClassificationRow,
    Task,
    TaskStatus,
    TestRun,
)
from app.runtime.task_lifecycle import advance_task_with_agents, transition_task

# --- canned proposal fragments ----------------------------------------------

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
CODE_FAILURE = json.dumps(
    {"category": "CODE_FAILURE",
     "root_cause": "VALUE does not match the test expectation.",
     "fix_instruction": "Update src/app.py so VALUE equals 3.",
     "fixable": True}
)
ENV_FAILURE = json.dumps(
    {"category": "ENVIRONMENT_FAILURE",
     "root_cause": "integration services unreachable.",
     "fix_instruction": None, "fixable": False}
)


def run(coro):
    return asyncio.run(coro)


def make_repo(tmp_path, *, test_asserts: str) -> Path:
    """A repo whose test asserts ``test_asserts`` is IN src/app.py content."""
    repo = tmp_path / "phase8repo"
    repo.mkdir()
    run_git(repo, "init", "-b", "main")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("VALUE = 1\n")
    (repo / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = ['tests']\n"
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_suite.py").write_text(
        "from pathlib import Path\n\n"
        "def test_value():\n"
        f"    assert {test_asserts!r} in Path('src/app.py').read_text()\n"
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


def agents(developer_queue=None, classification=CODE_FAILURE):
    """Fresh agent set for one lifecycle drive. The developer's proposal
    queue defaults to a single write->commit->final cycle."""
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
    debugger = DebuggerAgent(
        StubLLMProvider(
            by_schema={
                "ToolCallProposal": [READ_APP, FINAL_PROPOSAL],
                "FailureClassification": [classification],
            }
        )
    )
    return planner, researcher, developer, debugger


def drive(db_session, task: Task, a) -> TaskStatus | None:
    planner, researcher, developer, debugger = a
    return run(
        advance_task_with_agents(
            db_session, task.id,
            planner=planner, researcher=researcher, developer=developer,
            tester=TestAgent(), debugger=debugger,
        )
    )


def test_testing_passed_routes_to_reviewing(db_session, tmp_path) -> None:
    """TESTING with a PASSING suite (developer wrote VALUE=2, test wants 2)
    routes to REVIEWING — the happy path through a real subprocess."""
    repo = make_repo(tmp_path, test_asserts="VALUE = 2")
    task = make_task(db_session, repo)
    a = agents()

    assert drive(db_session, task, a) is TaskStatus.PLANNING  # stub? no: PLANNING is real
    db_session.expire_all()
    task = db_session.get(Task, task.id)

    # Walk the pipeline until REVIEWING (the happy path destination).
    status = drive(db_session, task, a)
    db_session.expire_all()
    task = db_session.get(Task, task.id)
    assert status == TaskStatus.RESEARCHING
    status = drive(db_session, task, a)
    db_session.expire_all()
    task = db_session.get(Task, task.id)
    assert status == TaskStatus.IMPLEMENTING
    status = drive(db_session, task, a)
    db_session.expire_all()
    task = db_session.get(Task, task.id)
    assert status == TaskStatus.TESTING
    status = drive(db_session, task, a)
    assert status == TaskStatus.REVIEWING
    # TESTING -> REVIEWING with reason tests_passed.
    from app.models import ExecutionEvent

    events = db_session.scalars(
        select(ExecutionEvent).where(ExecutionEvent.task_id == task.id)
    ).all()
    testing_events = [e for e in events if e.from_status == "TESTING"]
    assert testing_events[-1].to_status == "REVIEWING"
    assert testing_events[-1].reason == "tests_passed"
    # No classification rows — nothing failed.
    assert not db_session.scalars(
        select(ClassificationRow).where(ClassificationRow.task_id == task.id)
    ).all()


def test_testing_failed_routes_to_debugging(db_session, tmp_path) -> None:
    """TESTING with a FAILING suite (developer wrote VALUE=2, test wants 3)
    routes to DEBUGGING — the failure branch, driven by a real failing exit
    code."""
    repo = make_repo(tmp_path, test_asserts="VALUE = 3")
    task = make_task(db_session, repo)
    a = agents()

    status = None
    for _ in range(6):  # walk: planning -> ... -> testing
        status = drive(db_session, task, a)
        db_session.expire_all()
        task = db_session.get(Task, task.id)
        if status == TaskStatus.DEBUGGING:
            break
    assert status == TaskStatus.DEBUGGING
    # The failing run is persisted as a TestRun with status failed.
    runs = db_session.scalars(
        select(TestRun).where(TestRun.task_id == task.id)
    ).all()
    assert runs[-1].status == "failed"


def test_full_loop_fail_once_debug_then_pass(db_session, tmp_path) -> None:
    """The realistic full loop: IMPLEMENTING -> TESTING(fail) -> DEBUGGING ->
    IMPLEMENTING(replan, fix instruction applied) -> TESTING(pass) ->
    REVIEWING. The developer's second run writes the FIXED value (3) because
    the debugger's fix instruction is handed to it as DATA."""
    repo = make_repo(tmp_path, test_asserts="VALUE = 3")
    task = make_task(db_session, repo)
    # Developer: run 1 writes 2 (fails), run 2 writes 3 (passes).
    a = agents(developer_queue=[WRITE_2, COMMIT, FINAL_PROPOSAL, WRITE_3, COMMIT, FINAL_PROPOSAL])

    status = drive(db_session, task, a)  # CREATED -> PLANNING (stub)
    db_session.expire_all()
    task = db_session.get(Task, task.id)
    assert status == TaskStatus.PLANNING

    status = drive(db_session, task, a)  # PLANNING -> RESEARCHING
    db_session.expire_all()
    task = db_session.get(Task, task.id)
    assert status == TaskStatus.RESEARCHING

    status = drive(db_session, task, a)  # -> IMPLEMENTING
    db_session.expire_all()
    task = db_session.get(Task, task.id)
    assert status == TaskStatus.IMPLEMENTING

    status = drive(db_session, task, a)  # implement (VALUE=2) -> TESTING
    db_session.expire_all()
    task = db_session.get(Task, task.id)
    assert status == TaskStatus.TESTING

    status = drive(db_session, task, a)  # test fails -> DEBUGGING
    db_session.expire_all()
    task = db_session.get(Task, task.id)
    assert status == TaskStatus.DEBUGGING
    assert task.replan_count == 0

    status = drive(db_session, task, a)  # debug (CODE_FAILURE, fixable) -> IMPLEMENTING
    db_session.expire_all()
    task = db_session.get(Task, task.id)
    assert status == TaskStatus.IMPLEMENTING
    assert task.replan_count == 1  # the debugger replan incremented it

    # The second implementation ran with the fix instruction in its prompt.
    from app.llm.mock import FIX_INSTRUCTION

    status = drive(db_session, task, a)  # implement (VALUE=3) -> TESTING
    db_session.expire_all()
    task = db_session.get(Task, task.id)
    assert status == TaskStatus.TESTING

    status = drive(db_session, task, a)  # test passes -> REVIEWING
    assert status == TaskStatus.REVIEWING

    db_session.expire_all()
    task = db_session.get(Task, task.id)
    assert task.replan_count == 1  # no further replans

    from app.models import ExecutionEvent

    events = db_session.scalars(
        select(ExecutionEvent).where(ExecutionEvent.task_id == task.id)
    ).all()
    trail = [(e.from_status, e.to_status) for e in events]
    assert ("DEBUGGING", "IMPLEMENTING") in trail
    assert ("TESTING", "DEBUGGING") in trail
    testing_events = [e for e in events if e.from_status == "TESTING"]
    assert testing_events[-1].reason == "tests_passed"

    # Two real test runs happened (the debugger's flakiness re-run failed the
    # same way — not flaky), then a passing run.
    runs = db_session.scalars(
        select(TestRun).where(TestRun.task_id == task.id)
    ).all()
    assert [r.status for r in runs] == ["failed", "failed", "passed"]

    # The classification was persisted with the concrete fix instruction.
    rows = db_session.scalars(
        select(ClassificationRow).where(ClassificationRow.task_id == task.id)
    ).all()
    assert rows[-1].fix_instruction == FIX_INSTRUCTION


def test_debugging_flaky_routes_to_reviewing(db_session, tmp_path) -> None:
    """A flaky failure (re-run passes) routes to REVIEWING — never IMPLEMENTING
    — and the flaky classification is persisted, never swept away."""
    repo = tmp_path / "flakyrepo"
    repo.mkdir()
    run_git(repo, "init", "-b", "main")
    (repo / "src").mkdir()
    # VALUE = 1 initially, so the developer's write to 2 creates a REAL diff
    # (writing identical content would make git.commit refuse the empty
    # commit and hard-fail the developer).
    (repo / "src" / "app.py").write_text("VALUE = 1\n")
    (repo / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = ['tests']\n"
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_suite.py").write_text(
        "from pathlib import Path\n\n"
        "def test_flaky():\n"
        "    marker = Path('flaky-marker.txt')\n"
        "    if not marker.exists():\n"
        "        marker.write_text('done')\n"
        "        raise AssertionError('first run fails; re-run passes')\n"
    )
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-m", "initial")
    task = make_task(db_session, repo)
    a = agents(developer_queue=[WRITE_2, COMMIT, FINAL_PROPOSAL])

    status = None
    for _ in range(6):
        status = drive(db_session, task, a)
        db_session.expire_all()
        task = db_session.get(Task, task.id)
        if status in (TaskStatus.REVIEWING, TaskStatus.DEBUGGING):
            break
    assert status == TaskStatus.DEBUGGING
    status = drive(db_session, task, a)
    assert status == TaskStatus.REVIEWING  # flaky -> REVIEWING, not IMPLEMENTING

    rows = db_session.scalars(
        select(ClassificationRow).where(ClassificationRow.task_id == task.id)
    ).all()
    assert rows[-1].category == "FLAKY_TEST"
    assert rows[-1].is_flaky is True
    # Never routed back to IMPLEMENTING.
    db_session.expire_all()
    task = db_session.get(Task, task.id)
    assert task.replan_count == 0


def test_debugging_unfixable_fails_task(db_session, tmp_path) -> None:
    """An ENVIRONMENT_FAILURE (fixable=False) fails the task with the category
    attached — re-running the Developer cannot fix the environment."""
    repo = make_repo(tmp_path, test_asserts="VALUE = 3")
    task = make_task(db_session, repo)
    a = agents(classification=ENV_FAILURE)

    status = None
    for _ in range(6):
        status = drive(db_session, task, a)
        db_session.expire_all()
        task = db_session.get(Task, task.id)
        if status in (TaskStatus.DEBUGGING, TaskStatus.FAILED):
            break
    assert status == TaskStatus.DEBUGGING
    status = drive(db_session, task, a)
    assert status == TaskStatus.FAILED

    from app.models import ExecutionEvent

    events = db_session.scalars(
        select(ExecutionEvent).where(ExecutionEvent.task_id == task.id)
    ).all()
    debugging_events = [e for e in events if e.from_status == "DEBUGGING"]
    assert debugging_events[-1].to_status == "FAILED"
    assert debugging_events[-1].reason == "unfixable:ENVIRONMENT_FAILURE"
    # The task never replanned.
    db_session.expire_all()
    task = db_session.get(Task, task.id)
    assert task.replan_count == 0


def test_debugging_replan_budget_exhausted_escalates(db_session, tmp_path) -> None:
    """max_replans enforcement at the TRANSITION: with budget 0, the fixable
    classification escalates immediately instead of replanning again."""
    repo = make_repo(tmp_path, test_asserts="VALUE = 3")
    task = make_task(db_session, repo)
    task.max_replans = 0
    db_session.commit()
    a = agents()

    status = None
    for _ in range(6):
        status = drive(db_session, task, a)
        db_session.expire_all()
        task = db_session.get(Task, task.id)
        if status in (TaskStatus.DEBUGGING, TaskStatus.ESCALATED):
            break
    assert status == TaskStatus.DEBUGGING
    status = drive(db_session, task, a)
    assert status == TaskStatus.ESCALATED

    from app.models import ExecutionEvent

    events = db_session.scalars(
        select(ExecutionEvent).where(ExecutionEvent.task_id == task.id)
    ).all()
    debugging_events = [e for e in events if e.from_status == "DEBUGGING"]
    assert debugging_events[-1].to_status == "ESCALATED"
    assert debugging_events[-1].reason == "replan_budget_exhausted"
    db_session.expire_all()
    task = db_session.get(Task, task.id)
    assert task.status == TaskStatus.ESCALATED.value
    assert task.replan_count == 0  # never incremented past the budget


def test_testing_without_tester_fails_cleanly(db_session, tmp_path) -> None:
    """TESTING with no tester agent fails cleanly instead of hanging."""
    repo = make_repo(tmp_path, test_asserts="VALUE = 2")
    task = make_task(db_session, repo)
    # Walk legally to TESTING (CREATED -> TESTING is not a legal jump).
    for target in (
        TaskStatus.PLANNING,
        TaskStatus.RESEARCHING,
        TaskStatus.IMPLEMENTING,
        TaskStatus.TESTING,
    ):
        transition_task(db_session, task, target)
    db_session.commit()

    from app.agents.planner.agent import PlanningAgent

    status = run(
        advance_task_with_agents(
            db_session, task.id, planner=PlanningAgent(StubLLMProvider())
        )
    )
    assert status == TaskStatus.FAILED
