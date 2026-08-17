"""Debugger Agent integration tests (Phase 8, architecture doc sections 10/E).

The Debugger investigates a REAL failing test run (read-only loop), re-runs
the suite EXACTLY ONCE via the Test Agent to OBSERVE flakiness rather than
guess it, and produces a persisted ``FailureClassification`` that drives the
lifecycle branching. These tests run real pytest subprocesses with a mocked
LLM for the classification.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from sqlalchemy import select

from app.agents.debugger.agent import DebuggerAgent, DebuggerError
from app.agents.debugger.schema import FailureClassification
from app.agents.tester.agent import TestAgent
from app.agents.tester.schema import TestResult, result_from_row
from app.git.runner import run_git
from app.git.worktree_manager import WorktreeManager
from app.llm import StubLLMProvider
from app.llm.mock import FINAL_PROPOSAL
from app.models import AuditLog, FailureClassification as ClassificationRow
from app.models import ImplementationSummary, Repository, Task, TestRun
from app.tools.base import ExecutionContext


def run(coro):
    return asyncio.run(coro)


def make_repo(tmp_path, *, app_content: str, tests: str) -> Path:
    repo = tmp_path / "bugrepo"
    repo.mkdir()
    run_git(repo, "init", "-b", "main")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text(app_content)
    (repo / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = ['tests']\n"
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_suite.py").write_text(tests)
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-m", "initial")
    return repo


def repo_task_worktree(db_session, repo_path: Path):
    from app.models import Repository

    repo = Repository(url=str(repo_path), default_branch="main")
    db_session.add(repo)
    db_session.flush()
    task = Task(objective="fix the failing test", repository_id=repo.id)
    db_session.add(task)
    db_session.commit()
    db_session.refresh(repo)
    db_session.refresh(task)
    worktree = WorktreeManager(db_session).create(task.id, repo)
    return repo, task, worktree


def fake_summary(task) -> ImplementationSummary:
    return ImplementationSummary(
        task_id=task.id,
        commit_sha="a" * 40,
        files_changed=["src/app.py"],
        summary="Changed VALUE per research.",
        tests_added=[],
        deviations_from_research=None,
        status="COMPLETE",
    )


def debugger_with(by_schema: dict) -> DebuggerAgent:
    return DebuggerAgent(StubLLMProvider(by_schema=by_schema))


READ_PROPOSAL = json.dumps(
    {"tool_call": {"tool": "repository.read_file", "input": {"path": "src/app.py"}}}
)
CODE_FAILURE_RESPONSE = json.dumps(
    {
        "category": "CODE_FAILURE",
        "root_cause": "VALUE does not match the test expectation.",
        "fix_instruction": "Update src/app.py so VALUE equals 3.",
        "fixable": True,
    }
)


def test_classifies_code_failure_with_fix_instruction(db_session, tmp_path) -> None:
    """A REAL code failure: the suite fails (VALUE=2 vs test expecting 3), the
    debugger investigates read-only, re-runs (fails the same way — not flaky),
    and classifies CODE_FAILURE with a concrete fix_instruction."""
    repo_path = make_repo(
        tmp_path,
        app_content="VALUE = 2\n",
        tests="from pathlib import Path\n\n"
        "def test_value():\n"
        "    assert 'VALUE = 3' in Path('src/app.py').read_text()\n",
    )
    repo, task, worktree = repo_task_worktree(db_session, repo_path)

    # First run: a real failing suite (the run DEBUGGING responds to).
    tester_ctx = ExecutionContext(
        task_id=task.id, agent_type="tester", db=db_session
    )
    first = run(TestAgent().run(task, worktree, tester_ctx))
    assert first.status == "failed"

    debugger = debugger_with(
        {
            "ToolCallProposal": [READ_PROPOSAL, FINAL_PROPOSAL],
            "FailureClassification": [CODE_FAILURE_RESPONSE],
        }
    )
    ctx = ExecutionContext(task_id=task.id, agent_type="debugger", db=db_session)
    classification = run(
        debugger.run(task, first, fake_summary(task), ctx)
    )

    assert isinstance(classification, FailureClassification)
    assert classification.category == "CODE_FAILURE"
    assert classification.fixable is True
    assert "src/app.py" in (classification.fix_instruction or "")

    # Persisted, linked to the first failing run.
    rows = db_session.scalars(
        select(ClassificationRow).where(ClassificationRow.task_id == task.id)
    ).all()
    assert len(rows) == 1
    assert rows[0].category == "CODE_FAILURE"
    assert rows[0].fixable is True
    assert rows[0].is_flaky is False

    # The re-run happened (2 TestRun rows total — the trace records both) and
    # the re-run's tool call is attributed to the TESTER.
    runs = db_session.scalars(
        select(TestRun).where(TestRun.task_id == task.id)
    ).all()
    assert len(runs) == 2
    assert runs[1].status == "failed"  # same failure mode -> not flaky

    # The investigation loop ran REAL read-only tool calls, audited.
    from app.models import ToolCall

    debugger_calls = db_session.scalars(
        select(ToolCall)
        .where(ToolCall.task_id == task.id, ToolCall.agent_type == "debugger")
        .order_by(ToolCall.created_at)
    ).all()
    assert [c.tool_name for c in debugger_calls] == ["repository.read_file"]
    assert all(c.status == "EXECUTED" for c in debugger_calls)


def test_read_only_boundary_denies_write_and_continues(db_session, tmp_path) -> None:
    """A write proposal (shouldn't happen, but verify) is DENIED + audited as
    debugger.unexpected_denial, and the loop keeps going — the denial is an
    observation, not a crash."""
    repo_path = make_repo(
        tmp_path,
        app_content="VALUE = 2\n",
        tests="from pathlib import Path\n\n"
        "def test_value():\n"
        "    assert 'VALUE = 3' in Path('src/app.py').read_text()\n",
    )
    repo, task, worktree = repo_task_worktree(db_session, repo_path)
    tester_ctx = ExecutionContext(
        task_id=task.id, agent_type="tester", db=db_session
    )
    first = run(TestAgent().run(task, worktree, tester_ctx))

    write_proposal = json.dumps(
        {
            "tool_call": {
                "tool": "filesystem.write_file",
                "input": {"path": "src/app.py", "content": "VALUE = 3\n"},
            }
        }
    )
    debugger = debugger_with(
        {
            "ToolCallProposal": [write_proposal, READ_PROPOSAL, FINAL_PROPOSAL],
            "FailureClassification": [CODE_FAILURE_RESPONSE],
        }
    )
    ctx = ExecutionContext(task_id=task.id, agent_type="debugger", db=db_session)
    classification = run(debugger.run(task, first, fake_summary(task), ctx))

    assert classification.category == "CODE_FAILURE"  # loop survived the denial
    audits = db_session.scalars(
        select(AuditLog).where(
            AuditLog.task_id == task.id,
            AuditLog.action == "debugger.unexpected_denial",
        )
    ).all()
    assert len(audits) == 1
    assert audits[0].details["tool"] == "filesystem.write_file"
    # The file was never touched.
    assert (Path(worktree.path) / "src" / "app.py").read_text() == "VALUE = 2\n"


def test_shell_proposal_denied_read_only_boundary(db_session, tmp_path) -> None:
    """The debugger holds NO shell capability — proposing ``shell.run_test``
    itself is denied + audited, and the loop continues (the flakiness re-run
    is the Test Agent's job, never a tool the debugger can call)."""
    repo_path = make_repo(
        tmp_path,
        app_content="VALUE = 2\n",
        tests="from pathlib import Path\n\n"
        "def test_value():\n"
        "    assert 'VALUE = 3' in Path('src/app.py').read_text()\n",
    )
    repo, task, worktree = repo_task_worktree(db_session, repo_path)
    tester_ctx = ExecutionContext(
        task_id=task.id, agent_type="tester", db=db_session
    )
    first = run(TestAgent().run(task, worktree, tester_ctx))

    # A well-formed proposal (valid worktree_id) so the CAPABILITY gate is
    # what denies it — the debugger holds no shell.test.
    shell_proposal = json.dumps(
        {
            "tool_call": {
                "tool": "shell.run_test",
                "input": {"worktree_id": str(uuid.uuid4())},
            }
        }
    )
    debugger = debugger_with(
        {
            "ToolCallProposal": [shell_proposal, READ_PROPOSAL, FINAL_PROPOSAL],
            "FailureClassification": [CODE_FAILURE_RESPONSE],
        }
    )
    ctx = ExecutionContext(task_id=task.id, agent_type="debugger", db=db_session)
    classification = run(debugger.run(task, first, fake_summary(task), ctx))

    assert classification.category == "CODE_FAILURE"  # loop survived the denial
    audits = db_session.scalars(
        select(AuditLog).where(
            AuditLog.task_id == task.id,
            AuditLog.action == "debugger.unexpected_denial",
        )
    ).all()
    assert any(a.details.get("tool") == "shell.run_test" for a in audits)


def test_flaky_detected_via_rerun_not_guess(db_session, tmp_path) -> None:
    """The ONLY legitimate flaky label: the single re-run PASSES. The suite is
    genuinely intermittent — first run fails (and creates a marker), re-run
    passes. Classified FLAKY_TEST deterministically, no LLM call involved."""
    repo_path = make_repo(
        tmp_path,
        app_content="VALUE = 2\n",
        tests="from pathlib import Path\n\n"
        "def test_flaky():\n"
        "    marker = Path('flaky-marker.txt')\n"
        "    if not marker.exists():\n"
        "        marker.write_text('done')\n"
        "        raise AssertionError('first run fails; re-run passes')\n",
    )
    repo, task, worktree = repo_task_worktree(db_session, repo_path)
    tester_ctx = ExecutionContext(
        task_id=task.id, agent_type="tester", db=db_session
    )
    first = run(TestAgent().run(task, worktree, tester_ctx))
    assert first.status == "failed"

    # The LLM is explicitly FORBIDDEN from classifying flaky — give it a
    # CODE_FAILURE response and verify the re-run overrides it.
    debugger = debugger_with(
        {
            "ToolCallProposal": [READ_PROPOSAL, FINAL_PROPOSAL],
            "FailureClassification": [CODE_FAILURE_RESPONSE],
        }
    )
    ctx = ExecutionContext(task_id=task.id, agent_type="debugger", db=db_session)
    classification = run(debugger.run(task, first, fake_summary(task), ctx))

    assert classification.category == "FLAKY_TEST"
    assert classification.is_flaky is True
    assert classification.fixable is False

    rows = db_session.scalars(
        select(ClassificationRow).where(ClassificationRow.task_id == task.id)
    ).all()
    assert rows[0].category == "FLAKY_TEST"
    assert rows[0].is_flaky is True

    # Loudly flagged in the trace.
    audits = db_session.scalars(
        select(AuditLog).where(
            AuditLog.task_id == task.id, AuditLog.action == "debugger.flaky_detected"
        )
    ).all()
    assert len(audits) == 1

    # The provider was NEVER consulted for the flaky label — the re-run's
    # pass is observed ground truth, not an LLM guess (Section 10).
    assert debugger.provider.structured_calls == []
    runs = db_session.scalars(
        select(TestRun).where(TestRun.task_id == task.id)
    ).all()
    assert len(runs) == 2
    assert runs[1].status == "passed"


def test_unfixable_classification_is_not_fixable(db_session, tmp_path) -> None:
    """ENVIRONMENT_FAILURE must come back fixable=False (the schema enforces
    it) so the lifecycle fails the task instead of replanning forever."""
    repo_path = make_repo(
        tmp_path,
        app_content="VALUE = 2\n",
        tests="from pathlib import Path\n\n"
        "def test_value():\n"
        "    assert 'VALUE = 3' in Path('src/app.py').read_text()\n",
    )
    repo, task, worktree = repo_task_worktree(db_session, repo_path)

    # Simulate an environment failure classification (schema-level guard).
    env_response = json.dumps(
        {
            "category": "ENVIRONMENT_FAILURE",
            "root_cause": "docker daemon not reachable for integration tests.",
            "fix_instruction": None,
            "fixable": False,
        }
    )
    debugger = debugger_with(
        {
            "ToolCallProposal": [FINAL_PROPOSAL],
            "FailureClassification": [env_response],
        }
    )
    ctx = ExecutionContext(task_id=task.id, agent_type="debugger", db=db_session)
    # The debugger re-runs the suite; it passes -> FLAKY would fire. Force the
    # classification path by passing an explicit failing TestResult instead.
    fake_failed = TestResult(status="error", exit_code=None)
    classification = run(debugger.run(task, fake_failed, fake_summary(task), ctx))
    assert classification.category == "ENVIRONMENT_FAILURE"
    assert classification.fixable is False


def test_malformed_classification_raises_after_retry(db_session, tmp_path) -> None:
    """A classification that stays schema-invalid after the retry-once path is
    a HARD failure (DebuggerError) — a fabricated category is worse than a
    failed task, because the category drives routing."""
    from app.llm.mock import MALFORMED_RESPONSE

    repo_path = make_repo(
        tmp_path,
        app_content="VALUE = 2\n",
        tests="from pathlib import Path\n\n"
        "def test_value():\n"
        "    assert 'VALUE = 3' in Path('src/app.py').read_text()\n",
    )
    repo, task, worktree = repo_task_worktree(db_session, repo_path)
    debugger = debugger_with(
        {
            "ToolCallProposal": [FINAL_PROPOSAL],
            "FailureClassification": [MALFORMED_RESPONSE, MALFORMED_RESPONSE],
        }
    )
    ctx = ExecutionContext(task_id=task.id, agent_type="debugger", db=db_session)
    try:
        run(debugger.run(task, TestResult(status="failed", exit_code=1), fake_summary(task), ctx))
    except DebuggerError:
        return
    raise AssertionError("schema-invalid classification must raise DebuggerError")


def test_more_informative_and_inconsistency_logic() -> None:
    """Pure logic: an error/timeout first run vs a clean failing re-run is a
    DIFFERENT failure mode, not flakiness — classify from the more
    informative run and note the inconsistency."""
    db = None  # type: ignore[assignment] — these helpers don't touch the DB
    debugger = debugger_with({})

    class FakeRun:
        def __init__(self, status, exit_code, timed_out=False):
            self.id = uuid.uuid4()
            self.status = status
            self.exit_code = exit_code
            self.timed_out = timed_out
            self.passed = 0
            self.failed = 1
            self.duration_ms = 0
            self.failures = []

    first_error = FakeRun(status="error", exit_code=None, timed_out=True)
    rerun_failed = FakeRun(status="failed", exit_code=1)
    informative = debugger._more_informative(first_error, rerun_failed)
    assert informative.status == "failed"  # the run that actually ran tests wins
    inconsistency = debugger._rerun_inconsistency(first_error, rerun_failed)
    assert inconsistency is not None and "errored" in inconsistency

    # Same failure mode twice -> no inconsistency, not flaky.
    same = debugger._rerun_inconsistency(
        rerun_failed, FakeRun(status="failed", exit_code=1)
    )
    assert same is None
    # Consistent error twice -> no inconsistency.
    consistent = debugger._rerun_inconsistency(
        first_error, FakeRun(status="error", exit_code=None, timed_out=True)
    )
    assert consistent is None
