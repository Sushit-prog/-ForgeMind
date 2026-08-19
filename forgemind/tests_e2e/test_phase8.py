"""E2E (Postgres + real worker): Phase 8 — TESTING and DEBUGGING branch for real.

The worker (stub LLM provider) drives a task through PLANNING -> RESEARCHING
-> IMPLEMENTING -> TESTING, where the Test Agent runs the repository's REAL
configured test command as an actual subprocess against the worktree (no
mocked shell). The failure branch then runs the real Debugger Agent
(read-only investigation + flakiness re-run + classification) and the
lifecycle branches on the persisted classification:

    TESTING:   passed -> REVIEWING | failed/error -> DEBUGGING
    DEBUGGING: flaky -> REVIEWING | unfixable -> FAILED
              | fixable -> IMPLEMENTING (replan budget enforced)

Three scenarios, each against a fixture repo whose test asserts a specific
src/app.py content:

1. fail-once-then-pass: the stub developer's first write (VALUE = 2) fails
   the test (asserts VALUE = 3); the debugger classifies CODE_FAILURE and
   the replan developer writes VALUE = 3; the re-run passes; task COMPLETED
   with replan_count == 1 and the full TESTING -> DEBUGGING -> IMPLEMENTING
   -> TESTING -> REVIEWING trail on the trace.
2. flaky: the first run fails, the debugger's single re-run passes -> the
   classification is FLAKY_TEST (deterministic, no LLM guess) and the task
   routes to REVIEWING, never back to IMPLEMENTING.
3. replan budget exhausted: max_replans = 0 -> the fixable classification
   escalates at the transition instead of replanning again.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy import select

from app.git.runner import run_git
from app.models import ExecutionEvent, FailureClassification, Task, TestRun
from tests_e2e.conftest import approve_task, spawn_worker, wait_for


def _file_url(repo: Path) -> str:
    return "file:///" + str(repo).replace("\\", "/")


def _make_repo(tmp_path, *, test_body: str, initial_value: str = "VALUE = 1\n") -> Path:
    """A real git repo with a pytest suite whose behavior is ``test_body``.

    ``pyproject.toml`` is the Phase 8 detection marker — discovery stores
    ``test_command = pytest``, so the tester runs a REAL subprocess.
    """
    repo = tmp_path / "phase8repo"
    repo.mkdir()
    run_git(repo, "init", "-b", "main")
    (repo / "README.md").write_text("# Phase 8 Fixture\n")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text(initial_value)
    (repo / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = ['tests']\n"
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_suite.py").write_text(
        "from pathlib import Path\n\n\n" + test_body
    )
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-m", "initial commit")
    return repo


def _submit(client, repo: Path, objective: str) -> uuid.UUID:
    created = client.post(
        "/tasks",
        json={
            "objective": objective,
            "repository_url": _file_url(repo),
            "fork_url": "https://github.com/fork-owner/forgemind-e2e-fork",
        },
    ).json()
    return uuid.UUID(created["id"])


def _event_trail(db, task_id: uuid.UUID) -> list[tuple[str, str]]:
    events = db.scalars(
        select(ExecutionEvent)
        .where(ExecutionEvent.task_id == task_id)
        .order_by(ExecutionEvent.created_at, ExecutionEvent.id)
    ).all()
    return [(e.from_status, e.to_status) for e in events]


def test_fail_once_debug_then_pass_full_loop(client, db_session, tmp_path) -> None:
    """The realistic full loop, end to end, with a REAL test subprocess:

    IMPLEMENTING (VALUE=2) -> TESTING (pytest FAILS, asserts VALUE=3) ->
    DEBUGGING (re-run fails same way -> CODE_FAILURE, fixable) ->
    IMPLEMENTING (replan, VALUE=3) -> TESTING (pytest PASSES) -> REVIEWING
    -> ... -> COMPLETED. replan_count == 1, classifications + test runs
    persisted, and the trail shows both failure and recovery.
    """
    repo = _make_repo(
        tmp_path,
        test_body=(
            "def test_value_is_three() -> None:\n"
            "    content = Path('src/app.py').read_text()\n"
            "    assert 'VALUE = 3' in content\n"
        ),
    )
    proc = spawn_worker()
    try:
        task_id = _submit(client, repo, "make the failing test pass")

        approve_task(client, task_id, timeout=120)

        trail = _event_trail(db_session, task_id)
        assert ("TESTING", "DEBUGGING") in trail, trail
        assert ("DEBUGGING", "IMPLEMENTING") in trail, trail
        assert ("TESTING", "REVIEWING") in trail, trail

        # Exactly one bounded replan happened.
        task = db_session.get(Task, task_id)
        assert task is not None
        assert task.replan_count == 1

        # Two implementations (initial + replan), two commits, one summary each.
        from app.models import ImplementationSummary

        summaries = db_session.scalars(
            select(ImplementationSummary).where(
                ImplementationSummary.task_id == task_id
            )
        ).all()
        assert len(summaries) == 2, f"expected 2 summaries, got {len(summaries)}"

        # Test runs: TESTING fail, debugger re-run fail (not flaky), TESTING pass.
        runs = db_session.scalars(
            select(TestRun).where(TestRun.task_id == task_id)
        ).all()
        assert [r.status for r in runs] == ["failed", "failed", "passed"], [
            r.status for r in runs
        ]
        # They were REAL subprocesses with real exit codes.
        assert all(r.exit_code is not None for r in runs)
        assert runs[0].exit_code != 0 and runs[2].exit_code == 0

        # The classification was persisted with the concrete fix instruction.
        rows = db_session.scalars(
            select(FailureClassification).where(
                FailureClassification.task_id == task_id
            )
        ).all()
        assert rows and rows[-1].category == "CODE_FAILURE"
        assert rows[-1].fixable is True
        assert rows[-1].is_flaky is False
        assert rows[-1].fix_instruction is not None
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_flaky_test_detected_by_rerun_routes_to_reviewing(
    client, db_session, tmp_path
) -> None:
    """A genuinely intermittent test: the first run fails, the debugger's
    single re-run passes -> classified FLAKY_TEST deterministically (no LLM
    guess) and the task routes to REVIEWING, never back to IMPLEMENTING —
    but the flaky result stays on the trace.
    """
    repo = _make_repo(
        tmp_path,
        test_body=(
            "def test_flaky() -> None:\n"
            "    marker = Path('flaky-marker.txt')\n"
            "    if not marker.exists():\n"
            "        marker.write_text('done')\n"
            "        raise AssertionError('first run fails; re-run passes')\n"
        ),
    )
    proc = spawn_worker()
    try:
        task_id = _submit(client, repo, "fix the flaky test")

        approve_task(client, task_id, timeout=120)

        trail = _event_trail(db_session, task_id)
        assert ("TESTING", "DEBUGGING") in trail, trail
        assert ("DEBUGGING", "REVIEWING") in trail, trail
        assert ("DEBUGGING", "IMPLEMENTING") not in trail, trail

        task = db_session.get(Task, task_id)
        assert task is not None
        assert task.replan_count == 0  # never replanned

        rows = db_session.scalars(
            select(FailureClassification).where(
                FailureClassification.task_id == task_id
            )
        ).all()
        assert rows and rows[-1].category == "FLAKY_TEST"
        assert rows[-1].is_flaky is True

        # Both runs are on the trace — the flake is flagged, never swept away.
        runs = db_session.scalars(
            select(TestRun).where(TestRun.task_id == task_id)
        ).all()
        assert [r.status for r in runs] == ["failed", "passed"], [
            r.status for r in runs
        ]
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_replan_budget_exhausted_escalates(client, db_session, tmp_path) -> None:
    """max_replans = 0: TESTING fails, DEBUGGING classifies it fixable, and
    the transition escalates instead of replanning — the budget is enforced
    at the transition layer, not left to the Developer to know about.
    """
    repo = _make_repo(
        tmp_path,
        test_body=(
            "def test_value_is_three() -> None:\n"
            "    content = Path('src/app.py').read_text()\n"
            "    assert 'VALUE = 3' in content\n"
        ),
    )
    proc = spawn_worker()
    try:
        task_id = _submit(client, repo, "make the failing test pass")

        # The worker moves fast; set the budget to 0 as soon as the task exists.
        def set_budget() -> bool:
            task = db_session.get(Task, task_id)
            if task is None:
                return False
            task.max_replans = 0
            db_session.commit()
            return True

        assert wait_for(set_budget, timeout=30), "task row never appeared"

        def escalated() -> bool:
            return client.get(f"/tasks/{task_id}").json()["status"] == "ESCALATED"

        assert wait_for(escalated, timeout=120), "task never escalated"

        trail = _event_trail(db_session, task_id)
        assert ("TESTING", "DEBUGGING") in trail, trail
        assert ("DEBUGGING", "ESCALATED") in trail, trail
        assert ("DEBUGGING", "IMPLEMENTING") not in trail, trail

        task = db_session.get(Task, task_id)
        assert task is not None
        assert task.status == "ESCALATED"
        assert task.replan_count == 0  # never incremented past the budget
    finally:
        proc.terminate()
        proc.wait(timeout=10)
