"""E2E (Postgres + real worker): Phase 9 — REVIEWING, SECURITY_REVIEW, and
VERIFICATION branch for real.

The worker (stub LLM provider) drives a task through the full pipeline; the
Reviewer Agent and Security Agent run as read-only tool loops against the
developer's REAL commit, and the lifecycle branches on their persisted
verdicts:

    REVIEWING:       APPROVE -> SECURITY_REVIEW
                     REQUEST_CHANGES/REJECT -> IMPLEMENTING (shared budget)
    SECURITY_REVIEW: PASS -> VERIFICATION | FAIL -> IMPLEMENTING (same budget)
    VERIFICATION:    reviewed commit still HEAD + last test run still passed
                     -> PR_CREATION; otherwise -> TESTING (stale review)

Three scenarios, each against a fixture repo whose test accepts BOTH
VALUE = 2 and VALUE = 3 (the reviewer's fix instruction asks for 3):

1. reviewer reject-then-approve (FORGEMIND_MOCK_REVIEW_REJECT=1): the first
   review sees VALUE = 2 -> REQUEST_CHANGES -> one replan -> the developer
   writes VALUE = 3 -> the second review sees VALUE = 3 -> APPROVE ->
   SECURITY_REVIEW passes -> VERIFICATION -> COMPLETED, with
   replan_count == 1 and two persisted review rows.
2. security fail-then-pass (FORGEMIND_MOCK_SECURITY_FAIL=1): the reviewer
   approves, Security sees VALUE = 2 -> FAIL -> one replan -> VALUE = 3 ->
   Security sees VALUE = 3 -> PASS -> VERIFICATION -> COMPLETED,
   replan_count == 1 and two persisted security rows.
3. reviewer-triggered budget exhaustion: max_replans = 0 -> the reviewer's
   REQUEST_CHANGES replan escalates at the transition, exactly like the
   Phase 8 debugger path — one shared budget, enforced at the transition.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy import select

from app.git.runner import run_git
from app.models import ExecutionEvent, ReviewResult, SecurityResult, Task
from tests_e2e.conftest import spawn_worker, wait_for


def _file_url(repo: Path) -> str:
    return "file:///" + str(repo).replace("\\", "/")


def _make_repo(tmp_path) -> Path:
    """A real git repo whose pytest suite passes for VALUE = 2 OR VALUE = 3
    (the reviewer's fix instruction asks for 3, so the replan write must not
    fail the suite — the Phase 8 fixture's VALUE=2-only test would loop)."""
    repo = tmp_path / "phase9repo"
    repo.mkdir()
    run_git(repo, "init", "-b", "main")
    (repo / "README.md").write_text("# Phase 9 Fixture\n")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("VALUE = 1\n")
    (repo / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = ['tests']\n"
    )
    (repo / "tests").mkdir()
    (repo / "tests" / "test_suite.py").write_text(
        "from pathlib import Path\n\n\n"
        "def test_value_is_two_or_three() -> None:\n"
        "    content = Path('src/app.py').read_text()\n"
        "    assert 'VALUE = 2' in content or 'VALUE = 3' in content\n"
    )
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-m", "initial commit")
    return repo


def _submit(client, repo: Path, objective: str) -> uuid.UUID:
    created = client.post(
        "/tasks",
        json={"objective": objective, "repository_url": _file_url(repo)},
    ).json()
    return uuid.UUID(created["id"])


def _event_trail(db, task_id: uuid.UUID) -> list[tuple[str, str]]:
    events = db.scalars(
        select(ExecutionEvent)
        .where(ExecutionEvent.task_id == task_id)
        .order_by(ExecutionEvent.created_at, ExecutionEvent.id)
    ).all()
    return [(e.from_status, e.to_status) for e in events]


def test_reviewer_reject_then_approve_full_loop(
    client, db_session, tmp_path
) -> None:
    """The reviewer genuinely rejects (REQUEST_CHANGES on VALUE = 2), the
    developer replans to VALUE = 3, the second review approves, and the
    task completes through SECURITY_REVIEW and VERIFICATION — with both
    verdicts persisted and replan_count == 1.
    """
    repo = _make_repo(tmp_path)
    proc = spawn_worker({"FORGEMIND_MOCK_REVIEW_REJECT": "1"})
    try:
        task_id = _submit(client, repo, "make the value acceptable to review")

        def completed() -> bool:
            return client.get(f"/tasks/{task_id}").json()["status"] == "COMPLETED"

        assert wait_for(completed, timeout=120), "task never reached COMPLETED"

        trail = _event_trail(db_session, task_id)
        assert ("REVIEWING", "IMPLEMENTING") in trail, trail  # rejected -> replan
        assert ("REVIEWING", "SECURITY_REVIEW") in trail, trail  # approved
        assert ("SECURITY_REVIEW", "VERIFICATION") in trail, trail
        assert ("VERIFICATION", "PR_CREATION") in trail, trail

        task = db_session.get(Task, task_id)
        assert task is not None
        assert task.replan_count == 1

        reviews = db_session.scalars(
            select(ReviewResult).where(ReviewResult.task_id == task_id)
        ).all()
        assert [r.decision for r in reviews] == ["REQUEST_CHANGES", "APPROVE"], [
            r.decision for r in reviews
        ]
        assert reviews[0].issues, "the rejection must carry real issues"
        assert reviews[1].issues == []

        securities = db_session.scalars(
            select(SecurityResult).where(SecurityResult.task_id == task_id)
        ).all()
        assert [s.decision for s in securities] == ["PASS"], [
            s.decision for s in securities
        ]
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_security_fail_then_pass_full_loop(client, db_session, tmp_path) -> None:
    """The reviewer approves, Security FAILs on VALUE = 2, the developer
    replans to VALUE = 3, Security PASSes, and the task completes — the
    developer's next run saw the SECURITY finding, not a merged instruction.
    """
    repo = _make_repo(tmp_path)
    proc = spawn_worker({"FORGEMIND_MOCK_SECURITY_FAIL": "1"})
    try:
        task_id = _submit(client, repo, "make the value acceptable to security")

        def completed() -> bool:
            return client.get(f"/tasks/{task_id}").json()["status"] == "COMPLETED"

        assert wait_for(completed, timeout=120), "task never reached COMPLETED"

        trail = _event_trail(db_session, task_id)
        assert ("SECURITY_REVIEW", "IMPLEMENTING") in trail, trail  # failed -> replan
        assert ("SECURITY_REVIEW", "VERIFICATION") in trail, trail  # passed

        task = db_session.get(Task, task_id)
        assert task is not None
        assert task.replan_count == 1

        securities = db_session.scalars(
            select(SecurityResult).where(SecurityResult.task_id == task_id)
        ).all()
        assert [s.decision for s in securities] == ["FAIL", "PASS"], [
            s.decision for s in securities
        ]
        assert securities[0].findings, "the FAIL must carry real findings"

        # The reviewer approved both times (only security was set to fail).
        reviews = db_session.scalars(
            select(ReviewResult).where(ReviewResult.task_id == task_id)
        ).all()
        assert [r.decision for r in reviews] == ["APPROVE", "APPROVE"], [
            r.decision for r in reviews
        ]
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_reviewer_replan_budget_exhausted_escalates(
    client, db_session, tmp_path
) -> None:
    """max_replans = 0: the reviewer's REQUEST_CHANGES replan escalates at
    the transition — the reviewer-triggered replan draws from the SAME
    shared budget as the debugger's, enforced at the transition layer.
    """
    repo = _make_repo(tmp_path)
    proc = spawn_worker({"FORGEMIND_MOCK_REVIEW_REJECT": "1"})
    try:
        task_id = _submit(client, repo, "make the value acceptable to review")

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
        assert ("REVIEWING", "ESCALATED") in trail, trail
        assert ("REVIEWING", "IMPLEMENTING") not in trail, trail

        task = db_session.get(Task, task_id)
        assert task is not None
        assert task.status == "ESCALATED"
        assert task.replan_count == 0  # never incremented past the budget
    finally:
        proc.terminate()
        proc.wait(timeout=10)
