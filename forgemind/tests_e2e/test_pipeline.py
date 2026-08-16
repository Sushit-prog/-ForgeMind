"""End-to-end: a task submitted via the API reaches COMPLETED driven entirely
by a real worker process, with a complete execution-event trail.

Also covers the Section-J crash windows: killing the worker mid-run leaves the
task at its last persisted status (never half-written), and the startup sweep
resumes processing.
"""

from __future__ import annotations

import time
import uuid

from sqlalchemy import select

from app.models import ExecutionEvent, Task, TaskStatus
from app.runtime.state_machine import LEGAL_TRANSITIONS
from app.runtime.task_lifecycle import AUTO_PIPELINE, USER_CANCELLED
from tests_e2e.conftest import spawn_worker, wait_for

EXPECTED_STATUSES = [s.value for s in AUTO_PIPELINE]


def valid_payload(source_repo) -> dict:
    """A real clonable repo: RESEARCHING now runs the real agent, which
    needs a real worktree (a fake github URL would fail at clone time)."""
    return {
        "objective": "Fix the flaky test in auth",
        "repository_url": "file:///" + str(source_repo).replace("\\", "/"),
    }


def task_events(db, task_id: uuid.UUID) -> list[ExecutionEvent]:
    return list(
        db.scalars(
            select(ExecutionEvent)
            .where(ExecutionEvent.task_id == task_id)
            .order_by(ExecutionEvent.created_at, ExecutionEvent.id)
        )
    )


def assert_full_trail(db, task_id: uuid.UUID) -> None:
    """The task completed with exactly the Section-D pipeline, in order."""
    events = task_events(db, task_id)
    assert [e.to_status for e in events] == EXPECTED_STATUSES[1:]
    # Every pair must be legal (no skips, no duplicates, no double-processing).
    for e in events:
        assert TaskStatus(e.to_status) in LEGAL_TRANSITIONS[TaskStatus(e.from_status)]


def test_full_pipeline_worker_driven_to_completed(client, db_session, source_repo) -> None:
    proc = spawn_worker()
    try:
        created = client.post("/tasks", json=valid_payload(source_repo)).json()
        task_id = uuid.UUID(created["id"])

        def completed() -> bool:
            return client.get(f"/tasks/{task_id}").json()["status"] == "COMPLETED"

        assert wait_for(completed, timeout=60), "task never reached COMPLETED"
        assert_full_trail(db_session, task_id)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_cancelled_task_stays_failed_under_worker(client, db_session, source_repo) -> None:
    """Cancel enqueues nothing; even a running worker must not resurrect it."""
    created = client.post("/tasks", json=valid_payload(source_repo)).json()
    task_id = uuid.UUID(created["id"])

    resp = client.post(f"/tasks/{task_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "FAILED"

    proc = spawn_worker()
    try:
        # Give the worker time to (wrongly) pick it up if it could.
        time.sleep(3)
        assert client.get(f"/tasks/{task_id}").json()["status"] == "FAILED"
        events = task_events(db_session, task_id)
        assert len(events) == 1
        assert events[0].to_status == "FAILED"
        assert events[0].reason == USER_CANCELLED
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_crash_after_commit_is_healed_by_restart(client, db_session, source_repo) -> None:
    """Crash in the commit->re-enqueue window: last status persists, restart resumes."""
    proc = spawn_worker({"FORGEMIND_CRASH_AFTER_COMMIT": "1"})
    try:
        created = client.post("/tasks", json=valid_payload(source_repo)).json()
        task_id = uuid.UUID(created["id"])

        # The worker commits CREATED->PLANNING, then dies before re-enqueuing.
        def at_planning() -> bool:
            return client.get(f"/tasks/{task_id}").json()["status"] == "PLANNING"

        assert wait_for(at_planning, timeout=30), "worker never applied first transition"
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    # Crash state: exactly one persisted transition, legal, no half-writes.
    db_session.expire_all()
    task = db_session.get(Task, task_id)
    assert task.status == "PLANNING"
    events = task_events(db_session, task_id)
    assert len(events) == 1
    assert (events[0].from_status, events[0].to_status) == ("CREATED", "PLANNING")

    # Restart without the crash flag: the startup sweep re-enqueues and resumes.
    proc = spawn_worker()
    try:
        def completed() -> bool:
            return client.get(f"/tasks/{task_id}").json()["status"] == "COMPLETED"

        assert wait_for(completed, timeout=60), "task never resumed to COMPLETED"
        assert_full_trail(db_session, task_id)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_kill_between_transitions_resumes_cleanly(client, db_session, source_repo) -> None:
    """Killing the worker between two transitions: task at last persisted state."""
    proc = spawn_worker({"FORGEMIND_STEP_DELAY_MS": "8000"})
    try:
        created = client.post("/tasks", json=valid_payload(source_repo)).json()
        task_id = uuid.UUID(created["id"])

        # Wait for the FIRST transition to commit; the worker is then sleeping
        # through the 8s pre-transition delay of the next job.
        def first_event() -> bool:
            return len(task_events(db_session, task_id)) >= 1

        assert wait_for(first_event, timeout=30), "first transition never applied"
        # Kill while the next job is mid-delay (between two transitions).
        proc.terminate()
        proc.wait(timeout=10)

        db_session.expire_all()
        task = db_session.get(Task, task_id)
        assert task.status == "PLANNING"  # last persisted status, not half-written
        assert len(task_events(db_session, task_id)) == 1
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)

    # Restart with no delay: sweep resumes from the checkpoint.
    proc = spawn_worker()
    try:
        def completed() -> bool:
            return client.get(f"/tasks/{task_id}").json()["status"] == "COMPLETED"

        assert wait_for(completed, timeout=60), "task never resumed to COMPLETED"
        assert_full_trail(db_session, task_id)
    finally:
        proc.terminate()
        proc.wait(timeout=10)
