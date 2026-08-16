"""E2E concurrency: two worker instances racing the same task.

``advance_task_once`` takes the task row FOR UPDATE, so concurrent workers
serialize: each job applies exactly one transition and re-enqueues the next.
The assertion is structural — the event trail must be exactly the Section-D
pipeline, one transition each, in order, with no duplicates or skips.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select

from app.models import ExecutionEvent, Task
from app.runtime.task_lifecycle import AUTO_PIPELINE
from tests_e2e.conftest import spawn_worker, wait_for

EXPECTED_STATUSES = [s.value for s in AUTO_PIPELINE][1:]


def test_two_workers_same_task_no_double_processing(client, db_session) -> None:
    # Two workers, a widened transition window so races are actually possible.
    workers = [
        spawn_worker({"FORGEMIND_STEP_DELAY_MS": "150"}),
        spawn_worker({"FORGEMIND_STEP_DELAY_MS": "150"}),
    ]
    try:
        tasks = []
        for i in range(3):
            created = client.post(
                "/tasks",
                json={
                    "objective": f"Task {i}",
                    "repository_url": "https://github.com/org/repo.git",
                },
            ).json()
            tasks.append(uuid.UUID(created["id"]))

        for task_id in tasks:
            def completed(tid: uuid.UUID = task_id) -> bool:
                return client.get(f"/tasks/{tid}").json()["status"] == "COMPLETED"

            assert wait_for(completed, timeout=60), f"task {task_id} never completed"

            events = db_session.scalars(
                select(ExecutionEvent)
                .where(ExecutionEvent.task_id == task_id)
                .order_by(ExecutionEvent.created_at, ExecutionEvent.id)
            ).all()

            # Exactly one transition per pipeline step — no double-processing.
            assert [e.to_status for e in events] == EXPECTED_STATUSES, (
                f"task {task_id}: event trail was not the exact pipeline"
            )
            # Every transition must have moved forward by one step.
            for e in events:
                assert e.from_status != e.to_status

            db_session.expire_all()
            task = db_session.get(Task, task_id)
            assert task.replan_count == 0
    finally:
        for proc in workers:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)
