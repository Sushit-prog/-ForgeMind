"""Regression: a poisoned (pending-rollback) session must not strand a task.

The research crash of task 28cb92ec-… left its PostgreSQL session pending a
rollback after the failed JSONB flush, so the run's ``_cas_transition`` FAILED
path itself raised ``PendingRollbackError`` before writing anything and the
task stayed in RESEARCHING forever (arq's retry of the deterministic job id is
silently deduped). SQLite's driver never enters pending-rollback, so the
poisoned-session contract is asserted deterministically here via a stub that
mimics the pg behavior: ``in_transaction()`` stays True and every statement
raises ``PendingRollbackError`` until ``rollback()`` is called.
"""

from sqlalchemy.exc import PendingRollbackError

from app.models import Repository, Task, TaskStatus
from app.runtime.task_lifecycle import _cas_transition


def test_cas_transition_heals_poisoned_session(db_session, monkeypatch) -> None:
    repo = Repository(url="https://example.com/poisoned.git", default_branch="main")
    db_session.add(repo)
    db_session.flush()
    task = Task(
        objective="poisoned-session regression",
        repository_id=repo.id,
        status=TaskStatus.CREATED,
    )
    db_session.add(task)
    db_session.commit()
    db_session.refresh(task)

    real_execute = db_session.execute
    real_rollback = db_session.rollback
    state = {"rollback_called": False, "statements": 0}

    def poisoned_execute(*args, **kwargs):
        state["statements"] += 1
        if not state["rollback_called"]:
            raise PendingRollbackError("(simulated failed JSONB flush)")
        return real_execute(*args, **kwargs)

    def poisoned_in_transaction():
        return not state["rollback_called"]

    def healing_rollback():
        state["rollback_called"] = True
        real_rollback()

    monkeypatch.setattr(db_session, "execute", poisoned_execute)
    monkeypatch.setattr(db_session, "in_transaction", poisoned_in_transaction)
    monkeypatch.setattr(db_session, "rollback", healing_rollback)

    # Pre-fix, _cas_transition issued its SELECT first and PendingRollbackError
    # escaped — the task never reached FAILED.
    result = _cas_transition(
        db_session,
        task.id,
        TaskStatus.CREATED,
        TaskStatus.FAILED,
        "test: poisoned session",
    )
    assert result == TaskStatus.FAILED
    assert state["rollback_called"] is True
    assert state["statements"] >= 1

    db_session.expire_all()
    refreshed = db_session.get(Task, task.id)
    assert TaskStatus(refreshed.status) == TaskStatus.FAILED