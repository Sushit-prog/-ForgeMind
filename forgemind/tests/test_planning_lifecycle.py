"""The PLANNING -> RESEARCHING transition now runs the real PlanningAgent
(Phase 5), replacing the stub's hardcoded step — verified end-to-end with a
mocked LLM provider.
"""

from __future__ import annotations

import asyncio
import uuid

from sqlalchemy import select

from app.agents.planner.agent import PlanningAgent
from app.llm import StubLLMProvider
from app.llm.mock import MALFORMED_RESPONSE
from app.models import ExecutionEvent, Plan, Task, TaskStatus
from app.runtime.task_lifecycle import advance_task_with_agents, transition_task


def run(coro):
    return asyncio.run(coro)


def make_planning_task(db_session, repo_task) -> Task:
    repo, task = repo_task
    transition_task(db_session, task, TaskStatus.PLANNING)
    db_session.commit()
    db_session.refresh(task)
    return task


def events_for(db_session, task_id: uuid.UUID) -> list[ExecutionEvent]:
    return list(
        db_session.scalars(
            select(ExecutionEvent)
            .where(ExecutionEvent.task_id == task_id)
            .order_by(ExecutionEvent.created_at, ExecutionEvent.id)
        )
    )


def test_planning_transition_persists_plan_and_moves_to_researching(
    db_session, repo_task
) -> None:
    task = make_planning_task(db_session, repo_task)
    planner = PlanningAgent(StubLLMProvider())

    new_status = run(advance_task_with_agents(db_session, task.id, planner=planner))

    assert new_status is TaskStatus.RESEARCHING
    db_session.expire_all()
    assert db_session.get(Task, task.id).status == "RESEARCHING"
    # A real plan was persisted.
    plans = db_session.scalars(select(Plan).where(Plan.task_id == task.id)).all()
    assert len(plans) == 1
    assert plans[0].status == "ACTIVE"
    # The event trail shows the reason.
    events = events_for(db_session, task.id)
    assert events[-1].to_status == "RESEARCHING"
    assert events[-1].reason == "plan_persisted"


def test_failed_plan_goes_to_failed_not_escalated(db_session, repo_task) -> None:
    task = make_planning_task(db_session, repo_task)
    planner = PlanningAgent(StubLLMProvider(responses=[MALFORMED_RESPONSE, MALFORMED_RESPONSE]))

    new_status = run(advance_task_with_agents(db_session, task.id, planner=planner))

    assert new_status is TaskStatus.FAILED  # never ESCALATED
    db_session.expire_all()
    task = db_session.get(Task, task.id)
    assert task.status == "FAILED"
    events = events_for(db_session, task.id)
    assert events[-1].to_status == "FAILED"
    assert events[-1].reason == "plan_validation_failed"
    # The failed raw output is preserved on an INVALID plan row.
    plans = db_session.scalars(select(Plan).where(Plan.task_id == task.id)).all()
    assert plans[0].status == "INVALID"


def test_no_planner_fails_task_cleanly(db_session, repo_task) -> None:
    task = make_planning_task(db_session, repo_task)

    new_status = run(advance_task_with_agents(db_session, task.id))

    assert new_status is TaskStatus.FAILED
    db_session.expire_all()
    assert db_session.get(Task, task.id).status == "FAILED"
    assert events_for(db_session, task.id)[-1].reason == "no_llm_provider"


def test_non_planning_states_use_stub_and_never_call_planner(db_session, repo_task) -> None:
    repo, task = repo_task
    # Advance the stub to IMPLEMENTING (bypassing PLANNING semantics).
    for target in (TaskStatus.PLANNING, TaskStatus.RESEARCHING, TaskStatus.IMPLEMENTING):
        transition_task(db_session, task, target)
    db_session.commit()

    provider = StubLLMProvider()
    planner = PlanningAgent(provider)
    new_status = run(advance_task_with_agents(db_session, task.id, planner=planner))

    assert new_status is TaskStatus.TESTING  # stub IMPLEMENTING -> TESTING
    assert provider.structured_calls == []  # planner never invoked
    assert not db_session.scalars(select(Plan)).all()  # no plan persisted
