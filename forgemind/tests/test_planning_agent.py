"""PlanningAgent integration tests (mocked LLM, real persistence).

Covers: valid plan -> persisted; malformed -> retry-once -> success;
malformed twice -> raise + raw preserved; timeout retries bounded; empty
objective never hangs; planner capabilities are structurally empty.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from sqlalchemy import select

from app.agents.planner.agent import PlannerConfigError, PlanningAgent
from app.agents.planner.schema import Plan, PlanValidationError
from app.llm import LLMTimeoutError, StubLLMProvider
from app.llm.mock import DEFAULT_PLAN_RESPONSE, MALFORMED_RESPONSE
from app.llm.provider import LLMProvider, Message
from app.models import Plan as PlanRow
from app.models import PlanStep as PlanStepRow
from app.tools.base import ExecutionContext


def run(coro):
    return asyncio.run(coro)


def ctx_for(db_session, task) -> ExecutionContext:
    return ExecutionContext(task_id=task.id, agent_type="planner", db=db_session)


def plans_for(db_session, task_id) -> list[PlanRow]:
    return list(db_session.scalars(select(PlanRow).where(PlanRow.task_id == task_id)))


def steps_for(db_session, plan_id) -> list[PlanStepRow]:
    return list(db_session.scalars(select(PlanStepRow).where(PlanStepRow.plan_id == plan_id)))


def test_planner_has_no_capabilities() -> None:
    assert PlanningAgent.capabilities == []


def test_valid_plan_persisted_and_returned(db_session, repo_task) -> None:
    repo, task = repo_task
    agent = PlanningAgent(StubLLMProvider())
    ctx = ctx_for(db_session, task)

    plan = run(agent.run(task, ctx))

    assert isinstance(plan, Plan)
    assert plan.steps[0].step_type == "research"

    rows = plans_for(db_session, task.id)
    assert len(rows) == 1
    assert rows[0].status == "ACTIVE"
    assert rows[0].raw_llm_output  # canonical JSON preserved (Section 47)
    assert "research" in rows[0].raw_llm_output

    steps = steps_for(db_session, rows[0].id)
    assert len(steps) == 4
    assert [s.step_type for s in steps] == ["research", "implement", "test", "review"]
    assert steps[1].depends_on is not None  # implement depends on research
    # Full DAG preserved in params.
    assert steps[1].params["depends_on"] == ["research-1"]


def test_malformed_then_valid_retries_once(db_session, repo_task) -> None:
    repo, task = repo_task
    provider = StubLLMProvider(responses=[MALFORMED_RESPONSE, DEFAULT_PLAN_RESPONSE])
    agent = PlanningAgent(provider)
    ctx = ctx_for(db_session, task)

    plan = run(agent.run(task, ctx))
    assert isinstance(plan, Plan)
    assert len(provider.structured_calls) == 2  # exactly one correction retry
    # The correction prompt must mention the rejection reason.
    assert "rejected" in provider.structured_calls[1][-1].content


def test_malformed_twice_raises_and_preserves_raw(db_session, repo_task) -> None:
    repo, task = repo_task
    provider = StubLLMProvider(responses=[MALFORMED_RESPONSE, MALFORMED_RESPONSE])
    agent = PlanningAgent(provider)
    ctx = ctx_for(db_session, task)

    with pytest.raises(PlanValidationError) as exc_info:
        run(agent.run(task, ctx))
    assert len(provider.structured_calls) == 2

    # Raw output preserved on an INVALID plan row — never silently dropped.
    rows = plans_for(db_session, task.id)
    assert len(rows) == 1
    assert rows[0].status == "INVALID"
    assert "this is not json" in rows[0].raw_llm_output
    assert steps_for(db_session, rows[0].id) == []  # nothing garbage persisted
    assert exc_info.value.raw_output == MALFORMED_RESPONSE


def test_dag_invalid_plan_triggers_retry(db_session, repo_task) -> None:
    """A schema-valid but DAG-illegal plan (cycle) is retried, not accepted."""
    cyclic = json.dumps(
        {
            "objective": "x",
            "steps": [
                {"id": "a", "step_type": "research", "description": "a", "depends_on": ["b"]},
                {"id": "b", "step_type": "implement", "description": "b", "depends_on": ["a"]},
            ],
        }
    )
    repo, task = repo_task
    provider = StubLLMProvider(responses=[cyclic, DEFAULT_PLAN_RESPONSE])
    agent = PlanningAgent(provider)
    ctx = ctx_for(db_session, task)

    plan = run(agent.run(task, ctx))
    assert isinstance(plan, Plan)
    assert len(provider.structured_calls) == 2
    assert "cycle" in provider.structured_calls[1][-1].content.lower()


class TimeoutThenSuccess(LLMProvider):
    """Raises LLMTimeoutError ``failures`` times, then delegates."""

    def __init__(self, failures: int, delegate: LLMProvider) -> None:
        self.failures = failures
        self.delegate = delegate
        self.calls = 0

    async def generate(self, messages: list[Message], **kwargs: object) -> str:
        return await self.delegate.generate(messages, **kwargs)

    async def structured_output(self, messages, schema, **kwargs) -> object:
        self.calls += 1
        if self.calls <= self.failures:
            raise LLMTimeoutError("simulated timeout")
        return await self.delegate.structured_output(messages, schema, **kwargs)


def test_timeout_retries_with_backoff_then_succeeds(db_session, repo_task) -> None:
    repo, task = repo_task
    provider = TimeoutThenSuccess(
        failures=2, delegate=StubLLMProvider(responses=[DEFAULT_PLAN_RESPONSE])
    )
    agent = PlanningAgent(provider, timeout_retries=3, backoff_base_seconds=0.01)
    ctx = ctx_for(db_session, task)

    plan = run(agent.run(task, ctx))
    assert isinstance(plan, Plan)
    assert provider.calls == 3  # 2 timeouts + 1 success, all within one attempt


def test_timeout_exhaustion_raises_and_persists_failure(db_session, repo_task) -> None:
    repo, task = repo_task
    provider = TimeoutThenSuccess(
        failures=99, delegate=StubLLMProvider(responses=[DEFAULT_PLAN_RESPONSE])
    )
    agent = PlanningAgent(provider, timeout_retries=2, backoff_base_seconds=0.01)
    ctx = ctx_for(db_session, task)

    with pytest.raises(LLMTimeoutError):
        run(agent.run(task, ctx))
    assert provider.calls == 3  # initial + 2 retries, then give up
    rows = plans_for(db_session, task.id)
    assert rows[0].status == "INVALID"  # failure recorded, not silent


def test_near_empty_objective_produces_valid_plan(db_session, repo_task) -> None:
    repo, task = repo_task
    task.objective = "   "  # whitespace-only
    agent = PlanningAgent(StubLLMProvider())
    plan = run(agent.run(task, ctx_for(db_session, task)))
    assert isinstance(plan, Plan)  # never hangs, never crashes on empty input


def test_planner_config_error_without_key_or_mock(monkeypatch) -> None:
    monkeypatch.delenv("FORGEMIND_MOCK_LLM", raising=False)
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        from app.agents.planner.agent import build_planner

        with pytest.raises(PlannerConfigError):
            build_planner()
    finally:
        get_settings.cache_clear()
