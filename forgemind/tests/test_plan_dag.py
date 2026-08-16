"""DAG validation tests (Section 6.1): cycles, dangling refs, ordering."""

import pytest

from app.agents.planner.schema import Plan, PlanValidationError, validate_plan_dag


def step(sid: str, step_type: str = "research", depends_on: list[str] | None = None):
    return {
        "id": sid,
        "step_type": step_type,
        "description": f"step {sid}",
        "depends_on": depends_on or [],
    }


def plan(*steps) -> Plan:
    return Plan(objective="fix", steps=steps)  # type: ignore[arg-type]


def test_valid_linear_plan_passes() -> None:
    p = plan(
        step("research-1"),
        step("implement-1", "implement", ["research-1"]),
        step("test-1", "test", ["implement-1"]),
        step("review-1", "review", ["test-1"]),
    )
    validate_plan_dag(p)  # no raise


def test_valid_diamond_dag_passes() -> None:
    p = plan(
        step("r1"),
        step("r2"),
        step("i1", "implement", ["r1", "r2"]),
        step("t1", "test", ["i1"]),
    )
    validate_plan_dag(p)


def test_cycle_rejected() -> None:
    p = plan(
        step("a"),
        step("b", "implement", ["c"]),
        step("c", "test", ["b"]),  # b -> c -> b
    )
    with pytest.raises(PlanValidationError, match="cycle"):
        validate_plan_dag(p)


def test_self_cycle_rejected() -> None:
    p = plan(step("a", depends_on=["a"]))
    with pytest.raises(PlanValidationError, match="cycle"):
        validate_plan_dag(p)


def test_dangling_dependency_rejected() -> None:
    p = plan(
        step("a"),
        step("b", "implement", ["ghost"]),
    )
    with pytest.raises(PlanValidationError, match="ghost"):
        validate_plan_dag(p)


def test_duplicate_ids_rejected() -> None:
    p = plan(step("a"), step("a", "implement"))
    with pytest.raises(PlanValidationError, match="unique"):
        validate_plan_dag(p)


def test_implement_without_research_rejected() -> None:
    p = plan(step("i1", "implement"))
    with pytest.raises(PlanValidationError, match="no research step"):
        validate_plan_dag(p)


def test_implement_depending_only_on_implement_rejected() -> None:
    p = plan(
        step("i1", "implement"),
        step("i2", "implement", ["i1"]),
    )
    with pytest.raises(PlanValidationError, match="no research step"):
        validate_plan_dag(p)


def test_implement_with_transitive_research_passes() -> None:
    p = plan(
        step("r1"),
        step("d1", "debug", ["r1"]),
        step("i1", "implement", ["d1"]),  # research ancestor is transitive
    )
    validate_plan_dag(p)


def test_empty_plan_rejected_at_schema_level() -> None:
    """Empty steps can't even be constructed — the schema is the first wall
    (an LLM payload with no steps raises a schema error at parse, before DAG
    validation is ever reached)."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Plan(objective="x", steps=[])  # type: ignore[arg-type]
