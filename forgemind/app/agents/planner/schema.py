"""Plan schema and dependency-graph validation (architecture doc 6.1 / E).

The LLM emits a plan as JSON; this module defines the schema it must
match and the DAG rules it must satisfy:

- every ``depends_on`` references a real step id (no dangling edges),
- no cycles (the graph is a DAG),
- step ids are unique,
- every ``implement`` step transitively depends on at least one
  ``research`` step (research precedes implementation).

Validation is deterministic and schema-driven — no LLM involved.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

StepType = Literal["research", "implement", "test", "debug", "review", "security", "github"]


class PlanStep(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    step_type: StepType
    description: str = Field(min_length=1, max_length=2000)
    depends_on: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    objective: str = Field(min_length=1, max_length=100_000)
    steps: list[PlanStep] = Field(min_length=1)


class PlanValidationError(ValueError):
    """The LLM's plan cannot be coerced into a legal dependency graph.

    Raised only after the retry-once path has been exhausted — never as a
    silent fallback. Carries the raw output for debugging.
    """

    def __init__(self, detail: str, raw_output: str = "") -> None:
        self.detail = detail
        self.raw_output = raw_output
        super().__init__(f"invalid plan: {detail}")


def validate_plan_dag(plan: Plan) -> None:
    """Raise ``PlanValidationError`` if ``plan`` is not a legal DAG.

    Rules (Section 6.1): unique step ids, every depends_on edge targets a
    real step, no cycles, and every implement step has a research
    ancestor. Returns None on success.
    """
    if not plan.steps:
        raise PlanValidationError("plan has no steps")

    ids = [step.id for step in plan.steps]
    if len(set(ids)) != len(ids):
        raise PlanValidationError("step ids are not unique")

    id_set = set(ids)
    for step in plan.steps:
        for dep in step.depends_on:
            if dep not in id_set:
                raise PlanValidationError(
                    f"step {step.id!r} depends on unknown step {dep!r}"
                )

    # Original edges, kept intact — the Kahn loop below mutates its own copy.
    edges: dict[str, set[str]] = {s.id: set(s.depends_on) for s in plan.steps}

    # Kahn's algorithm: detect cycles (also yields a topological order).
    indegree: dict[str, int] = {sid: len(deps) for sid, deps in edges.items()}
    dependents: dict[str, set[str]] = {s.id: set() for s in plan.steps}
    for step in plan.steps:
        for dep in step.depends_on:
            dependents[dep].add(step.id)

    ready = [sid for sid, deg in indegree.items() if deg == 0]
    ordered: list[str] = []
    while ready:
        sid = ready.pop()
        ordered.append(sid)
        for dependent in list(dependents[sid]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
    if len(ordered) != len(plan.steps):
        raise PlanValidationError("plan contains a dependency cycle")

    # research-before-implement: every implement step must reach a research
    # ancestor through the ORIGINAL edges (research steps are the roots of
    # the subgraph an implement step hangs off).
    def ancestors(step_id: str, seen: set[str]) -> set[str]:
        if step_id in seen:
            return seen
        seen.add(step_id)
        for dep in edges[step_id]:
            ancestors(dep, seen)
        return seen

    types = {s.id: s.step_type for s in plan.steps}
    for step in plan.steps:
        if step.step_type == "implement":
            reachable = ancestors(step.id, set()) - {step.id}
            if not any(types.get(sid) == "research" for sid in reachable):
                raise PlanValidationError(
                    f"implement step {step.id!r} has no research step before it"
                )
