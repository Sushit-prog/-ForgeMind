"""Tool contracts (architecture doc section F).

Every tool declares: name, description, typed input/output schemas, the
capabilities REQUIRED to invoke it, and a risk tier. The ``ToolPipeline``
enforces the five-step sequence (validate -> capability -> policy ->
execute -> audit) on every call — no tool bypasses the pipeline.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class ExecutionContext(BaseModel):
    """What the pipeline knows about the current execution.

    All fields optional — a tool call outside a task (e.g. registry tests)
    still audits correctly; the row just carries nulls for what's absent.

    ``db`` is the persistence session tools use to resolve server-side
    state (e.g. ``worktree_id`` -> path). Agent runtimes populate it;
    tools that need it raise a clear error if it's absent.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    task_id: uuid.UUID | None = None
    step_id: uuid.UUID | None = None
    agent_type: str | None = None
    db: Session | None = None


class Tool(ABC):
    """Base class for every tool. Subclasses must set the schema attributes
    and implement ``execute``; the ABC enforces that at class-creation time
    so a misdeclared tool fails loudly at registration, not mid-pipeline."""

    name: str = ""
    description: str = ""
    input_schema: type[BaseModel] | None = None
    output_schema: type[BaseModel] | None = None
    capabilities: list[str] = []
    risk: RiskLevel = "LOW"

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.name:
            raise TypeError(f"{cls.__name__} must define a non-empty `name`")
        if not cls.description:
            raise TypeError(f"{cls.__name__} must define a non-empty `description`")
        if cls.input_schema is None or cls.output_schema is None:
            raise TypeError(f"{cls.__name__} must define input_schema and output_schema")
        if not isinstance(cls.input_schema, type) or not issubclass(cls.input_schema, BaseModel):
            raise TypeError(f"{cls.__name__}.input_schema must be a pydantic BaseModel class")
        if not isinstance(cls.output_schema, type) or not issubclass(cls.output_schema, BaseModel):
            raise TypeError(f"{cls.__name__}.output_schema must be a pydantic BaseModel class")
        if cls.risk not in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            raise TypeError(f"{cls.__name__}.risk must be one of LOW/MEDIUM/HIGH/CRITICAL")

    @abstractmethod
    async def execute(self, input: BaseModel, ctx: ExecutionContext) -> BaseModel:
        """Perform the tool's work. Implementations never talk to the policy
        engine or write audit rows — the pipeline owns all of that."""
