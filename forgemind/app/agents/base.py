"""Agent contract (architecture doc section E).

Every agent declares its name and its capability set. The capability set
is the STRUCTURAL enforcement boundary: an agent can only ever invoke
tools whose required capabilities are inside its set (the pipeline checks
this at call time), and agents like the Planner declare an EMPTY set —
they cannot invoke any tool, enforced by the pipeline, not by convention.

``run`` produces a typed artifact (pydantic model) and receives only the
task objective + whatever it needs via ``ExecutionContext`` — never the
full task history dumped into context.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel

from app.models import Task
from app.tools.base import ExecutionContext


class Agent(ABC):
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    # Tools this agent may invoke; the pipeline denies anything outside.
    capabilities: ClassVar[list[str]] = []

    @abstractmethod
    async def run(self, task: Task, ctx: ExecutionContext) -> BaseModel:
        """Produce the agent's typed artifact for ``task``."""
