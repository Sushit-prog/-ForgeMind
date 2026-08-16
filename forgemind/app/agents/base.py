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

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel

from app.llm.errors import LLMTimeoutError
from app.llm.openrouter import is_transient_error
from app.llm.provider import LLMProvider, Message
from app.models import Task
from app.tools.base import ExecutionContext

logger = logging.getLogger(__name__)


async def structured_output_with_retries(
    provider: LLMProvider,
    messages: list[Message],
    schema: type[BaseModel],
    *,
    timeout_retries: int,
    backoff_base_seconds: float = 0.5,
) -> BaseModel:
    """One structured_output call with bounded TRANSIENT retries.

    Only timeouts and 429/5xx retry here; malformed output propagates
    immediately so the caller's correction retry can fire. This is the
    shared retry seam for every agent (planner, researcher, ...) — one
    pattern, not one per agent (Section 42 budget discipline).
    """
    attempt = 0
    while True:
        try:
            return await provider.structured_output(messages, schema)
        except LLMTimeoutError:
            if attempt >= timeout_retries:
                raise
        except Exception as exc:  # noqa: BLE001
            if not is_transient_error(exc):
                raise
            if attempt >= timeout_retries:
                raise
            logger.warning("transient LLM error (%s), retry %d/%d", exc, attempt + 1, timeout_retries)
        await asyncio.sleep(backoff_base_seconds * (2**attempt))
        attempt += 1


class Agent(ABC):
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    # Tools this agent may invoke; the pipeline denies anything outside.
    capabilities: ClassVar[list[str]] = []

    @abstractmethod
    async def run(self, task: Task, ctx: ExecutionContext) -> BaseModel:
        """Produce the agent's typed artifact for ``task``."""
