"""Shared policy types.

Kept separate from ``engine.py`` so individual rules can import them
without a cycle: rules are leaf modules, the engine composes them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from app.tools.base import Tool


class PolicyDecision(BaseModel):
    """Outcome of one rule (or the engine): ALLOW or DENY with a reason."""

    allowed: bool
    reason: str
    rule: str


class PolicyRule(ABC):
    """One deterministic rule. ``evaluate`` returns a decision, or ``None``
    to abstain (this rule has nothing to say about this call)."""

    name: str = ""

    @abstractmethod
    def evaluate(
        self,
        tool: Tool,
        input: BaseModel,
        agent_capabilities: set[str],
    ) -> PolicyDecision | None: ...
