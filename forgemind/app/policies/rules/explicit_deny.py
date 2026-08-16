"""Explicit deny rule: a fixed set of tool names that are always denied.

This is the deterministic mechanism behind "denied_tool.py's test": the
policy names the tool, the engine denies it, ``execute`` is never reached.
The same mechanism will carry the shell-command allowlist / no-push-to-main
rules in later phases, parameterized over typed tool input.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.policies.base import PolicyDecision, PolicyRule
from app.tools.base import Tool


class ExplicitDenyRule(PolicyRule):
    name = "explicit-deny"

    def __init__(self, denied_tools: set[str] | None = None) -> None:
        self.denied_tools: set[str] = denied_tools or set()

    def evaluate(
        self,
        tool: Tool,
        input: BaseModel,
        agent_capabilities: set[str],
    ) -> PolicyDecision | None:
        if tool.name in self.denied_tools:
            return PolicyDecision(
                allowed=False,
                reason=f"tool {tool.name!r} is explicitly denied by policy",
                rule=self.name,
            )
        return None  # not in the deny list — abstain
