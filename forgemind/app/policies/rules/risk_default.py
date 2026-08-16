"""Default-by-risk rule: a risk tier denies a call unless explicitly allowed.

Phase 3 has no CRITICAL tools, so with the default ``deny_at_or_above``
this rule abstains on everything — it is a standing guard so that a future
CRITICAL tool fails CLOSED (denied) until someone explicitly allowlists it,
rather than failing open by omission.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.policies.base import PolicyDecision, PolicyRule
from app.tools.base import RiskLevel, Tool

_RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


class RiskTierRule(PolicyRule):
    name = "risk-default"

    def __init__(
        self,
        deny_at_or_above: RiskLevel = "CRITICAL",
        allowlist: set[str] | None = None,
    ) -> None:
        self.deny_at_or_above = deny_at_or_above
        self.allowlist: set[str] = allowlist or set()

    def evaluate(
        self,
        tool: Tool,
        input: BaseModel,
        agent_capabilities: set[str],
    ) -> PolicyDecision | None:
        if tool.risk not in _RISK_ORDER:
            return None  # unknown tier — let other rules decide
        if _RISK_ORDER[tool.risk] >= _RISK_ORDER[self.deny_at_or_above]:
            if tool.name in self.allowlist:
                return None  # explicitly allowlisted — abstain, others may decide
            return PolicyDecision(
                allowed=False,
                reason=(
                    f"risk tier {tool.risk} is denied by default "
                    f"(deny_at_or_above={self.deny_at_or_above}); "
                    "an explicit allowlist entry is required"
                ),
                rule=self.name,
            )
        return None  # below the deny threshold — abstain
