"""Deterministic policy engine (architecture doc sections F and H).

Hard constraints, enforced by construction:

- Every rule takes TYPED input (``Tool``, validated ``BaseModel`` input,
  ``set[str]`` capabilities) and returns a ``PolicyDecision`` or abstains.
- No rule may call an LLM, inspect free-text reasoning, or read the
  database. The engine is a pure function of its arguments.
- Fail-closed: the engine starts as ALLOW, and ANY rule that denies wins.
  A rule voting ALLOW never overrides another rule's DENY. This is the
  "deny wins" guarantee — see ``PolicyEngine.evaluate`` docstring.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.policies.base import PolicyDecision, PolicyRule
from app.policies.rules.explicit_deny import ExplicitDenyRule
from app.policies.rules.risk_default import RiskTierRule
from app.tools.base import Tool


def default_policy_rules() -> list[PolicyRule]:
    """The default rule set:

    - ``RiskTierRule`` — CRITICAL risk is denied unless explicitly allowlisted
      (no CRITICAL tools exist yet, so this is a no-op guard for later).
    - ``ExplicitDenyRule`` — denies ``example.denied`` by name, proving the
      DENY path. ``github.merge`` is denied by name as a SECOND layer under
      the missing capability: no merge capability or tool exists anywhere,
      and even if one were ever registered it would be denied here — this
      system never merges anything, ever.
    """
    return [
        RiskTierRule(),
        ExplicitDenyRule(
            denied_tools={"example.denied", "github.merge"},
        ),
    ]


class PolicyEngine:
    """Evaluates all rules; the first DENY wins, otherwise ALLOW.

    A rule that returns ALLOW does NOT short-circuit evaluation — later
    rules still get a chance to deny. That is what makes the engine
    fail-closed: the ALLOW/DENY conflict resolves to DENY by construction.
    """

    def __init__(self, rules: list[PolicyRule] | None = None) -> None:
        self.rules: list[PolicyRule] = (
            rules if rules is not None else default_policy_rules()
        )

    def evaluate(
        self,
        tool: Tool,
        input: BaseModel,
        agent_capabilities: set[str],
    ) -> PolicyDecision:
        for rule in self.rules:
            decision = rule.evaluate(tool, input, agent_capabilities)
            if decision is not None and not decision.allowed:
                # Fail-closed: first denial decides; later ALLOW votes lose.
                return decision
        return PolicyDecision(
            allowed=True,
            reason="no rule denied this call",
            rule="default-allow",
        )
