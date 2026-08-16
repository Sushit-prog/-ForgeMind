"""Policy engine tests.

The engine is pure: given a tool, validated input, and a capability set it
returns ALLOW or DENY deterministically — no I/O, no LLM, no free-text
reasoning. Every rule here is exercised over typed inputs only.
"""

import pytest
from pydantic import BaseModel

from app.policies.base import PolicyDecision, PolicyRule
from app.policies.engine import PolicyEngine
from app.policies.rules import ExplicitDenyRule, RiskTierRule
from app.tools.base import Tool


class _Input(BaseModel):
    value: str = "x"


class _Output(BaseModel):
    ok: bool = True


def make_tool(tool_name: str, risk: str = "LOW", capabilities: list[str] | None = None) -> Tool:
    # Alias to distinct names: a class body can't read an enclosing function
    # local that it also assigns (``name = name`` is a NameError), but it CAN
    # read one it never assigns (``risk = _risk`` works fine).
    _risk = risk
    _caps = capabilities or []

    class _T(Tool):
        name = tool_name
        description = "test tool"
        input_schema = _Input
        output_schema = _Output
        capabilities = _caps
        risk = _risk

        async def execute(self, input: _Input, ctx) -> _Output:
            return _Output()

    return _T()


class AllowAllRule(PolicyRule):
    """A rule that always votes ALLOW — must never override another DENY."""

    name = "allow-all-test"

    def evaluate(self, tool, input, agent_capabilities) -> PolicyDecision | None:
        return PolicyDecision(allowed=True, reason="test rule allows", rule=self.name)


# --- risk-default rule ------------------------------------------------------

def test_risk_tier_denies_critical_unless_allowlisted() -> None:
    tool = make_tool("critical_tool", risk="CRITICAL")
    engine = PolicyEngine(rules=[RiskTierRule()])
    decision = engine.evaluate(tool, _Input(), set())
    assert not decision.allowed
    assert decision.rule == "risk-default"
    assert "CRITICAL" in decision.reason


def test_risk_tier_allowlists_critical_tool() -> None:
    tool = make_tool("critical_tool", risk="CRITICAL")
    engine = PolicyEngine(rules=[RiskTierRule(allowlist={"critical_tool"})])
    assert engine.evaluate(tool, _Input(), set()).allowed


def test_risk_tier_abstains_below_threshold() -> None:
    # No CRITICAL tools exist in Phase 3; LOW/MEDIUM/HIGH pass the default.
    engine = PolicyEngine(rules=[RiskTierRule()])
    for risk in ("LOW", "MEDIUM", "HIGH"):
        tool = make_tool(f"t_{risk}", risk=risk)
        assert engine.evaluate(tool, _Input(), set()).allowed, risk


def test_risk_tier_respects_custom_threshold() -> None:
    tool_high = make_tool("high_tool", risk="HIGH")
    engine = PolicyEngine(rules=[RiskTierRule(deny_at_or_above="HIGH")])
    assert not engine.evaluate(tool_high, _Input(), set()).allowed
    tool_medium = make_tool("medium_tool", risk="MEDIUM")
    assert engine.evaluate(tool_medium, _Input(), set()).allowed


# --- explicit deny rule -----------------------------------------------------

def test_explicit_deny_denies_listed_tool() -> None:
    tool = make_tool("banned")
    engine = PolicyEngine(rules=[ExplicitDenyRule(denied_tools={"banned"})])
    decision = engine.evaluate(tool, _Input(), set())
    assert not decision.allowed
    assert decision.rule == "explicit-deny"
    assert "banned" in decision.reason


def test_explicit_deny_abstains_for_other_tools() -> None:
    engine = PolicyEngine(rules=[ExplicitDenyRule(denied_tools={"banned"})])
    assert engine.evaluate(make_tool("fine"), _Input(), set()).allowed


# --- engine: fail-closed, deny wins -----------------------------------------

def test_deny_wins_over_allow_vote() -> None:
    """One rule ALLOW + one rule DENY must resolve to DENY (fail-closed)."""
    engine = PolicyEngine(
        rules=[AllowAllRule(), ExplicitDenyRule(denied_tools={"banned"})]
    )
    decision = engine.evaluate(make_tool("banned"), _Input(), set())
    assert not decision.allowed
    assert decision.rule == "explicit-deny"


def test_allow_vote_does_not_short_circuit_later_rules() -> None:
    """An ALLOW from an earlier rule must not prevent a later DENY."""
    engine = PolicyEngine(
        rules=[AllowAllRule(), ExplicitDenyRule(denied_tools={"banned"})]
    )
    decision = engine.evaluate(make_tool("banned"), _Input(), set())
    assert not decision.allowed  # even though the first rule allowed


def test_default_engine_allows_when_no_rule_denies() -> None:
    engine = PolicyEngine()
    decision = engine.evaluate(make_tool("example.echo"), _Input(), set())
    assert decision.allowed
    assert decision.rule == "default-allow"


def test_default_engine_denies_example_denied_tool() -> None:
    """The shipped default policy set denies example.denied by name."""
    engine = PolicyEngine()
    decision = engine.evaluate(make_tool("example.denied"), _Input(), set())
    assert not decision.allowed
    assert decision.rule == "explicit-deny"


def test_rules_are_pure_no_io() -> None:
    """Rules never touch the DB, network, or clocks — pure functions.

    Guard: each rule is instantiated with no arguments and evaluated with
    only typed inputs. (This is a structural check; the design review point
    is that the engine has no I/O surface at all — see the module docstring.)
    """
    for rule in (RiskTierRule(), ExplicitDenyRule()):
        assert rule.name
        decision = rule.evaluate(make_tool("whatever"), _Input(), set())
        assert decision is None or isinstance(decision, PolicyDecision)
