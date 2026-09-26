"""FallbackLLMProvider + build_provider role-selection tests (hermetic).

Two seams, no network:

1. ``build_provider(role)`` must honor its ``role`` argument on the REAL
   OpenRouter path — it once hardcoded the planner's model there, so every
   agent silently ran on LLM_MODEL_PLANNER. This file is the regression
   net that was missing when that shipped.
2. The fallback chain hops to the next model ONLY after a model's bounded
    transient-retry budget is exhausted (429 free-tier rate limits above
    all); non-transient errors propagate immediately — fallback is for
    availability, never for correctness. Malformed output gets its own
    BOUNDED hop (free-tier models emit schema-invalid JSON in json_mode):
    it may hop to the next model up to ``max_malformed_hops``, then the
    malformed error propagates — never an implicit success (see the
    malformed tests below).
"""

from __future__ import annotations

import asyncio

import pytest

from app.agents.planner.agent import build_provider
from app.agents.planner.schema import Plan
from app.config import get_settings
from app.llm import (
    LLMMalformedOutputError,
    LLMProviderError,
    OpenRouterProvider,
)
from app.llm.fallback import FallbackLLMProvider
from app.llm.mock import DEFAULT_PLAN_RESPONSE, MALFORMED_RESPONSE
from app.llm.provider import LLMProvider, Message, parse_and_validate


def run(coro):
    return asyncio.run(coro)


RATE_LIMITED = LLMProviderError(429, "rate limited")
BAD_REQUEST = LLMProviderError(400, "bad request")
UNAVAILABLE = LLMProviderError(503, "unavailable")


class ScriptedProvider(LLMProvider):
    """Plays a scripted sequence of results/exceptions; counts attempts.

    A single-entry script repeats forever (always-fail / always-succeed);
    multi-entry scripts are consumed left to right with the LAST entry
    repeating — mirroring StubLLMProvider's queue semantics.
    """

    def __init__(self, *script: str | Exception) -> None:
        self.script = list(script)
        self.attempts = 0

    def _next(self) -> str:
        self.attempts += 1
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return item

    async def generate(self, messages: list[Message], **kwargs: object) -> str:
        return self._next()

    async def structured_output(
        self, messages: list[Message], schema: type, **kwargs: object
    ):
        return parse_and_validate(self._next(), schema)


def _chain(a: ScriptedProvider, b: ScriptedProvider) -> FallbackLLMProvider:
    """A→B chain; tiny backoff so tests stay fast."""
    return FallbackLLMProvider(
        [("model-a", a), ("model-b", b)],
        max_retries=2,
        backoff_base_seconds=0.01,
    )


# --- fallback behavior -------------------------------------------------------


def test_429_exhaustion_falls_to_next_model() -> None:
    rate_limited = ScriptedProvider(RATE_LIMITED)  # 429 on EVERY attempt
    healthy = ScriptedProvider(DEFAULT_PLAN_RESPONSE)

    plan = run(_chain(rate_limited, healthy).structured_output([], Plan))  # type: ignore[arg-type]

    assert isinstance(plan, Plan)
    # Both models were REALLY attempted: A burned its whole budget
    # (initial + max_retries), B served the request fresh.
    assert rate_limited.attempts == 3
    assert healthy.attempts == 1


def test_non_transient_error_raises_immediately_never_falls_back() -> None:
    broken = ScriptedProvider(BAD_REQUEST)
    never_called = ScriptedProvider(DEFAULT_PLAN_RESPONSE)

    with pytest.raises(LLMProviderError) as exc_info:
        run(_chain(broken, never_called).structured_output([], Plan))  # type: ignore[arg-type]

    assert exc_info.value.status_code == 400
    assert broken.attempts == 1  # a real error is not retried…
    assert never_called.attempts == 0  # …and NEVER falls through to B


def test_malformed_output_hops_to_next_model_boundedly() -> None:
    # Default: bounded malformed fallback ON. A malformed response from the
    # primary hops to the next model — the free-tier reality that motivated it.
    malformed = ScriptedProvider(MALFORMED_RESPONSE)
    healthy = ScriptedProvider(DEFAULT_PLAN_RESPONSE)

    plan = run(_chain(malformed, healthy).structured_output([], Plan))  # type: ignore[arg-type]

    assert isinstance(plan, Plan)
    assert malformed.attempts == 1
    assert healthy.attempts == 1


def test_malformed_budget_exhausted_propagates_never_succeeds_silently() -> None:
    # Every model malformed: the LAST malformed error propagates after bounded
    # hops — a partially-valid object is never synthesized.
    malformed_a = ScriptedProvider(MALFORMED_RESPONSE)
    malformed_b = ScriptedProvider(MALFORMED_RESPONSE)

    with pytest.raises(LLMMalformedOutputError):
        run(_chain(malformed_a, malformed_b).structured_output([], Plan))  # type: ignore[arg-type]

    assert malformed_a.attempts == 1
    assert malformed_b.attempts == 1


def test_malformed_fallback_disabled_propagates_immediately() -> None:
    # fallback_on_malformed=False restores the old strict contract: malformed
    # propagates at once, the next model is never consulted.
    malformed = ScriptedProvider(MALFORMED_RESPONSE)
    never_called = ScriptedProvider(DEFAULT_PLAN_RESPONSE)
    chain = FallbackLLMProvider(
        [("model-a", malformed), ("model-b", never_called)],
        max_retries=2,
        backoff_base_seconds=0.01,
        fallback_on_malformed=False,
    )

    with pytest.raises(LLMMalformedOutputError):
        run(chain.structured_output([], Plan))  # type: ignore[arg-type]

    assert malformed.attempts == 1
    assert never_called.attempts == 0


def test_malformed_hop_limit_zero_behaves_like_disabled() -> None:
    malformed = ScriptedProvider(MALFORMED_RESPONSE)
    never_called = ScriptedProvider(DEFAULT_PLAN_RESPONSE)
    chain = FallbackLLMProvider(
        [("model-a", malformed), ("model-b", never_called)],
        max_retries=2,
        backoff_base_seconds=0.01,
        max_malformed_hops=0,
    )

    with pytest.raises(LLMMalformedOutputError):
        run(chain.structured_output([], Plan))  # type: ignore[arg-type]

    assert malformed.attempts == 1
    assert never_called.attempts == 0


def test_json_validate_400_hops_like_malformed() -> None:
    # The raw-HTTP form of malformed output: a 400 body complaining about
    # JSON validation (json_mode violated upstream). Same bounded hop.
    json_reject = ScriptedProvider(
        LLMProviderError(400, "Failed to validate JSON request: invalid json")
    )
    healthy = ScriptedProvider(DEFAULT_PLAN_RESPONSE)

    plan = run(_chain(json_reject, healthy).structured_output([], Plan))  # type: ignore[arg-type]

    assert isinstance(plan, Plan)
    assert json_reject.attempts == 1
    assert healthy.attempts == 1


def test_json_validate_400_chain_exhaustion_propagates_as_malformed_error() -> None:
    # REGRESSION (real-proof bug): when hop-mode classifies a FAILURE as
    # malformed (a raw LLMProviderError 400 whose body carries a json cue)
    # and the whole chain comes back that way, the propagating error must be
    # typed LLMMalformedOutputError — the type EVERY agent handler catches.
    # Re-raising the raw provider 400 verbatim escapes every handler and
    # surfaces as an uncaught agent crash (3 replans, then ESCALATED).
    json_reject_a = ScriptedProvider(
        LLMProviderError(400, "Failed to validate JSON request: invalid json")
    )
    json_reject_b = ScriptedProvider(
        LLMProviderError(400, "Failed to validate JSON request: invalid json")
    )

    with pytest.raises(LLMMalformedOutputError):
        run(_chain(json_reject_a, json_reject_b).structured_output([], Plan))  # type: ignore[arg-type]

    assert json_reject_a.attempts == 1
    assert json_reject_b.attempts == 1


def test_unrelated_400_still_propagates_immediately() -> None:
    # A 400 with NO json/validate cue (bad model name, auth, request size)
    # is a real error — hop must NOT engage, exactly as before.
    broken = ScriptedProvider(BAD_REQUEST)  # "bad request", no json cue
    never_called = ScriptedProvider(DEFAULT_PLAN_RESPONSE)

    with pytest.raises(LLMProviderError) as exc_info:
        run(_chain(broken, never_called).structured_output([], Plan))  # type: ignore[arg-type]

    assert exc_info.value.status_code == 400
    assert broken.attempts == 1
    assert never_called.attempts == 0


def test_all_models_exhausted_raises_last_transient_error() -> None:
    down_a = ScriptedProvider(RATE_LIMITED)
    down_b = ScriptedProvider(UNAVAILABLE)

    with pytest.raises(LLMProviderError):
        run(_chain(down_a, down_b).structured_output([], Plan))  # type: ignore[arg-type]

    # B received its OWN full retry budget, not A's leftovers.
    assert down_a.attempts == 3
    assert down_b.attempts == 3


def test_generate_path_falls_over_too() -> None:
    rate_limited = ScriptedProvider(RATE_LIMITED)
    healthy = ScriptedProvider("hello from b")

    result = run(_chain(rate_limited, healthy).generate([]))  # type: ignore[arg-type]

    assert result == "hello from b"
    assert rate_limited.attempts == 3
    assert healthy.attempts == 1


def test_single_model_chain_matches_plain_provider_behavior() -> None:
    healthy = ScriptedProvider(DEFAULT_PLAN_RESPONSE)
    provider = FallbackLLMProvider(
        [("only-model", healthy)], max_retries=2, backoff_base_seconds=0.01
    )

    plan = run(provider.structured_output([], Plan))  # type: ignore[arg-type]

    assert isinstance(plan, Plan)
    assert provider.model == "only-model"
    assert provider.models == ["only-model"]
    assert healthy.attempts == 1


def test_empty_chain_rejected_at_construction() -> None:
    with pytest.raises(ValueError):
        FallbackLLMProvider([])


# --- build_provider wiring ---------------------------------------------------


def test_build_provider_honors_role_on_real_provider_path(monkeypatch) -> None:
    """THE regression: build_provider used to hardcode get_model_for_role
    ("planner") on the real-OpenRouter branch, so every role silently ran
    on the planner's model."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.delenv("FORGEMIND_MOCK_LLM", raising=False)
    monkeypatch.setenv("LLM_MODEL_PLANNER", "planner-primary")
    monkeypatch.setenv("LLM_MODEL_RESEARCH", "research-primary")
    # Present-but-empty BEATS the dotenv layer (a real .env may carry
    # fallback chains); delenv alone would let .env values leak through.
    monkeypatch.setenv("LLM_MODEL_PLANNER_FALLBACKS", "")
    monkeypatch.setenv("LLM_MODEL_RESEARCH_FALLBACKS", "")
    get_settings.cache_clear()

    try:
        researcher = build_provider(role="research")

        assert isinstance(researcher, OpenRouterProvider)
        assert researcher.model == "research-primary"  # NOT planner-primary

        planner = build_provider(role="planner")
        assert planner.model == "planner-primary"
    finally:
        # Env is restored by monkeypatch; drop the cached Settings built
        # from the patched values so later tests re-read clean env.
        get_settings.cache_clear()


def test_build_provider_without_fallbacks_returns_plain_provider(monkeypatch) -> None:
    """Backward compat: no chain configured → plain single-model provider."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.delenv("FORGEMIND_MOCK_LLM", raising=False)
    monkeypatch.setenv("LLM_MODEL_DEBUGGER", "debugger-primary")
    # Empty-string beats the dotenv layer (see honors_role test).
    monkeypatch.setenv("LLM_MODEL_DEBUGGER_FALLBACKS", "")
    get_settings.cache_clear()

    try:
        provider = build_provider(role="debugger")

        assert isinstance(provider, OpenRouterProvider)
        assert not isinstance(provider, FallbackLLMProvider)
        assert provider.model == "debugger-primary"
    finally:
        get_settings.cache_clear()


def test_build_provider_wires_fallback_chain_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.delenv("FORGEMIND_MOCK_LLM", raising=False)
    monkeypatch.setenv("LLM_MODEL_RESEARCH", "research-primary")
    monkeypatch.setenv(
        "LLM_MODEL_RESEARCH_FALLBACKS",
        " cohere/north-mini-code:free ,, openai/gpt-oss-20b:free ",
    )
    get_settings.cache_clear()

    try:
        provider = build_provider(role="research")

        assert isinstance(provider, FallbackLLMProvider)
        assert provider.model == "research-primary"
        assert provider.models == [
            "research-primary",
            "cohere/north-mini-code:free",
            "openai/gpt-oss-20b:free",
        ]
        # Same connection config across the chain, only `model` differs.
        first, second = provider._chain[0][1], provider._chain[1][1]  # noqa: SLF001
        assert isinstance(first, OpenRouterProvider)
        assert isinstance(second, OpenRouterProvider)
        assert first.api_key == second.api_key == "sk-test"
        assert first.base_url == second.base_url
        assert first.timeout_seconds == second.timeout_seconds
        assert second.model == "cohere/north-mini-code:free"
    finally:
        get_settings.cache_clear()
