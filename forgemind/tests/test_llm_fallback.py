"""FallbackLLMProvider + build_provider role-selection tests (hermetic).

Two seams, no network:

1. ``build_provider(role)`` must honor its ``role`` argument on the REAL
   OpenRouter path — it once hardcoded the planner's model there, so every
   agent silently ran on LLM_MODEL_PLANNER. This file is the regression
   net that was missing when that shipped.
2. The fallback chain hops to the next model ONLY after a model's bounded
   transient-retry budget is exhausted (429 free-tier rate limits above
   all); non-transient errors propagate immediately — fallback is for
   availability, never for correctness.
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


def test_malformed_output_propagates_without_fallback() -> None:
    malformed = ScriptedProvider(MALFORMED_RESPONSE)
    never_called = ScriptedProvider(DEFAULT_PLAN_RESPONSE)

    with pytest.raises(LLMMalformedOutputError):
        run(_chain(malformed, never_called).structured_output([], Plan))  # type: ignore[arg-type]

    assert malformed.attempts == 1
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
