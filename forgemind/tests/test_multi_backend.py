"""Multi-backend provider tests (hermetic).

Pins the generalization contract: one OpenAICompatibleProvider serves
OpenRouter, Groq, and NVIDIA - identical guard/retry behavior regardless
of backend, byte-for-byte unchanged requests for bare-slug (OpenRouter)
roles, per-role backend selection via slug prefixes, and fallback chains
that may hop ACROSS providers.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
from pydantic import BaseModel

import app.llm.openai_compat as compat
from app.config import get_settings
from app.agents.planner.agent import PlannerConfigError
from app.llm import LLMProviderError, is_transient_error
from app.llm.openai_compat import BACKENDS, OpenAICompatibleProvider
from app.llm.provider import Message


def run(coro):
    return asyncio.run(coro)


def _msg(text="hi") -> Message:
    return Message(role="user", content=text)


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = ""

    def json(self):
        return self._payload


def _patch_http(monkeypatch, *payloads) -> list[dict]:
    """Swap in a recording AsyncClient playing the payload queue.

    Patches compat.httpx.AsyncClient - compat.httpx IS the shared httpx
    package object, so this redirects every backend identically.
    """
    responses = list(payloads)
    requests: list[dict] = []

    class _Client:
        def __init__(self, **kwargs):
            self._responses = responses

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, headers=None, json=None):
            requests.append({"url": url, "headers": dict(headers or {}), "json": json})
            return (
                self._responses.pop(0)
                if len(self._responses) > 1
                else self._responses[0]
            )

    monkeypatch.setattr(compat.httpx, "AsyncClient", lambda **kw: _Client())
    return requests


def ok(content="hello"):
    return _FakeResponse(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ]
        }
    )


def null_content():
    return _FakeResponse(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "length",
                }
            ],
            "usage": {"completion_tokens": 5},
        }
    )


def no_choices_envelope():
    return _FakeResponse({"error": {"message": "All providers failed", "code": 502}})


def rate_limited():
    return _FakeResponse(
        {
            "error": {
                "message": "Rate limit exceeded: free-models-per-day",
                "code": 429,
            }
        },
        status_code=429,
    )


def _provider(base_url: str) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(api_key="test-key", base_url=base_url, model="m")


@pytest.mark.parametrize("backend", sorted(BACKENDS))
@pytest.mark.parametrize(
    "response_factory,expect_status",
    [
        (ok, None),
        (null_content, 503),
        (no_choices_envelope, 503),
        (rate_limited, 429),
    ],
)
def test_guard_parity_across_backends(
    monkeypatch, backend, response_factory, expect_status
):
    reqs = _patch_http(monkeypatch, response_factory())
    try:
        result = ("result", run(_provider(BACKENDS[backend]).generate([_msg()])))
    except LLMProviderError as exc:
        result = ("error", exc)

    kind, value = result
    if expect_status is None:
        assert kind == "result" and value == "hello"
    else:
        assert kind == "error"
        assert value.status_code == expect_status
        if expect_status == 503:
            assert is_transient_error(value)

    req = reqs[-1]
    assert req["url"] == f"{BACKENDS[backend]}/chat/completions"
    assert req["headers"]["Authorization"] == "Bearer test-key"
    assert req["headers"]["Content-Type"] == "application/json"
    assert req["json"]["model"] == "m"


def test_bare_slug_openrouter_request_is_unchanged(monkeypatch) -> None:
    reqs = _patch_http(monkeypatch, ok())
    p = OpenAICompatibleProvider(
        api_key="sk-or-test",
        base_url=BACKENDS["openrouter"],
        model="anthropic/claude-3.5-sonnet",
    )
    run(p.generate([_msg("hi")]))

    req = reqs[-1]
    assert req["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert req["headers"]["Authorization"] == "Bearer sk-or-test"
    assert req["json"] == {
        "model": "anthropic/claude-3.5-sonnet",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.2,
    }


def test_structured_output_body_json_mode(monkeypatch) -> None:
    class P(BaseModel):
        a: int = 1

    reqs = _patch_http(monkeypatch, ok('{"a": 1}'))
    p = OpenAICompatibleProvider(
        api_key="k", base_url=BACKENDS["groq"], model="openai/gpt-oss-120b"
    )
    obj = run(p.structured_output([], P))
    assert obj.a == 1
    assert reqs[-1]["json"]["response_format"] == {"type": "json_object"}
    assert reqs[-1]["json"]["temperature"] == 0.1


def test_fallback_chain_hops_across_backends(monkeypatch) -> None:
    from app.llm.fallback import FallbackLLMProvider

    reqs = _patch_http(
        monkeypatch,
        no_choices_envelope(),  # openrouter primary: transient 503 x budget
        no_choices_envelope(),
        no_choices_envelope(),
        ok("from nvidia"),  # nvidia hop succeeds first try
    )
    chain = FallbackLLMProvider(
        [
            ("cohere/north-mini-code:free", _provider(BACKENDS["openrouter"])),
            ("meta/llama-3.3-70b-instruct", _provider(BACKENDS["nvidia"])),
        ],
        max_retries=2,
        backoff_base_seconds=0.01,
    )

    result = run(chain.generate([_msg()]))

    assert result == "from nvidia"
    urls = [r["url"] for r in reqs]
    assert urls.count(f"{BACKENDS['openrouter']}/chat/completions") == 3
    assert urls[-1] == f"{BACKENDS['nvidia']}/chat/completions"


def test_split_backend_slug_cases() -> None:
    from app.llm.config import split_backend_slug

    assert split_backend_slug("groq::openai/gpt-oss-120b") == (
        "groq",
        "openai/gpt-oss-120b",
    )
    assert split_backend_slug("nvidia::meta/llama-3.3-70b-instruct") == (
        "nvidia",
        "meta/llama-3.3-70b-instruct",
    )
    assert split_backend_slug("cohere/north-mini-code:free") == (
        "openrouter",
        "cohere/north-mini-code:free",
    )
    assert split_backend_slug("  bare-model  ") == ("openrouter", "bare-model")


def _wire(monkeypatch, *, openrouter="", groq="", nvidia="", model="", fallbacks=""):
    """Configure role=research via env exactly as production would read it.

    Present-but-empty strings BEAT the dotenv layer (a real .env carries
    keys + fallback chains); delenv alone would let those leak through.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", openrouter)
    monkeypatch.setenv("GROQ_API_KEY", groq)
    monkeypatch.setenv("NVIDIA_API_KEY", nvidia)
    monkeypatch.setenv("LLM_MODEL_RESEARCH", model)
    monkeypatch.setenv("LLM_MODEL_RESEARCH_FALLBACKS", fallbacks)
    get_settings.cache_clear()
    try:
        from app.agents.planner.agent import build_provider

        return build_provider(role="research")
    finally:
        get_settings.cache_clear()


def test_wiring_groq_prefixed_role_targets_groq(monkeypatch) -> None:
    provider = cast(
        Any, _wire(monkeypatch, groq="gsk_x", model="groq::openai/gpt-oss-120b")
    )
    assert provider.base_url == BACKENDS["groq"]
    assert provider.model == "openai/gpt-oss-120b"
    assert provider.api_key == "gsk_x"


def test_wiring_bare_role_still_openrouter(monkeypatch) -> None:
    provider = cast(
        Any,
        _wire(monkeypatch, openrouter="sk-or", model="cohere/north-mini-code:free"),
    )
    assert provider.base_url == BACKENDS["openrouter"]
    assert provider.model == "cohere/north-mini-code:free"


def test_wiring_missing_backend_key_fails_loudly(monkeypatch) -> None:
    with pytest.raises(PlannerConfigError) as exc_info:
        _wire(
            monkeypatch,
            nvidia="",
            model="nvidia::nvidia/nemotron-3-ultra-550b-a55b",
        )
    assert isinstance(exc_info.value, PlannerConfigError)
    text = str(exc_info.value)
    assert "role 'research'" in text
    assert "backend 'nvidia'" in text
    assert "NVIDIA_API_KEY" in text


def test_wiring_unknown_backend_rejected(monkeypatch) -> None:
    with pytest.raises(PlannerConfigError) as exc_info:
        _wire(monkeypatch, model="grog::whatever")
    assert isinstance(exc_info.value, RuntimeError)
    assert "unknown backend 'grog'" in str(exc_info.value)


def test_wiring_mixed_chain_builds_two_backends(monkeypatch) -> None:
    provider = cast(
        Any,
        _wire(
            monkeypatch,
            openrouter="sk-or",
            groq="gsk_x",
            model="groq::openai/gpt-oss-120b",
            fallbacks="cohere/north-mini-code:free",
        ),
    )
    from app.llm.fallback import FallbackLLMProvider

    assert isinstance(provider, FallbackLLMProvider)
    models = provider.models
    assert models[0] == "groq::openai/gpt-oss-120b"
    assert models[1] == "cohere/north-mini-code:free"
