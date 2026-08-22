"""LLM provider boundary tests.

The contract under test: ``structured_output`` either returns a fully
valid object or raises ``LLMMalformedOutputError`` with the raw output
attached — malformed JSON and valid-JSON-with-wrong-shape are BOTH
rejected (never a lenient parse), and code-fenced JSON is tolerated.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.agents.planner.schema import Plan
from app.llm import (
    LLMMalformedOutputError,
    LLMProviderError,
    LLMTimeoutError,
    StubLLMProvider,
    is_transient_error,
    parse_and_validate,
)
from app.llm.mock import DEFAULT_PLAN_RESPONSE, MALFORMED_RESPONSE, default_by_schema

VALID = json.loads(DEFAULT_PLAN_RESPONSE)


def test_valid_response_passes_through() -> None:
    plan = parse_and_validate(DEFAULT_PLAN_RESPONSE, Plan)
    assert isinstance(plan, Plan)
    assert plan.steps[0].step_type == "research"


def test_malformed_json_raises_with_raw_attached() -> None:
    with pytest.raises(LLMMalformedOutputError) as exc_info:
        parse_and_validate(MALFORMED_RESPONSE, Plan)
    assert exc_info.value.raw_output == MALFORMED_RESPONSE
    assert "invalid JSON" in exc_info.value.detail


def test_valid_json_wrong_schema_raises() -> None:
    wrong = json.dumps({"objective": "x", "steps": [{"id": 1}]})  # id is int
    with pytest.raises(LLMMalformedOutputError):
        parse_and_validate(wrong, Plan)


def test_missing_fields_raises() -> None:
    missing = json.dumps({"objective": "x"})  # no steps
    with pytest.raises(LLMMalformedOutputError):
        parse_and_validate(missing, Plan)


def test_code_fenced_json_is_tolerated() -> None:
    fenced = f"```json\n{DEFAULT_PLAN_RESPONSE}\n```"
    plan = parse_and_validate(fenced, Plan)
    assert isinstance(plan, Plan)


def test_unknown_step_type_raises() -> None:
    bad = dict(VALID)
    bad["steps"][0]["step_type"] = "delete-everything"
    with pytest.raises(LLMMalformedOutputError):
        parse_and_validate(json.dumps(bad), Plan)


# --- stub provider ----------------------------------------------------------

def test_stub_structured_output_returns_valid_plan() -> None:
    import asyncio

    provider = StubLLMProvider(by_schema=default_by_schema())
    plan = asyncio.run(provider.structured_output([], Plan))  # type: ignore[arg-type]
    assert isinstance(plan, Plan)


def test_stub_structured_output_raises_on_malformed() -> None:
    import asyncio

    provider = StubLLMProvider(responses=[MALFORMED_RESPONSE])
    with pytest.raises(LLMMalformedOutputError):
        asyncio.run(provider.structured_output([], Plan))  # type: ignore[arg-type]


def test_stub_repeats_last_response() -> None:
    import asyncio

    provider = StubLLMProvider(responses=[MALFORMED_RESPONSE])
    with pytest.raises(LLMMalformedOutputError):
        asyncio.run(provider.structured_output([], Plan))  # type: ignore[arg-type]
    with pytest.raises(LLMMalformedOutputError):
        asyncio.run(provider.structured_output([], Plan))  # type: ignore[arg-type]


def test_stub_generate_returns_raw_string() -> None:
    import asyncio

    provider = StubLLMProvider(responses=["hello"])
    assert asyncio.run(provider.generate([])) == "hello"  # type: ignore[arg-type]


# --- transient classification ----------------------------------------------

def test_is_transient_error_classification() -> None:
    assert is_transient_error(LLMTimeoutError("slow"))
    assert is_transient_error(LLMProviderError(429, "rate limited"))
    assert is_transient_error(LLMProviderError(503, "unavailable"))
    assert not is_transient_error(LLMProviderError(401, "bad key"))
    assert not is_transient_error(LLMProviderError(400, "bad request"))
    assert not is_transient_error(ValueError("unrelated"))


# --- null-content guard (run #3 incident: reasoning models answering
# --- HTTP 200 with content=null crashed parse_and_validate on .strip()) ------

import logging

import app.llm.openrouter as openrouter_module
from app.llm.openrouter import OpenRouterProvider
from app.llm.provider import Message


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.text = ""

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, response, **_kwargs):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, *a, **k):
        return self._response


def _patch_http(monkeypatch, payload) -> None:
    monkeypatch.setattr(
        openrouter_module.httpx,
        "AsyncClient",
        lambda **kwargs: _FakeAsyncClient(_FakeResponse(payload)),
    )


def _provider() -> OpenRouterProvider:
    return OpenRouterProvider(api_key="k", model="test-model")


def test_null_content_raises_transient_503_with_diagnostics(
    monkeypatch, caplog
) -> None:
    payload = {
        "id": "resp-1",
        "model": "cohere/north-mini-code:free",
        "choices": [
            {"message": {"role": "assistant", "content": None}, "finish_reason": "length"}
        ],
        "usage": {"completion_tokens": 410},
    }
    _patch_http(monkeypatch, payload)

    with caplog.at_level(logging.WARNING, logger="app.llm.openrouter"):
        with pytest.raises(LLMProviderError) as exc_info:
            asyncio.run(_provider().generate([Message(role="user", content="hi")]))

    assert exc_info.value.status_code == 503
    assert is_transient_error(exc_info.value), "null content must engage retry+fallback"
    assert "finish_reason=length" in str(exc_info.value)
    warning = [r for r in caplog.records if "null content" in r.getMessage()]
    assert warning, "diagnostic warning must be logged"
    text = warning[0].getMessage()
    assert "length" in text and "completion_tokens" in text and "'id'" in text


def test_missing_message_is_unexpected_shape_not_crash(monkeypatch) -> None:
    payload = {"choices": [{"message": None}]}
    _patch_http(monkeypatch, payload)

    with pytest.raises(LLMProviderError) as exc_info:
        asyncio.run(_provider().generate([Message(role="user", content="hi")]))

    assert "unexpected response shape" in str(exc_info.value)
