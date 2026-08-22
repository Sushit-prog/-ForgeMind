"""Backend-agnostic OpenAI-compatible provider.

One class serves every OpenAI-compatible chat/completions backend —
OpenRouter, Groq's ``api.groq.com/openai/v1``, NVIDIA's
``integrate.api.nvidia.com/v1``, or any self-hosted gateway — parameterized
by ``base_url`` + ``api_key`` + ``model``. All response-shape hardening is
backend-agnostic by construction (it operates on the OpenAI envelope, not
on vendor-specific fields):

- HTTP 200 with no ``choices`` (error envelope) -> transient 503
- HTTP 200 with null/missing content            -> transient 503
- malformed envelope                            -> explicit shape error

so the bounded retry + FallbackLLMProvider machinery behaves identically no
matter which backend a role is pointed at.

Model slugs may carry a backend prefix (``"groq::openai/gpt-oss-120b"``,
``"nvidia::meta/llama-3.3-70b-instruct"``); bare slugs default to the
OpenRouter endpoint. See :func:`app.llm.config.split_backend_slug`.
"""

from __future__ import annotations

import logging

import httpx
from pydantic import BaseModel

from app.llm.errors import LLMProviderError, LLMTimeoutError
from app.llm.provider import LLMProvider, Message, parse_and_validate

logger = logging.getLogger(__name__)

# HTTP statuses treated as transient (retried by the caller with backoff).
TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

# Known OpenAI-compatible backends and their base URLs. API keys are NOT
# stored here — they resolve from Settings per backend at provider build
# time so a missing key fails loudly with the role's name attached.
BACKENDS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
}


def is_transient_error(exc: Exception) -> bool:
    """True for errors worth a bounded retry: timeouts + transient statuses."""
    if isinstance(exc, LLMTimeoutError):
        return True
    return isinstance(exc, LLMProviderError) and exc.status_code in TRANSIENT_STATUSES


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str,
        model: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def _headers(self) -> dict[str, str]:
        # Plain Bearer + JSON: the intersection of what OpenRouter, Groq,
        # and NVIDIA all accept — no vendor-specific headers anywhere.
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _chat(
        self,
        messages: list[Message],
        *,
        model: str | None,
        json_mode: bool,
        temperature: float,
        max_tokens: int | None,
    ) -> str:
        model_name = model or self.model
        if not model_name:
            raise LLMProviderError(
                400, "no model configured — set LLM_MODEL_PLANNER (or per-role env)"
            )
        body: dict = {
            "model": model_name,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=body,
                )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"provider timed out after {self.timeout_seconds}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError(0, f"transport error: {exc}") from exc

        if resp.status_code != 200:
            raise LLMProviderError(resp.status_code, resp.text)
        try:
            payload = resp.json()
        except ValueError as exc:
            raise LLMProviderError(
                resp.status_code, f"unexpected response shape: {exc}"
            ) from exc
        if not isinstance(payload, dict) or not payload.get("choices"):
            # Backends occasionally answer HTTP 200 with NO choices at all
            # — typically an error envelope after upstream routing failed
            # (OpenRouter does this when every provider for the model is
            # exhausted). That is availability, not correctness: 503 so the
            # bounded retry + fallback hop engage instead of crashing the
            # agent or silently blocking hops deeper in the chain.
            err = payload.get("error") if isinstance(payload, dict) else None
            err_summary = (
                {k: err.get(k) for k in ("message", "code", "metadata") if k in err}
                if isinstance(err, dict)
                else err
            )
            logger.warning(
                "llm provider returned no choices (error=%s top_level_keys=%s)",
                err_summary,
                sorted(payload.keys())
                if isinstance(payload, dict)
                else type(payload).__name__,
            )
            message = err.get("message") if isinstance(err, dict) else err
            raise LLMProviderError(
                503, f"provider returned no choices (error={message})"
            )
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMProviderError(
                resp.status_code, f"unexpected response shape: {exc}"
            ) from exc
        if content is None:
            # Reasoning models on constrained tiers occasionally answer
            # HTTP 200 with content=null. Downstream .strip() would crash
            # the agent on a bare AttributeError; classified as 503 instead
            # so bounded retry + fallback hop treat it as availability.
            finish_reason = payload["choices"][0].get("finish_reason")
            logger.warning(
                "llm provider returned null content "
                "(finish_reason=%s usage=%s top_level_keys=%s)",
                finish_reason,
                payload.get("usage"),
                sorted(payload.keys()),
            )
            raise LLMProviderError(
                503, f"provider returned null content (finish_reason={finish_reason})"
            )
        return content

    async def generate(self, messages: list[Message], **kwargs: object) -> str:
        return await self._chat(
            messages,
            model=kwargs.get("model"),  # type: ignore[arg-type]
            json_mode=False,
            temperature=float(kwargs.get("temperature", 0.2)),
            max_tokens=kwargs.get("max_tokens"),  # type: ignore[arg-type]
        )

    async def structured_output(
        self,
        messages: list[Message],
        schema: type[BaseModel],
        **kwargs: object,
    ) -> BaseModel:
        raw = await self._chat(
            messages,
            model=kwargs.get("model"),  # type: ignore[arg-type]
            json_mode=True,
            temperature=float(kwargs.get("temperature", 0.1)),
            max_tokens=kwargs.get("max_tokens"),  # type: ignore[arg-type]
        )
        return parse_and_validate(raw, schema)
