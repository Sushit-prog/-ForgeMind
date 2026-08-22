"""OpenRouter provider — OpenAI-compatible ``/chat/completions``.

The API key comes from settings (env / .env) and is never logged; the
base URL is configurable so a self-hosted OpenAI-compatible gateway can
be swapped in without code changes.
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


def is_transient_error(exc: Exception) -> bool:
    """True for errors worth a bounded retry: timeouts + transient statuses."""
    if isinstance(exc, LLMTimeoutError):
        return True
    return isinstance(exc, LLMProviderError) and exc.status_code in TRANSIENT_STATUSES


class OpenRouterProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def _headers(self) -> dict[str, str]:
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
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMProviderError(
                resp.status_code, f"unexpected response shape: {exc}"
            ) from exc
        if content is None:
            # Reasoning models on free tiers occasionally answer HTTP 200
            # with content=null. Downstream .strip() would crash the agent
            # on a bare AttributeError; classified as 503 instead so the
            # bounded retry + fallback hop treat it as availability.
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
