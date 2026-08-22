"""Multi-model fallback provider (availability, not correctness).

Wraps an ORDERED chain of ``(model_name, provider)`` pairs sharing one
api_key/base_url/timeout and differing only in ``model``. A request runs
against the FIRST model with the same bounded transient retry used
everywhere else (timeouts + 408/429/500/502/503/504 — see
``is_transient_error`` and ``structured_output_with_retries``); only when
THAT model's retry budget is exhausted does the request move to the next
model fresh (its own budget). OpenRouter free-tier models rate-limit PER
MODEL (429), so one exhausted model must not take down the whole pipeline.

Non-transient failures (400/401/…, malformed output) propagate
immediately — those are real errors, not availability problems; fallback
is never used to paper over them. Only when EVERY model in the chain is
exhausted does the call raise (the last transient error).

Each hop is logged (which model failed, why, which model it fell to) so a
fallback event is visible in the audit trail alongside tool-call logs.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Sequence, cast

from pydantic import BaseModel

from app.llm.openrouter import is_transient_error
from app.llm.provider import LLMProvider, Message

logger = logging.getLogger(__name__)


class FallbackLLMProvider(LLMProvider):
    """Ordered model chain; hops only on TRANSIENT-exhaustion boundaries."""

    def __init__(
        self,
        chain: Sequence[tuple[str | None, LLMProvider]],
        *,
        max_retries: int = 2,
        backoff_base_seconds: float = 0.5,
    ) -> None:
        if not chain:
            raise ValueError("FallbackLLMProvider needs a non-empty model chain")
        self._chain = list(chain)
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds

    @property
    def model(self) -> str | None:
        """The primary (first) model in the chain."""
        return self._chain[0][0]

    @property
    def models(self) -> list[str | None]:
        """Every model in the chain, in fallback order (introspection)."""
        return [model_name for model_name, _ in self._chain]

    async def generate(self, messages: list[Message], **kwargs: object) -> str:
        return cast(
            "str", await self._call_with_fallback("generate", messages, **kwargs)
        )

    async def structured_output(
        self,
        messages: list[Message],
        schema: type[BaseModel],
        **kwargs: object,
    ) -> BaseModel:
        return cast(
            "BaseModel",
            await self._call_with_fallback(
                "structured_output", messages, schema, **kwargs
            ),
        )

    async def _call_with_fallback(
        self,
        method_name: str,
        messages: list[Message],
        *args: object,
        **kwargs: object,
    ) -> object:
        """Run ONE request against the chain; hop only on transient exhaustion."""
        last_error: Exception | None = None
        for index, (model_name, provider) in enumerate(self._chain):
            attempt = 0
            while True:
                try:
                    call = getattr(provider, method_name)
                    return await call(messages, *args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if not is_transient_error(exc):
                        raise  # correctness problem — propagate, no fallback
                    if attempt >= self.max_retries:
                        break  # this model's budget is spent: hop to the next
                await asyncio.sleep(self.backoff_base_seconds * (2**attempt))
                attempt += 1

            next_model = (
                self._chain[index + 1][0] if index + 1 < len(self._chain) else None
            )
            logger.warning(
                "llm fallback: model %s exhausted after %d attempts (%s); "
                "falling to model %s",
                model_name,
                self.max_retries + 1,
                last_error,
                next_model,
            )
        # Every model in the chain exhausted its bounded budget.
        assert last_error is not None  # noqa: S101 — non-empty chain guarantees it
        raise last_error
