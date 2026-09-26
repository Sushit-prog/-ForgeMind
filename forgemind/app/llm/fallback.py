"""Multi-model fallback provider (availability, not correctness).

Wraps an ORDERED chain of ``(model_name, provider)`` pairs sharing one
api_key/base_url/timeout and differing only in ``model``. A request runs
against the FIRST model with the same bounded transient retry used
everywhere else (timeouts + 0/408/429/500/502/503/504 — see
``is_transient_error`` and ``structured_output_with_retries``); only when
THAT model's retry budget is exhausted does the request move to the next
model fresh (its own budget). OpenRouter free-tier models rate-limit PER
MODEL (429), so one exhausted model must not take down the whole pipeline.

Non-transient failures (400/401/…) propagate immediately — those are real
errors, not availability problems. MALFORMED output (``LLMMalformedOutputError``)
gets a separate, BOUNDED treatment via ``fallback_on_malformed``:
free-tier models occasionally emit schema-invalid JSON in json_mode, so a
bounded number of malformed outputs hop to the next model in the chain
instead of crashing the agent — but never silently (if EVERY model comes
back malformed, or the hop budget is spent, the last malformed error still
propagates; a partially-valid object is never produced). Set
``fallback_on_malformed=False`` (or a ``malformed_hop_limit`` of 0) for the
exact old behavior — immediate propagation.

Only when EVERY model in the chain is exhausted does the call raise (the
last error). Each hop is logged (which model failed, why, which model it
fell to) so a fallback event is visible in the audit trail alongside
tool-call logs.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Sequence, cast

from pydantic import BaseModel

from app.llm.errors import LLMMalformedOutputError, LLMProviderError
from app.llm.openrouter import is_transient_error
from app.llm.provider import LLMProvider, Message

logger = logging.getLogger(__name__)


class FallbackLLMProvider(LLMProvider):
    """Ordered model chain; hops on transient-exhaustion boundaries and
    (bounded) malformed output."""

    # A 400 whose body is a JSON-validation complaint is the raw HTTP form
    # of malformed output (json_mode violated upstream, e.g. "Failed to
    # validate JSON"). The same applies to a provider rejecting a response
    # because the MODEL FORGED a tool call in a call the provider ran with
    # tool_choice=none (NVIDIA: "Tool choice is none, but model called a
    # tool", code tool_use_failed) — that is the raw-HTTP form of a
    # malformed PROPOSAL, and hop-mode should re-prompt the chain instead of
    # crashing the agent. A 400 for any OTHER reason (bad model name, auth,
    # oversized request…) is a real error and still propagates instantly.
    _JSON_CUE_TOKENS = ("json", "validate", "tool choice is none", "tool_use_failed")

    @classmethod
    def _is_malformed_output(cls, exc: Exception) -> bool:
        if isinstance(exc, LLMMalformedOutputError):
            return True
        if isinstance(exc, LLMProviderError) and exc.status_code == 400:
            lowered = str(exc.body).lower()
            return any(token in lowered for token in cls._JSON_CUE_TOKENS)
        return False

    @staticmethod
    def _as_malformed_error(exc: Exception) -> Exception:
        """Normalize any malformed-CLASSIFIED failure to LLMMalformedOutputError.

        Agents handle malformed output in ONE place (``except
        LLMMalformedOutputError``): a re-prompt or a bounded correction. A raw
        ``LLMProviderError(400, "…json…")`` that hop-mode re-raises VERBATIM
        would fall through every agent handler and surface as an uncaught
        crash, so once a failure has been classified malformed it must leave
        the chain typed as ``LLMMalformedOutputError`` (strict mode still
        re-raises verbatim — old behavior).
        """
        if isinstance(exc, LLMMalformedOutputError):
            return exc
        detail = str(exc.body) if isinstance(exc, LLMProviderError) else str(exc)
        return LLMMalformedOutputError(raw_output="", detail=f"provider 400: {detail}")

    def __init__(
        self,
        chain: Sequence[tuple[str | None, LLMProvider]],
        *,
        max_retries: int = 2,
        backoff_base_seconds: float = 0.5,
        fallback_on_malformed: bool = True,
        max_malformed_hops: int = 2,
    ) -> None:
        if not chain:
            raise ValueError("FallbackLLMProvider needs a non-empty model chain")
        self._chain = list(chain)
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.fallback_on_malformed = fallback_on_malformed
        # Total malformed-triggered hops allowed per request before the
        # malformed error propagates. 0 disables malformed fallback entirely.
        self.max_malformed_hops = max_malformed_hops

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
        """Run ONE request against the chain; hop only on transient
        exhaustion or (bounded) malformed output."""
        last_error: Exception | None = None
        malformed_hops = 0
        for index, (model_name, provider) in enumerate(self._chain):
            attempt = 0
            malformed = False
            while True:
                try:
                    call = getattr(provider, method_name)
                    return await call(messages, *args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if self._is_malformed_output(exc):
                        if not self.fallback_on_malformed:
                            raise  # strict mode: propagate malformed at once
                        malformed_hops += 1
                        if malformed_hops > self.max_malformed_hops:
                            logger.warning(
                                "llm fallback: model %s returned malformed output "
                                "%d time(s) (budget %d); propagating",
                                model_name,
                                malformed_hops,
                                self.max_malformed_hops,
                            )
                            # Budget spent — propagate AS malformed: no agent
                            # handles a raw 400/“json” provider error, so a
                            # classified-malformed failure must surface in the
                            # type every agent's handler expects.
                            raise self._as_malformed_error(exc)
                        malformed = True
                        break  # hop to the next model — do not retry SAME one
                    if not is_transient_error(exc):
                        raise  # correctness problem — propagate, no fallback
                    if attempt >= self.max_retries:
                        break  # this model's budget is spent: hop to the next
                await asyncio.sleep(self.backoff_base_seconds * (2**attempt))
                attempt += 1

            next_model = (
                self._chain[index + 1][0] if index + 1 < len(self._chain) else None
            )
            if malformed:
                logger.warning(
                    "llm fallback: model %s returned malformed output (%s); "
                    "falling to model %s (malformed hop %d/%d)",
                    model_name,
                    last_error,
                    next_model,
                    malformed_hops,
                    self.max_malformed_hops,
                )
            else:
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
        if self._is_malformed_output(last_error) and self.fallback_on_malformed:
            raise self._as_malformed_error(last_error)
        raise last_error
