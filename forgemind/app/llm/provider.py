"""LLM provider abstraction (architecture doc section 34).

One interface, many providers (OpenRouter now; Ollama/local later). The
contract that matters for the rest of the system:

- ``structured_output`` NEVER returns a partially-valid object. It parses
  the raw response, validates it against the requested pydantic schema,
  and either returns a fully-valid instance or raises
  ``LLMMalformedOutputError`` with the raw output attached.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ValidationError

from app.llm.errors import LLMMalformedOutputError

MessageRole = Literal["system", "user", "assistant"]


class Message(BaseModel):
    role: MessageRole
    content: str


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_and_validate(raw_output: str, schema: type[BaseModel]) -> BaseModel:
    """Parse ``raw_output`` into ``schema``, or raise with raw attached.

    Tolerates a single ```json ...``` code fence around the payload (some
    models add one even when told not to); anything else must be plain
    JSON. Both malformed JSON and valid-JSON-with-wrong-shape raise
    ``LLMMalformedOutputError`` — a lenient parse is never allowed through.
    """
    text = raw_output.strip()
    fenced = _FENCE_RE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMMalformedOutputError(raw_output, f"invalid JSON: {exc}") from exc
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        raise LLMMalformedOutputError(
            raw_output, f"JSON did not match schema: {exc}"
        ) from exc


class LLMProvider(ABC):
    """Async LLM provider. Providers are stateless wrt calls; a fresh client
    per request is fine (cheap), and callers own retry policy."""

    @abstractmethod
    async def generate(
        self, messages: list[Message], **kwargs: object
    ) -> str:
        """Plain text completion — returns the assistant message content."""

    @abstractmethod
    async def structured_output(
        self,
        messages: list[Message],
        schema: type[BaseModel],
        **kwargs: object,
    ) -> BaseModel:
        """Schema-validated completion. Raises ``LLMMalformedOutputError``
        (raw output attached) rather than returning a partial object."""
