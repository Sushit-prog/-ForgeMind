"""Deprecated module location — the provider has been generalized.

Historically this module held ``OpenRouterProvider``. The class now lives
in :mod:`app.llm.openai_compat` as the backend-agnostic
``OpenAICompatibleProvider`` (OpenRouter, Groq, NVIDIA, or any
OpenAI-compatible gateway). This shim re-exports it under the old name so
existing imports keep working.

Model slugs may carry a backend prefix (``"groq::…"``, ``"nvidia::…"``);
bare slugs still default to the OpenRouter endpoint. See
:func:`app.llm.config.split_backend_slug`.
"""

from __future__ import annotations

from app.llm.openai_compat import (
    TRANSIENT_STATUSES,
    OpenAICompatibleProvider,
    is_transient_error,
)

OpenRouterProvider = OpenAICompatibleProvider

__all__ = [
    "TRANSIENT_STATUSES",
    "OpenAICompatibleProvider",
    "OpenRouterProvider",
    "is_transient_error",
]
