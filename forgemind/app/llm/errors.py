"""LLM provider errors (architecture doc section 34)."""

from __future__ import annotations


class LLMError(RuntimeError):
    """Base class for all LLM-provider failures."""


class LLMTimeoutError(LLMError):
    """The provider did not answer within the configured timeout.

    Treated as a TRANSIENT failure: the planner retries with bounded
    backoff. Never fatal on first occurrence.
    """


class LLMMalformedOutputError(LLMError):
    """The provider returned output that could not be coerced into the
    requested schema — malformed JSON, or valid JSON with the wrong shape.

    The raw output is attached for debugging/reproducibility; nothing is
    returned to the caller (never a partially-valid object).
    """

    def __init__(self, raw_output: str, detail: str = "") -> None:
        self.raw_output = raw_output
        self.detail = detail
        message = f"LLM output did not match schema{': ' + detail if detail else ''}"
        super().__init__(message)


class LLMProviderError(LLMError):
    """The provider returned a non-success HTTP response.

    Carries the status code so callers can distinguish transient (429/5xx)
    from permanent (401/400) failures.
    """

    def __init__(self, status_code: int, body: str = "") -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"LLM provider error (HTTP {status_code}): {body[:300]}")
