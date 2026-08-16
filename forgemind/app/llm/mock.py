"""Stub LLM provider.

A deterministic provider for hermetic tests and for running the worker
without an API key (``FORGEMIND_MOCK_LLM=1``). It behaves EXACTLY like the
real provider at the boundary that matters: it parses/validates its canned
responses through the same ``parse_and_validate`` path, so a "malformed"
canned response raises ``LLMMalformedOutputError`` exactly as OpenRouter
would — the planner's retry logic is exercised for real.

Responses are consumed in order; the last one repeats. The default is a
schema-valid plan so key-less runs complete normally.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from app.llm.provider import LLMProvider, Message, parse_and_validate

DEFAULT_PLAN_RESPONSE = json.dumps(
    {
        "objective": "fix the bug",
        "steps": [
            {
                "id": "research-1",
                "step_type": "research",
                "description": "Locate the faulty code and its tests",
                "depends_on": [],
            },
            {
                "id": "implement-1",
                "step_type": "implement",
                "description": "Implement the fix",
                "depends_on": ["research-1"],
            },
            {
                "id": "test-1",
                "step_type": "test",
                "description": "Run the test suite",
                "depends_on": ["implement-1"],
            },
            {
                "id": "review-1",
                "step_type": "review",
                "description": "Review the diff",
                "depends_on": ["test-1"],
            },
        ],
    }
)

MALFORMED_RESPONSE = "{this is not json"


class StubLLMProvider(LLMProvider):
    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses) if responses else [DEFAULT_PLAN_RESPONSE]
        self._index = 0
        self.generate_calls: list[list[Message]] = []
        self.structured_calls: list[list[Message]] = []

    def _next(self) -> str:
        if self._index >= len(self._responses):
            self._index = len(self._responses) - 1
        response = self._responses[self._index]
        self._index += 1
        return response

    def set_responses(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._index = 0

    async def generate(self, messages: list[Message], **kwargs: object) -> str:
        self.generate_calls.append(messages)
        return self._next()

    async def structured_output(
        self, messages: list[Message], schema: type[BaseModel], **kwargs: object
    ) -> BaseModel:
        self.structured_calls.append(messages)
        raw = self._next()
        return parse_and_validate(raw, schema)
