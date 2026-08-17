"""Stub LLM provider.

A deterministic provider for hermetic tests and for running the worker
without an API key (``FORGEMIND_MOCK_LLM=1``). It behaves EXACTLY like the
real provider at the boundary that matters: it parses/validates its canned
responses through the same ``parse_and_validate`` path, so a "malformed"
canned response raises ``LLMMalformedOutputError`` exactly as OpenRouter
would — the agents' retry logic is exercised for real.

Responses can be supplied two ways:

- ``by_schema``: a per-schema queue (``{"Plan": [...], "ToolCallProposal":
  [...], "ResearchArtifact": [...]}``). Each schema has its own index, so
  one provider safely serves multi-agent flows (planner, then research)
  without the queues colliding. The last response per schema repeats.
- ``responses``: a single flat queue (back-compat; consumed in order, last
  repeats). Used by the Phase-5 planner tests.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from app.llm.errors import LLMMalformedOutputError
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

# --- research agent canned responses (Phase 6) ------------------------------
# The search query matches the fixture repo (src/app.py contains "VALUE = 1"),
# so the artifact's relevant_files stay consistent with what was observed.

SEARCH_PROPOSAL = json.dumps(
    {"tool_call": {"tool": "repository.search", "input": {"query": "VALUE"}}}
)
FINAL_PROPOSAL = json.dumps({"final": True})
RESEARCH_ARTIFACT_RESPONSE = json.dumps(
    {
        "root_cause_hypothesis": "The bug is in src/app.py (stub hypothesis).",
        "relevant_files": ["src/app.py"],
        "relevant_tests": [],
        "evidence": ["Searched the worktree for 'VALUE'."],
        "confidence": 0.7,
    }
)

# --- developer agent canned responses (Phase 7) ------------------------------
# The write bumps src/app.py ("VALUE = 1" -> "VALUE = 2"), the commit captures
# it, and the summary reports exactly the file that was written, so the
# files-changed cross-check stays grounded.

WRITE_PROPOSAL = json.dumps(
    {
        "tool_call": {
            "tool": "filesystem.write_file",
            "input": {"path": "src/app.py", "content": "VALUE = 2\n"},
        }
    }
)
COMMIT_PROPOSAL = json.dumps(
    {"tool_call": {"tool": "git.commit", "input": {"message": "fix: bump VALUE to 2"}}}
)
IMPLEMENTATION_SUMMARY_RESPONSE = json.dumps(
    {
        "files_changed": ["src/app.py"],
        "summary": "Updated VALUE to 2 in src/app.py per the research hypothesis.",
        "tests_added": [],
        "deviations_from_research": None,
    }
)

# --- debugger agent canned responses (Phase 8) ------------------------------
# The debugger investigates (reads src/app.py), then classifies the failure
# as CODE_FAILURE — fixable, with a CONCRETE instruction. The fix instruction
# text doubles as the mock's retry marker: the developer's NEXT run (which
# receives this text as DATA) then writes the FIXED value (3) instead of the
# original (2), which is what makes the fail-once-then-pass scenario work
# end to end with canned responses.

DEBUGGER_READ_PROPOSAL = json.dumps(
    {"tool_call": {"tool": "repository.read_file", "input": {"path": "src/app.py"}}}
)
FIX_INSTRUCTION = "Update src/app.py so VALUE equals 3."
FAILURE_CLASSIFICATION_RESPONSE = json.dumps(
    {
        "category": "CODE_FAILURE",
        "root_cause": "The test expects VALUE to equal 3 but the implementation sets it to 2.",
        "fix_instruction": FIX_INSTRUCTION,
        "fixable": True,
    }
)

# The developer's retry write — the FIXED value, matching FIX_INSTRUCTION.
WRITE_RETRY_PROPOSAL = json.dumps(
    {
        "tool_call": {
            "tool": "filesystem.write_file",
            "input": {"path": "src/app.py", "content": "VALUE = 3\n"},
        }
    }
)


def _joined(messages: list[Message] | None) -> str:
    return "\n".join(m.content for m in (messages or []))


def default_by_schema(
    flaky_planner: bool = False, *, agent: str = "research"
) -> dict[str, list[str]]:
    """The worker's default per-schema script (planner + research + developer
    + debugger).

    Every agent builds its OWN provider instance in the worker, so the
    ``ToolCallProposal`` queue must be correct for that agent's FIRST tool
    call — research starts with a search (then final), the developer starts
    with a write (then commit, then final), the debugger starts with a read
    (then final). A developer run whose first proposal is a search (or
    final) would burn the whole run without committing — a hard failure,
    not a recoverable probe.
    """
    plan_queue = (
        [MALFORMED_RESPONSE, DEFAULT_PLAN_RESPONSE]
        if flaky_planner
        else [DEFAULT_PLAN_RESPONSE]
    )
    if agent == "developer":
        tool_queue = [WRITE_PROPOSAL, COMMIT_PROPOSAL, FINAL_PROPOSAL]
    elif agent == "debugger":
        tool_queue = [DEBUGGER_READ_PROPOSAL, FINAL_PROPOSAL]
    else:  # research
        tool_queue = [SEARCH_PROPOSAL, FINAL_PROPOSAL]
    return {
        "Plan": plan_queue,
        "ToolCallProposal": tool_queue,
        "ResearchArtifact": [RESEARCH_ARTIFACT_RESPONSE],
        "ImplementationSummaryDraft": [IMPLEMENTATION_SUMMARY_RESPONSE],
        "FailureClassification": [FAILURE_CLASSIFICATION_RESPONSE],
    }


class StubLLMProvider(LLMProvider):
    def __init__(
        self,
        responses: list[str] | None = None,
        by_schema: dict[str, list[str]] | None = None,
        *,
        retry_by_schema: dict[str, list[str]] | None = None,
        retry_marker: str | None = None,
    ) -> None:
        self._responses = list(responses) if responses else []
        # Bare constructor (Phase 5 tests) = the default per-schema script.
        # An explicit ``responses`` flat queue means explicit by_schema is
        # empty (the flat queue must not be shadowed by a default).
        if by_schema is None and not self._responses:
            by_schema = default_by_schema()
        self._by_schema = {name: list(rs) for name, rs in (by_schema or {}).items()}
        # Message-keyed queues: served INSTEAD of ``by_schema`` when the
        # prompt contains ``retry_marker`` (mock-only mechanism that lets a
        # canned agent respond differently to a debugger fix instruction).
        self._retry_by_schema = {
            name: list(rs) for name, rs in (retry_by_schema or {}).items()
        }
        self._retry_marker = retry_marker
        self._retry_index: dict[str, int] = {}
        self._index = 0
        self._schema_index: dict[str, int] = {}
        self.generate_calls: list[list[Message]] = []
        self.structured_calls: list[list[Message]] = []

    def _next(self, schema_name: str | None = None, messages: list[Message] | None = None) -> str:
        if (
            schema_name
            and self._retry_marker
            and schema_name in self._retry_by_schema
            and self._retry_marker in _joined(messages)
        ):
            queue = self._retry_by_schema[schema_name]
            idx = min(self._retry_index.get(schema_name, 0), len(queue) - 1)
            self._retry_index[schema_name] = idx + 1
            return queue[idx]
        if schema_name and schema_name in self._by_schema:
            queue = self._by_schema[schema_name]
            idx = min(self._schema_index.get(schema_name, 0), len(queue) - 1)
            self._schema_index[schema_name] = idx + 1
            return queue[idx]
        if not self._responses:
            raise LLMMalformedOutputError("", "no canned response for this schema")
        idx = min(self._index, len(self._responses) - 1)
        self._index = idx + 1
        return self._responses[idx]

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
        raw = self._next(schema.__name__, messages)
        return parse_and_validate(raw, schema)
