"""``example.denied`` — HIGH risk, deliberately denied by policy.

The ``ExplicitDenyRule`` (default policy set) denies this tool by name, so
``execute`` is never reached — this proves the DENY path end-to-end. It
models the shape of tools that will later be gated (e.g. ``shell.*``).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.tools.base import ExecutionContext, Tool


class DeniedInput(BaseModel):
    command: str = Field(min_length=1, max_length=500)


class DeniedOutput(BaseModel):
    result: str


class DeniedTool(Tool):
    name = "example.denied"
    description = "Deliberately denied by policy — execute() must never run."
    input_schema = DeniedInput
    output_schema = DeniedOutput
    capabilities: list[str] = []
    risk = "HIGH"

    async def execute(self, input: DeniedInput, ctx: ExecutionContext) -> DeniedOutput:
        # Unreachable while the explicit-deny policy is in place.
        return DeniedOutput(result=f"ran {input.command}")
