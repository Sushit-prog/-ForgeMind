"""``example.echo`` — LOW risk, requires no capability.

Proves the happy path: validates, passes every gate, executes, audits.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.tools.base import ExecutionContext, Tool


class EchoInput(BaseModel):
    message: str = Field(min_length=1, max_length=1000)


class EchoOutput(BaseModel):
    message: str


class EchoTool(Tool):
    name = "example.echo"
    description = "Echoes back the message. Harmless demonstration tool."
    input_schema = EchoInput
    output_schema = EchoOutput
    capabilities: list[str] = []
    risk = "LOW"

    async def execute(self, input: EchoInput, ctx: ExecutionContext) -> EchoOutput:
        return EchoOutput(message=input.message)
