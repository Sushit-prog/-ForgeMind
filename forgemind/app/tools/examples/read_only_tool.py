"""``example.read_file`` — requires ``repo.read``.

Proves capability enforcement: denied for agents without the capability,
executed for agents that have it. Does no real file I/O (dummy for Phase 3).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.tools.base import ExecutionContext, Tool


class ReadInput(BaseModel):
    path: str = Field(min_length=1, max_length=2048)


class ReadOutput(BaseModel):
    path: str
    size_bytes: int


class ReadOnlyTool(Tool):
    name = "example.read_file"
    description = "Pretends to read a file from the repository (dummy)."
    input_schema = ReadInput
    output_schema = ReadOutput
    capabilities: list[str] = ["repo.read"]
    risk = "LOW"

    async def execute(self, input: ReadInput, ctx: ExecutionContext) -> ReadOutput:
        return ReadOutput(path=input.path, size_bytes=0)
