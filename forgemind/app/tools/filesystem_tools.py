"""``filesystem.*`` tools (architecture doc section F).

``filesystem.write_file`` is the Developer Agent's write path. The security
boundary is REUSED, not reimplemented: the tool calls ``FileAccess.write_file``,
which resolves the path through the exact same ``_resolve`` containment check
as reads — a ``../`` climb, absolute path, or symlink escape is rejected with
``PathTraversalError`` (a ``SecurityError``) before anything is written. This
is a NEW attack surface (write, not just read), so the traversal defense is
exercised by its own adversarial test, not assumed covered by the read side.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from app.git.worktree_manager import WorktreeManager
from app.repository.file_access import FileAccess
from app.tools.base import ExecutionContext, Tool

# Upper bound on a single write: LLM input is untrusted, so content size is
# bounded at the schema before it ever reaches the filesystem.
MAX_WRITE_CONTENT = 1_000_000


class WriteFileInput(BaseModel):
    worktree_id: uuid.UUID
    path: str = Field(min_length=1, max_length=2048)
    content: str = Field(max_length=MAX_WRITE_CONTENT)


class WriteFileOutput(BaseModel):
    path: str
    existed: bool  # True = modified an existing file, False = created a new one


class WriteFileTool(Tool):
    name = "filesystem.write_file"
    description = (
        "Write file content inside the task's worktree (traversal-safe); "
        "creates parent directories for new files, overwrites existing ones."
    )
    input_schema = WriteFileInput
    output_schema = WriteFileOutput
    capabilities: list[str] = ["repo.write"]
    risk = "MEDIUM"

    async def execute(self, input: WriteFileInput, ctx: ExecutionContext) -> WriteFileOutput:
        if ctx.db is None:
            raise RuntimeError("ExecutionContext.db is required for filesystem tools")
        root = WorktreeManager(ctx.db).path_for(input.worktree_id)
        existed = FileAccess(root).write_file(input.path, input.content)
        return WriteFileOutput(path=input.path, existed=existed)


FILESYSTEM_TOOLS: list[Tool] = [WriteFileTool()]
