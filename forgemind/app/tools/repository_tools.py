"""``repository.*`` tools (architecture doc section F).

Every tool takes a ``worktree_id`` — resolved server-side to a path via
``WorktreeManager.path_for``. An agent can say "read config.py" but can
never say "read /etc/passwd": the path is interpreted relative to the
worktree root and anything escaping it is rejected (``PathTraversalError``,
a ``SecurityError`` that the pipeline records as a FAILED call).
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from app.git.worktree_manager import WorktreeManager
from app.repository.file_access import FileAccess
from app.repository.models import SearchMatch
from app.tools.base import ExecutionContext, Tool


class ReadFileInput(BaseModel):
    worktree_id: uuid.UUID
    path: str = Field(min_length=1, max_length=2048)


class ReadFileOutput(BaseModel):
    path: str
    content: str


class SearchInput(BaseModel):
    worktree_id: uuid.UUID
    query: str = Field(min_length=1, max_length=500)
    glob: str | None = Field(default=None, max_length=500)


class SearchOutput(BaseModel):
    matches: list[SearchMatch]


class ListFilesInput(BaseModel):
    worktree_id: uuid.UUID
    path: str = Field(default=".", max_length=2048)


class ListFilesOutput(BaseModel):
    files: list[str]


def _root_for(ctx: ExecutionContext, worktree_id: uuid.UUID):
    """Resolve worktree_id -> root path; tools fail clearly without a DB."""
    if ctx.db is None:
        raise RuntimeError("ExecutionContext.db is required for repository tools")
    return WorktreeManager(ctx.db).path_for(worktree_id)


class ReadFileTool(Tool):
    name = "repository.read_file"
    description = "Read a file from the task's worktree (traversal-safe)."
    input_schema = ReadFileInput
    output_schema = ReadFileOutput
    capabilities: list[str] = ["repo.read"]
    risk = "LOW"

    async def execute(self, input: ReadFileInput, ctx: ExecutionContext) -> ReadFileOutput:
        content = FileAccess(_root_for(ctx, input.worktree_id)).read_file(input.path)
        return ReadFileOutput(path=input.path, content=content)


class SearchTool(Tool):
    name = "repository.search"
    description = "Case-insensitive text search inside the worktree."
    input_schema = SearchInput
    output_schema = SearchOutput
    capabilities: list[str] = ["repo.read"]
    risk = "LOW"

    async def execute(self, input: SearchInput, ctx: ExecutionContext) -> SearchOutput:
        matches = FileAccess(_root_for(ctx, input.worktree_id)).search(
            input.query, glob=input.glob
        )
        return SearchOutput(matches=matches)


class ListFilesTool(Tool):
    name = "repository.list_files"
    description = "List files in the worktree (optionally under a directory)."
    input_schema = ListFilesInput
    output_schema = ListFilesOutput
    capabilities: list[str] = ["repo.read"]
    risk = "LOW"

    async def execute(self, input: ListFilesInput, ctx: ExecutionContext) -> ListFilesOutput:
        files = FileAccess(_root_for(ctx, input.worktree_id)).list_files(input.path)
        return ListFilesOutput(files=files)


REPOSITORY_TOOLS: list[Tool] = [ReadFileTool(), SearchTool(), ListFilesTool()]
