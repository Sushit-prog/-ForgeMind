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
    """Bounded listing. When ``truncated`` is set, ``files`` ends with an
    explicit sentinel entry so the CONSUMER (an LLM) cannot miss that it is
    seeing a partial view, and ``total_entries`` carries the true count."""

    files: list[str]
    truncated: bool = False
    total_entries: int | None = None


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

    async def execute(
        self, input: ReadFileInput, ctx: ExecutionContext
    ) -> ReadFileOutput:
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

    async def execute(
        self, input: ListFilesInput, ctx: ExecutionContext
    ) -> ListFilesOutput:
        from app.config import get_settings

        files = FileAccess(_root_for(ctx, input.worktree_id)).list_files(input.path)
        total = len(files)
        max_entries = get_settings().list_files_max_entries
        if total <= max_entries:
            # Under-cap trees keep today's exact behavior (alphabetical,
            # complete) — bounded output only exists for oversized trees.
            return ListFilesOutput(files=files)
        # Depth-first ordering (shallowest paths first): the agent gets a
        # progressively deeper map of the repo instead of an alphabetical
        # slab owned by whichever directory sorts first.
        files.sort(key=lambda p: (p.count("/"), p))
        kept = files[:max_entries]
        kept.append(
            f"[truncated: showing {max_entries} of {total} total entries "
            "-- use a more specific path to see more]"
        )
        return ListFilesOutput(files=kept, truncated=True, total_entries=total)


REPOSITORY_TOOLS: list[Tool] = [ReadFileTool(), SearchTool(), ListFilesTool()]
