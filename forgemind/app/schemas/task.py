"""Pydantic schemas for the Task API.

Validation rules:
- ``objective`` must be non-empty.
- ``repository_url`` must be a well-formed git URL (SSH or HTTPS forms).
  Invalid/missing values surface as HTTP 422 via FastAPI's validator handling.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

GIT_URL_PATTERNS = (
    # https://github.com/org/repo(.git)? — also http, ssh://, git://
    re.compile(r"^(https?|ssh|git)://[^\s/$.?#].[^\s]*$"),
    # scp-style: git@github.com:org/repo.git
    re.compile(r"^[^@\s]+@[^:\s]+:[^\s]+$"),
)


class TaskCreate(BaseModel):
    """Request body for POST /tasks."""

    objective: str = Field(min_length=1, max_length=100_000)
    repository_url: str = Field(min_length=1, max_length=2048)

    @field_validator("repository_url")
    @classmethod
    def validate_repository_url(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("repository_url must not be empty")

        if GIT_URL_PATTERNS[1].match(value):  # scp-style, no scheme — fine as-is
            return value

        parts = urlsplit(value)
        if parts.scheme not in {"http", "https", "ssh", "git"}:
            raise ValueError(
                "repository_url must be a well-formed git URL "
                "(e.g. https://github.com/org/repo.git or git@github.com:org/repo.git)"
            )
        if not parts.netloc or not parts.path or parts.path.strip("/") == "":
            raise ValueError("repository_url must include a host and a repository path")
        return value


class TaskRead(BaseModel):
    """Response body for a task record."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    objective: str
    repository_id: uuid.UUID
    status: str
    replan_count: int = 0
    created_at: datetime
    updated_at: datetime


class ExecutionEventRead(BaseModel):
    """Response body for one execution-event record."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    task_id: uuid.UUID
    from_status: str
    to_status: str
    reason: str | None
    created_at: datetime
