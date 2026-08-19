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
    # The FORK ForgeMind pushes to / opens PRs against (Phase 10). Optional:
    # when unset, git.push and github.create_pr fail closed at PR_CREATION.
    fork_url: str | None = Field(default=None, max_length=2048)
    # Optional GitHub issue this task originates from (target of
    # github.get_issue + the PR-comment link).
    issue_number: int | None = Field(default=None, ge=1)

    @field_validator("repository_url")
    @classmethod
    def validate_repository_url(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("repository_url must not be empty")

        if GIT_URL_PATTERNS[1].match(value):  # scp-style, no scheme — fine as-is
            return value

        parts = urlsplit(value)
        # file:// is accepted for local development/testing (clone a local
        # repo); production URLs use the remote schemes.
        if parts.scheme not in {"http", "https", "ssh", "git", "file"}:
            raise ValueError(
                "repository_url must be a well-formed git URL "
                "(e.g. https://github.com/org/repo.git or git@github.com:org/repo.git)"
            )
        if parts.scheme != "file" and (
            not parts.netloc or not parts.path or parts.path.strip("/") == ""
        ):
            raise ValueError("repository_url must include a host and a repository path")
        return value

    @field_validator("fork_url")
    @classmethod
    def validate_fork_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("fork_url must not be empty")
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https", "ssh", "git", "file"}:
            raise ValueError(
                "fork_url must be a well-formed git URL "
                "(e.g. https://github.com/you/fork.git or git@github.com:you/fork.git)"
            )
        return value


class ApprovalRequest(BaseModel):
    """Request body for POST /tasks/{id}/approve and /reject."""

    reason: str | None = Field(default=None, max_length=10_000)


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
