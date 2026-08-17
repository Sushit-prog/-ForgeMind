"""Application configuration.

All settings are loaded from environment variables / a local ``.env`` file.
Secrets live only in the environment — nothing is hardcoded here, and the
settings object must never be logged in full (see ``app.logging``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the ForgeMind API."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "forgemind"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    # SQLAlchemy URL. Local dev uses the docker-compose Postgres; production
    # points at Supabase Postgres (postgresql+psycopg://...). The password is
    # a secret and is never logged.
    database_url: str = Field(
        default="postgresql+psycopg://forgemind:forgemind@localhost:5433/forgemind",
        description="SQLAlchemy connection URL (never logged).",
    )

    # Fail-fast startup: how long to wait for the DB before giving up.
    db_connect_timeout_seconds: int = 5

    # arq/Redis backing the worker queue (local dev: docker-compose redis).
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis URL for the arq task queue (never logged).",
    )

    # On worker startup, re-enqueue any non-terminal tasks left behind by a
    # crash (Section J: resume from last checkpoint). Disable in tests that
    # need exact control over queued jobs.
    worker_sweep_enabled: bool = True

    # Master switch for the arq queue. When False (hermetic unit tests), the
    # API still creates/persists tasks but never touches Redis — the worker's
    # startup sweep would pick them up if a queue were present.
    queue_enabled: bool = True

    # Where repository clones and per-task worktrees live (Phase 4). Paths
    # are resolved relative to the process CWD unless absolute.
    repo_cache_dir: str = Field(
        default=".forgemind/repos",
        description="Directory for cached clones and per-task worktrees.",
    )

    # LLM provider (Phase 5). The API key is a SECRET — loaded from env only,
    # never hardcoded, never logged. Models are per-role env vars
    # (LLM_MODEL_PLANNER, ...) so swapping models is config, not code.
    openrouter_api_key: str | None = Field(
        default=None, description="OpenRouter API key (secret — never logged)."
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenAI-compatible base URL (self-hosted gateways supported).",
    )
    llm_model_planner: str | None = Field(
        default=None, description="Model for the planning agent (env: LLM_MODEL_PLANNER)."
    )
    llm_timeout_seconds: float = Field(default=60.0, description="Per-LLM-call timeout.")
    llm_max_retries: int = Field(
        default=2, description="Bounded transient (timeout/5xx) retries per call."
    )
    # Research agent (Phase 6): hard cap on tool calls per task before a
    # forced synthesis — the budget-limiting pattern from Section 42.
    max_research_tool_calls: int = Field(
        default=10, description="Max tool calls per research run (env: MAX_RESEARCH_TOOL_CALLS)."
    )
    # Developer agent (Phase 7): hard cap on tool calls per task. Exhausting
    # the budget with no commit is a hard failure (not forced synthesis) — an
    # implementation without a commit is nothing.
    max_developer_tool_calls: int = Field(
        default=20, description="Max tool calls per developer run (env: MAX_DEVELOPER_TOOL_CALLS)."
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (env is read once per process)."""
    return Settings()
