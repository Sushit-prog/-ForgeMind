"""Application configuration.

All settings are loaded from environment variables / a local ``.env`` file.
Secrets live only in the environment — nothing is hardcoded here, and the
settings object must never be logged in full (see ``app.logging``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Fallback bearer token for development/test when FORGEMIND_API_TOKEN is
# unset. Deterministic so curl examples, docs, and hermetic tests can rely on
# it. Production has NO default — it FAILS CLOSED at startup instead (see
# ``Settings._ensure_api_token``).
DEV_API_TOKEN = "forgemind-dev-token"


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

    # API auth (Phase 10.5): a SINGLE shared-secret bearer token gating the
    # mutating routes (POST /tasks, cancel, approve, reject) — single-operator
    # scope, no user accounts. Loaded from FORGEMIND_API_TOKEN only (the env
    # var name is set via validation_alias below), never hardcoded, never
    # logged. Falls back to DEV_API_TOKEN in development/test; production with
    # no token refuses to start (see ``_ensure_api_token``).
    api_token: str | None = Field(
        default=None,
        validation_alias="FORGEMIND_API_TOKEN",
        description="Bearer token for mutating API routes (secret — never logged).",
    )

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
        default=None,
        description="Model for the planning agent (env: LLM_MODEL_PLANNER).",
    )
    llm_model_debugger: str | None = Field(
        default=None,
        description="Model for the debugger agent (env: LLM_MODEL_DEBUGGER).",
    )
    llm_timeout_seconds: float = Field(
        default=60.0, description="Per-LLM-call timeout."
    )
    llm_max_retries: int = Field(
        default=2, description="Bounded transient (timeout/5xx) retries per call."
    )
    # Research agent (Phase 6): hard cap on tool calls per task before a
    # forced synthesis — the budget-limiting pattern from Section 42.
    max_research_tool_calls: int = Field(
        default=10,
        description="Max tool calls per research run (env: MAX_RESEARCH_TOOL_CALLS).",
    )
    # Developer agent (Phase 7): hard cap on tool calls per task. Exhausting
    # the budget with no commit is a hard failure (not forced synthesis) — an
    # implementation without a commit is nothing.
    max_developer_tool_calls: int = Field(
        default=20,
        description="Max tool calls per developer run (env: MAX_DEVELOPER_TOOL_CALLS).",
    )
    # Debugger agent (Phase 8): hard cap on investigation tool calls per task
    # before a forced classification.
    max_debugger_tool_calls: int = Field(
        default=10,
        description="Max tool calls per debugger run (env: MAX_DEBUGGER_TOOL_CALLS).",
    )
    # shell.run_test (Phase 8): hard timeout on the test subprocess. A hung
    # suite times out into status "error" (never "failed") so the Debugger
    # can tell a hang from a clean failing exit code.
    test_timeout_seconds: float = Field(
        default=300.0,
        description="Timeout for the test subprocess (env: TEST_TIMEOUT_SECONDS).",
    )
    # Reviewer + Security agents (Phase 9): hard caps on their read-only
    # investigation tool calls before a forced verdict. Models per role env
    # vars (LLM_MODEL_REVIEWER, LLM_MODEL_SECURITY).
    max_reviewer_tool_calls: int = Field(
        default=10,
        description="Max tool calls per reviewer run (env: MAX_REVIEWER_TOOL_CALLS).",
    )
    max_security_tool_calls: int = Field(
        default=10,
        description="Max tool calls per security run (env: MAX_SECURITY_TOOL_CALLS).",
    )
    llm_model_reviewer: str | None = Field(
        default=None,
        description="Model for the reviewer agent (env: LLM_MODEL_REVIEWER).",
    )
    llm_model_security: str | None = Field(
        default=None,
        description="Model for the security agent (env: LLM_MODEL_SECURITY).",
    )
    llm_model_research: str | None = Field(
        default=None,
        description="Model for the research agent (env: LLM_MODEL_RESEARCH).",
    )
    llm_model_developer: str | None = Field(
        default=None,
        description="Model for the developer agent (env: LLM_MODEL_DEVELOPER).",
    )
    # Optional per-role FALLBACK CHAINS: comma-separated model slugs tried in
    # order when the primary model keeps returning transient errors (429
    # free-tier rate limits above all — see FallbackLLMProvider). Empty/unset
    # means no fallback: single-model behavior unchanged.
    llm_model_planner_fallbacks: str | None = Field(
        default=None,
        description=(
            "Comma-separated fallback models for the planning agent "
            "(env: LLM_MODEL_PLANNER_FALLBACKS)."
        ),
    )
    llm_model_research_fallbacks: str | None = Field(
        default=None,
        description=(
            "Comma-separated fallback models for the research agent "
            "(env: LLM_MODEL_RESEARCH_FALLBACKS)."
        ),
    )
    llm_model_developer_fallbacks: str | None = Field(
        default=None,
        description=(
            "Comma-separated fallback models for the developer agent "
            "(env: LLM_MODEL_DEVELOPER_FALLBACKS)."
        ),
    )
    llm_model_debugger_fallbacks: str | None = Field(
        default=None,
        description=(
            "Comma-separated fallback models for the debugger agent "
            "(env: LLM_MODEL_DEBUGGER_FALLBACKS)."
        ),
    )
    llm_model_reviewer_fallbacks: str | None = Field(
        default=None,
        description=(
            "Comma-separated fallback models for the reviewer agent "
            "(env: LLM_MODEL_REVIEWER_FALLBACKS)."
        ),
    )
    llm_model_security_fallbacks: str | None = Field(
        default=None,
        description=(
            "Comma-separated fallback models for the security agent "
            "(env: LLM_MODEL_SECURITY_FALLBACKS)."
        ),
    )
    # GitHub integration (Phase 10). GITHUB_TOKEN is the ONLY credential and
    # a SECRET — loaded from env only, never hardcoded, never logged. The
    # base URL is configurable for GitHub Enterprise / self-hosted gateways.
    github_token: str | None = Field(
        default=None,
        description="GitHub personal access token (secret — never logged).",
    )
    github_base_url: str = Field(
        default="https://api.github.com",
        description="GitHub REST API base URL (self-hosted GitHub supported).",
    )
    github_api_timeout_seconds: float = Field(
        default=20.0, description="Per-GitHub-request timeout."
    )
    github_max_retries: int = Field(
        default=3,
        description="Bounded transient (429/5xx/timeout) retries per GitHub call.",
    )

    @model_validator(mode="after")
    def _ensure_api_token(self) -> "Settings":
        """Fail closed: no API runs in production without an explicit token.

        In development/test an unset token falls back to ``DEV_API_TOKEN`` so
        key-less dev and the hermetic suite still work. In production a missing
        token raises at config load — the API process refuses to start rather
        than silently serving unauthenticated mutating routes.
        """
        if not self.api_token:
            if self.environment == "production":
                raise ValueError(
                    "FORGEMIND_API_TOKEN must be set when ENVIRONMENT=production — "
                    "refusing to run the API unauthenticated."
                )
            self.api_token = DEV_API_TOKEN
        return self


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (env is read once per process)."""
    return Settings()
