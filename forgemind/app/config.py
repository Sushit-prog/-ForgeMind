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

    # DB-side ceilings (Postgres only, applied at connect time). These are
    # the last line of defense against the duplicate-delivery deadlock
    # root-caused with py-spy: a sync FOR UPDATE blocked on a row lock froze
    # the entire arq event loop, so NO Python-level timer could fire.
    #
    # lock_timeout=10s is well under the worker's 870s inner deadline: a
    # stuck lock fails fast and loud (QueryCanceled -> arq retry) instead of
    # silently eating the whole budget. Normal contention is single-digit
    # milliseconds — one transition commits before the next job arrives.
    #
    # statement_timeout=300s is ~100x every observed statement (all OLTP-
    # shaped) yet bounds pathological scans. The tester's pytest subprocess
    # is unaffected: it runs in its own process, outside the DB session.
    db_lock_timeout_seconds: int = Field(
        default=10,
        ge=1,
        description="Postgres lock_timeout (seconds) applied per connection.",
    )
    db_statement_timeout_seconds: int = Field(
        default=300,
        ge=1,
        description="Postgres statement_timeout (seconds) applied per connection.",
    )

    # arq/Redis backing the worker queue (local dev: docker-compose redis).
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis URL for the arq task queue (never logged).",
    )

    # On worker startup, re-enqueue any non-terminal tasks left behind by a
    # crash (Section J: resume from last checkpoint). Disable in tests that
    # need exact control over queued jobs.
    worker_sweep_enabled: bool = True

    # Periodic stale-CREATED sweep (cron, ~every minute): heals an enqueue
    # LOST after POST /tasks committed the row — the startup sweep can't see
    # those because workers were already running when Redis dropped the job.
    # Threshold must sit comfortably above normal planning-phase latency so a
    # healthy CREATED task is never double-enqueued (a duplicate would be a
    # harmless no-op anyway: row lock + state machine).
    sweep_stale_created_seconds: int = Field(
        default=120,
        ge=1,
        description=(
            "Age (seconds) a CREATED task must reach before the periodic "
            "sweep re-enqueues it (env: SWEEP_STALE_CREATED_SECONDS)."
        ),
    )
    # Bounded retries for that sweep: a task re-enqueued this many times and
    # STILL stuck in CREATED is escalated to FAILED(enqueue_lost) with an
    # audit entry instead of sweeping forever.
    sweep_stale_created_max_attempts: int = Field(
        default=5,
        ge=1,
        description=(
            "Max periodic-sweep recoveries per task before escalation "
            "(env: SWEEP_STALE_CREATED_MAX_ATTEMPTS)."
        ),
    )

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
    groq_api_key: str | None = Field(
        default=None,
        description=(
            "Groq API key for roles whose model slug carries the 'groq::' "
            "backend prefix (secret — never logged)."
        ),
    )
    nvidia_api_key: str | None = Field(
        default=None,
        description=(
            "NVIDIA build.nvidia.com API key for roles whose model slug "
            "carries the 'nvidia::' backend prefix (secret — never logged)."
        ),
    )
    inception_api_key: str | None = Field(
        default=None,
        description=(
            "Inception Labs (Mercury) API key for roles whose model slug "
            "carries the 'inception::' backend prefix (secret — never logged)."
        ),
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
    llm_fallback_on_malformed: bool = Field(
        default=True,
        description=(
            "Bounded fallback to the next model when a free-tier model "
            "returns schema-invalid JSON (env: LLM_FALLBACK_ON_MALFORMED)."
        ),
    )
    llm_max_malformed_hops: int = Field(
        default=2,
        description=(
            "Max malformed-triggered hops per request before the malformed "
            "error propagates; 0 restores immediate propagation "
            "(env: LLM_MAX_MALFORMED_HOPS)."
        ),
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
    # Tool-call penalty cap (Phase 7 fix): non-executed rejections — a
    # pre-execution schema-validation error, a capability/policy DENY, or an
    # unknown tool — never count against the real (executed) tool budget.
    # Model param-phrasing drift (e.g. repository.search "pattern" vs "query")
    # must not starve the agent of budget before it lands one real call. This
    # is a SMALL fixed allowance of successive rejections, independent of
    # max_*_tool_calls, so a pathological model loops cannot run forever.
    max_tool_call_rejections: int = Field(
        default=5,
        description=(
            "Max successive non-executed tool rejections per run "
            "(env: MAX_TOOL_CALL_REJECTIONS)."
        ),
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
    # shell.install_deps (Phase 13): hard timeout on the dependency
    # provisioning subprocess (venv creation + pip install). If this fires
    # the task takes the dependency_install_failed path, NOT tests_error —
    # a hang resolving dependencies is not a test failure the Debugger can
    # classify, and it must never burn a replan debugging it.
    install_timeout_seconds: float = Field(
        default=150.0,
        description=(
            "Timeout for the dependency-install subprocess "
            "(env: INSTALL_TIMEOUT_SECONDS)."
        ),
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
    # repository.list_files output bounding: on a huge real-world tree
    # (45k+ files) the full listing once serialized to ~3 MB of tool output
    # (~775k tokens) and blew the model's context window outright.
    list_files_max_entries: int = Field(
        default=1000,
        description=(
            "Max entries returned by repository.list_files before truncation; "
            "results are ordered shallow-first and carry an explicit "
            "truncation notice (env: LIST_FILES_MAX_ENTRIES)."
        ),
    )
    # arq's outer job ceiling. Attempt #5 orphaned a RESEARCHING task at the
    # previous hardcoded 300s default: with bounded retries + fallback hops
    # engaged on a flaky free tier, a research round-trip can legitimately
    # exceed 300s. The worker additionally enforces its own INNER deadline
    # slightly below this value so a hang converts into an explicit
    # FAILED(job_timeout) — routed through recovery — instead of an orphan.
    worker_job_timeout_seconds: int = Field(
        default=900,
        ge=60,
        description=(
            "arq job_timeout ceiling for advance_task jobs "
            "(env: WORKER_JOB_TIMEOUT_SECONDS)."
        ),
    )
    # Optional direct override of the worker's INNER deadline (the
    # asyncio.timeout around one advance_task invocation). None = derive as
    # worker_job_timeout_seconds - 30 (floored at 60). Tests set this small
    # to prove the deadline actually fires while a DB call blocks in a
    # thread — the failure mode that was structurally inert before Fix 3.
    worker_inner_deadline_seconds: int | None = Field(
        default=None,
        ge=1,
        description="Explicit inner deadline override (env: WORKER_INNER_DEADLINE_SECONDS).",
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

    # Phase 12 — Merge gate. Comma-separated ``owner/repo`` fork slugs
    # ForgeMind is permitted to merge into. Empty/unset means merge is
    # disabled everywhere (fail-closed: the merge tool always returns
    # ``merged=False`` with ``"repo not in MERGE_ALLOWED_REPOS"``).
    merge_allowed_repos: str = Field(
        default="",
        description=(
            "Comma-separated owner/repo fork slugs with merge permission "
            "(env: MERGE_ALLOWED_REPOS). Empty = merge disabled everywhere."
        ),
    )

    @property
    def merge_allowed_repo_set(self) -> frozenset[str]:
        """Normalized, lowercased, whitespace-stripped set of allowed slugs."""
        return frozenset(
            entry.strip().lower()
            for entry in self.merge_allowed_repos.split(",")
            if entry.strip()
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
