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


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (env is read once per process)."""
    return Settings()
