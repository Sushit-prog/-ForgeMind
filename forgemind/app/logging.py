"""Logging configuration.

Deliberately minimal: one formatter, console output. The one hard rule is
that secrets (DB credentials, tokens) are never logged — ``redact_url``
exists so connection strings can be logged safely if ever needed.
"""

from __future__ import annotations

import logging
from urllib.parse import urlsplit, urlunsplit

from app.config import get_settings


def redact_url(url: str) -> str:
    """Strip userinfo (username:password) from a URL for safe logging."""
    parts = urlsplit(url)
    if parts.username is None and parts.password is None:
        return url
    hostname = parts.hostname or ""
    if parts.port:
        hostname = f"{hostname}:{parts.port}"
    return urlunsplit((parts.scheme, hostname, parts.path, parts.query, parts.fragment))


def setup_logging() -> None:
    """Configure root logging for the app based on settings."""
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        force=True,
    )
    # Alembic logs at INFO by default; keep noise down unless configured.
    logging.getLogger("alembic").setLevel(settings.log_level.upper())
    logging.getLogger("sqlalchemy.engine").setLevel(
        "WARNING" if settings.environment != "development" else settings.log_level.upper()
    )
