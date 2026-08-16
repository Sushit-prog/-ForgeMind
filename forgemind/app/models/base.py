"""Shared model infrastructure.

``Base`` uses a fixed naming convention so every constraint/index gets a
deterministic name — required for stable Alembic migrations and for
cross-database behavior (SQLite in tests, Postgres in dev/prod).

``JsonType`` maps to Postgres ``JSONB`` and falls back to generic ``JSON``
on SQLite, so tests can run without a live Postgres.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, MetaData
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase


def utcnow() -> datetime:
    """Client-side UTC timestamp (microsecond precision).

    Used as an ORM default alongside the DB ``server_default`` so ordering
    by ``created_at`` is deterministic even on SQLite, whose
    ``CURRENT_TIMESTAMP`` truncates to whole seconds.
    """
    return datetime.now(timezone.utc)

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)

# JSONB on Postgres, plain JSON everywhere else (SQLite in tests).
JsonType = JSONB().with_variant(JSON(), "sqlite")


class Base(DeclarativeBase):
    metadata = metadata
