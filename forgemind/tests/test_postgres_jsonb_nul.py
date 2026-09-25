"""PG-gated reproduction of the exact production failure (SQLSTATE 22P05).

PostgreSQL JSONB rejects ``\\u0000`` — ``psycopg.errors.UntranslatableCharacter:
unsupported Unicode escape sequence`` — the error that crashed task
28cb92ec-… at the ``tool_calls`` flush. SQLite's plain JSON accepts it, so the
real behavior is pinned here against a Postgres. Skipped unless
``FORGEMIND_TEST_POSTGRES`` is set (e.g. the compose ``db``:
``postgresql+psycopg://forgemind:forgemind@localhost:5433/forgemind``).
"""

import json
import os

import pytest
from sqlalchemy import create_engine, text

from app.repository.file_access import sanitize_text

PG_URL = os.environ.get("FORGEMIND_TEST_POSTGRES")

pytestmark = pytest.mark.skipif(
    not PG_URL,
    reason="set FORGEMIND_TEST_POSTGRES to a reachable Postgres",
)


def test_postgres_jsonb_rejects_raw_nul_but_accepts_scrubbed() -> None:
    engine = create_engine(PG_URL)
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE test_jsonb_nul_row (id text PRIMARY KEY, payload jsonb)"))

        # 1) The raw NUL-char string is rejected exactly as in production:
        # json.dumps serializes it to a \\u0000 escape, and the JSONB parser
        # refuses that escape (22P05 "unsupported Unicode escape sequence").
        raw_literal = json.dumps({"snippet": "a\x00b"})
        with engine.begin() as conn:
            with pytest.raises(Exception) as exc_info:
                conn.execute(
                    text("INSERT INTO test_jsonb_nul_row (id, payload) VALUES (:id, CAST(:p AS jsonb))"),
                    {"id": "raw", "p": raw_literal},
                )
        msg = str(exc_info.value)
        assert "unsupported" in msg.lower() or "untranslatable" in type(
            exc_info.value
        ).__name__.lower()

        # 2) The scrubbed value round-trips: sanitize_text replaces the NUL
        # char BEFORE json.dumps — exactly what sanitize_persistable does.
        scrubbed_literal = json.dumps({"snippet": sanitize_text("a\x00b")})
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO test_jsonb_nul_row (id, payload) VALUES (:id, CAST(:p AS jsonb))"),
                {"id": "scrubbed", "p": scrubbed_literal},
            )
        with engine.connect() as conn:
            got = conn.scalar(
                text("SELECT payload FROM test_jsonb_nul_row WHERE id = 'scrubbed'")
            )
            assert got == {"snippet": "a\ufffdb"}
            assert "\x00" not in got["snippet"]
    finally:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS test_jsonb_nul_row"))
        engine.dispose()