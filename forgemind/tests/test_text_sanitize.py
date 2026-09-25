"""NUL / lone-surrogate scrubbing across the ingest + persistence boundaries.

PostgreSQL JSONB rejects ``\\u0000`` — even as a valid JSON escape — with
"unsupported Unicode escape sequence" (SQLSTATE 22P05). That is the exact
error that crashed the research job of task 28cb92ec-… and stranded it in
RESEARCHING. SQLite's plain JSON accepts ``\\u0000``, so the regression is
pinned at the scrubber level rather than by replaying the DB reject.
"""

from app.execution.tool_pipeline import sanitize_persistable
from app.repository.file_access import sanitize_text


def test_nul_byte_replaced_with_replacement_char():
    assert sanitize_text("a\x00b") == "a\ufffdb"


def test_lone_high_surrogate_replaced():
    assert sanitize_text("a\uD800b") == "a\ufffdb"


def test_lone_low_surrogate_replaced():
    assert sanitize_text("a\uDFFFb") == "a\ufffdb"


def test_valid_surrogate_pair_preserved():
    emoji = "\U0001F9E0"
    assert sanitize_text(emoji) == emoji


def test_clean_text_is_identity():
    assert sanitize_text("plain text 123") == "plain text 123"


def test_sanitize_persistable_recurses():
    value = {
        "snippet": "\x00lead",
        "items": [{"note": "x\x00y"}, "\uD800"],
        "keep": 3.5,
        "flag": True,
    }
    out = sanitize_persistable(value)
    assert out["snippet"] == "\ufffdlead"
    assert out["items"][0]["note"] == "x\ufffdy"
    assert out["items"][1] == "\ufffd"
    assert out["keep"] == 3.5
    assert out["flag"] is True