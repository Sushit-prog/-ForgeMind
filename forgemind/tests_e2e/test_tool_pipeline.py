"""E2E: the three example tools demonstrate all three pipeline outcomes
(ALLOW+execute, DENY on missing capability, DENY on policy) against real
Postgres — and every invocation produces exactly one ``tool_calls`` row.
"""

from __future__ import annotations

import asyncio

from sqlalchemy import func, select

from app.execution import ToolPipeline, make_execution_context
from app.models import ToolCall

CTX = make_execution_context(agent_type="developer")


def run(coro):
    return asyncio.run(coro)


def count_rows(db_session) -> int:
    return db_session.scalar(select(func.count()).select_from(ToolCall))


def test_three_tools_three_outcomes_one_row_each(client, db_session) -> None:
    pipeline = ToolPipeline(db=db_session)

    echo = run(pipeline.invoke("example.echo", {"message": "hello"}, set(), CTX))
    assert echo.status == "EXECUTED"
    assert echo.output == {"message": "hello"}

    denied_cap = run(pipeline.invoke("example.read_file", {"path": "a.py"}, set(), CTX))
    assert denied_cap.status == "DENIED"
    assert "repo.read" in denied_cap.denial_reason

    allowed_cap = run(
        pipeline.invoke("example.read_file", {"path": "a.py"}, {"repo.read"}, CTX)
    )
    assert allowed_cap.status == "EXECUTED"

    denied_policy = run(pipeline.invoke("example.denied", {"command": "x"}, set(), CTX))
    assert denied_policy.status == "DENIED"
    assert "explicitly denied" in denied_policy.denial_reason

    # Exactly one row per invocation: 4 calls above -> 4 rows.
    assert count_rows(db_session) == 4

    rows = db_session.scalars(select(ToolCall)).all()
    by_tool = {row.tool_name: row for row in rows}
    assert by_tool["example.echo"].status == "EXECUTED"
    assert by_tool["example.read_file"].status == "EXECUTED"  # the allowed call
    assert by_tool["example.denied"].status == "DENIED"
    # The denied read_file row is the other one — its denial_reason mentions repo.read.
    denied_rows = [r for r in rows if r.status == "DENIED"]
    assert any("repo.read" in (r.denial_reason or "") for r in denied_rows)
    # Risk tiers recorded as declared.
    assert by_tool["example.denied"].risk == "HIGH"
    assert by_tool["example.echo"].risk == "LOW"
