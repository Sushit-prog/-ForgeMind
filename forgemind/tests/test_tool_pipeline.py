"""ToolPipeline integration tests (validate -> capability -> policy ->
execute -> audit), run against the SQLite test DB.

The pipeline is async; tests drive it with ``asyncio.run`` so no plugin
configuration is needed.
"""

import asyncio
import uuid

import pytest
from pydantic import BaseModel
from sqlalchemy import func, select

from app.execution import (
    REDACTED,
    ToolInputValidationError,
    ToolPipeline,
    ToolResult,
    make_execution_context,
    redact_sensitive,
)
from app.models import ToolCall
from app.tools.base import ExecutionContext, Tool
from app.tools.examples import build_default_registry
from app.tools.registry import ToolNotFoundError, ToolRegistry


def run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def pipeline(db_session):
    return ToolPipeline(db=db_session)


CTX = make_execution_context(agent_type="developer", task_id=uuid.uuid4())


# --- test-only tools --------------------------------------------------------

class _Input(BaseModel):
    value: str


class _Output(BaseModel):
    ok: bool = True


class RaisingTool(Tool):
    name = "test.raises"
    description = "raises mid-execute to prove FAILED handling"
    input_schema = _Input
    output_schema = _Output
    risk = "LOW"

    async def execute(self, input: _Input, ctx: ExecutionContext) -> _Output:
        raise RuntimeError("boom mid-execute")


class TokenInput(BaseModel):
    message: str
    token: str


class TokenOutput(BaseModel):
    message: str
    token: str


class TokenTool(Tool):
    name = "test.token_echo"
    description = "echoes a value that must be redacted in the audit row"
    input_schema = TokenInput
    output_schema = TokenOutput
    risk = "LOW"

    async def execute(self, input: TokenInput, ctx: ExecutionContext) -> TokenOutput:
        return TokenOutput(message=input.message, token=input.token)


EXECUTE_CALLED: list[bool] = []


class SpyTool(Tool):
    name = "test.spy"
    description = "records whether execute was called"
    input_schema = _Input
    output_schema = _Output
    risk = "LOW"

    async def execute(self, input: _Input, ctx: ExecutionContext) -> _Output:
        EXECUTE_CALLED.append(True)
        return _Output()


def registry_with(*tools: Tool) -> ToolRegistry:
    reg = build_default_registry()
    for tool in tools:
        reg.register(tool)
    return reg


def count_rows(db_session) -> int:
    return db_session.scalar(select(func.count()).select_from(ToolCall))


def rows_for(db_session, tool_name: str) -> list[ToolCall]:
    return list(db_session.scalars(select(ToolCall).where(ToolCall.tool_name == tool_name)))


# --- happy path -------------------------------------------------------------

def test_echo_succeeds_with_one_executed_row(pipeline, db_session) -> None:
    result = run(pipeline.invoke("example.echo", {"message": "hi"}, set(), CTX))
    assert result.status == "EXECUTED"
    assert result.output == {"message": "hi"}
    assert result.latency_ms is not None and result.latency_ms >= 0

    rows = rows_for(db_session, "example.echo")
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "EXECUTED"
    assert row.risk == "LOW"
    assert row.input == {"message": "hi"}
    assert row.output == {"message": "hi"}
    assert row.denial_reason is None
    assert row.agent_type == "developer"
    assert row.task_id == CTX.task_id


# --- capability enforcement -------------------------------------------------

def test_read_file_denied_without_capability(pipeline, db_session) -> None:
    result = run(
        pipeline.invoke("example.read_file", {"path": "a/b.py"}, set(), CTX)
    )
    assert result.status == "DENIED"
    assert "repo.read" in result.denial_reason

    rows = rows_for(db_session, "example.read_file")
    assert len(rows) == 1
    assert rows[0].status == "DENIED"
    assert rows[0].denial_reason == result.denial_reason
    assert rows[0].output is None


def test_read_file_denied_with_empty_capability_set(pipeline, db_session) -> None:
    """Empty capability set -> DENY, not a crash."""
    result = run(pipeline.invoke("example.read_file", {"path": "x"}, set(), CTX))
    assert result.status == "DENIED"
    assert "repo.read" in result.denial_reason


def test_read_file_executes_with_capability(pipeline, db_session) -> None:
    result = run(
        pipeline.invoke(
            "example.read_file", {"path": "a/b.py"}, {"repo.read"}, CTX
        )
    )
    assert result.status == "EXECUTED"
    assert result.output == {"path": "a/b.py", "size_bytes": 0}
    assert len(rows_for(db_session, "example.read_file")) == 1


# --- policy deny ------------------------------------------------------------

def test_denied_tool_always_denied_by_policy(pipeline, db_session) -> None:
    result = run(
        pipeline.invoke("example.denied", {"command": "rm -rf /"}, set(), CTX)
    )
    assert result.status == "DENIED"
    assert "explicitly denied" in result.denial_reason

    rows = rows_for(db_session, "example.denied")
    assert len(rows) == 1
    assert rows[0].status == "DENIED"
    assert rows[0].risk == "HIGH"  # risk recorded even though denied


# --- failure path -----------------------------------------------------------

def test_raising_tool_records_failed_row(pipeline, db_session) -> None:
    pipeline.registry = registry_with(RaisingTool())
    result = run(pipeline.invoke("test.raises", {"value": "x"}, set(), CTX))
    assert result.status == "FAILED"
    assert "boom mid-execute" in result.error

    rows = rows_for(db_session, "test.raises")
    assert len(rows) == 1
    assert rows[0].status == "FAILED"
    assert rows[0].output == {"error": "boom mid-execute"}
    assert rows[0].latency_ms is not None


# --- contract errors --------------------------------------------------------

def test_unknown_tool_raises_not_found(pipeline, db_session) -> None:
    with pytest.raises(ToolNotFoundError):
        run(pipeline.invoke("no.such_tool", {"value": "x"}, set(), CTX))
    assert count_rows(db_session) == 0


def test_invalid_input_raises_before_execute(pipeline, db_session) -> None:
    EXECUTE_CALLED.clear()
    pipeline.registry = registry_with(SpyTool())
    with pytest.raises(ToolInputValidationError):
        run(pipeline.invoke("test.spy", {"wrong_field": "x"}, set(), CTX))
    # Never executed, never audited: malformed input is a contract error.
    assert EXECUTE_CALLED == []
    assert count_rows(db_session) == 0


def test_invalid_input_error_carries_tool_and_errors(pipeline) -> None:
    with pytest.raises(ToolInputValidationError) as exc_info:
        run(pipeline.invoke("example.echo", {}, set(), CTX))
    assert exc_info.value.tool_name == "example.echo"
    assert exc_info.value.errors  # pydantic error list


# --- exactly one row per invocation -----------------------------------------

def test_every_invocation_writes_exactly_one_row(pipeline, db_session) -> None:
    pipeline.registry = registry_with(RaisingTool())
    outcomes = [
        run(pipeline.invoke("example.echo", {"message": "a"}, set(), CTX)),
        run(pipeline.invoke("example.echo", {"message": "b"}, set(), CTX)),
        run(pipeline.invoke("example.read_file", {"path": "x"}, set(), CTX)),
        run(pipeline.invoke("example.read_file", {"path": "x"}, {"repo.read"}, CTX)),
        run(pipeline.invoke("example.denied", {"command": "x"}, set(), CTX)),
        run(pipeline.invoke("test.raises", {"value": "x"}, set(), CTX)),
    ]
    assert [r.status for r in outcomes] == [
        "EXECUTED", "EXECUTED", "DENIED", "EXECUTED", "DENIED", "FAILED",
    ]
    assert count_rows(db_session) == len(outcomes)
    # Echo succeeded twice -> two rows, both EXECUTED.
    assert len(rows_for(db_session, "example.echo")) == 2


# --- redaction --------------------------------------------------------------

def test_sensitive_input_and_output_redacted_in_row(pipeline, db_session) -> None:
    pipeline.registry = registry_with(TokenTool())
    result = run(
        pipeline.invoke(
            "test.token_echo",
            {"message": "hello", "token": "super-secret-value"},
            set(),
            CTX,
        )
    )
    assert result.status == "EXECUTED"
    # The caller still gets the real output — only the audit row is redacted.
    assert result.output == {"message": "hello", "token": "super-secret-value"}

    row = rows_for(db_session, "test.token_echo")[0]
    assert row.input["token"] == REDACTED
    assert row.output["token"] == REDACTED
    assert row.input["message"] == "hello"


def test_redact_sensitive_nested() -> None:
    data = {
        "headers": {"Authorization": "Bearer xyz", "x-ok": "1"},
        "body": {"api_key": "k", "safe": "s"},
        "list": [{"token": "t"}, "plain"],
    }
    out = redact_sensitive(data)
    assert out["headers"]["Authorization"] == REDACTED
    assert out["headers"]["x-ok"] == "1"
    assert out["body"]["api_key"] == REDACTED
    assert out["body"]["safe"] == "s"
    assert out["list"][0]["token"] == REDACTED
    assert out["list"][1] == "plain"


# --- pipeline metadata ------------------------------------------------------

def test_row_carries_task_and_step_context(pipeline, db_session) -> None:
    task_id = uuid.uuid4()
    step_id = uuid.uuid4()
    ctx = make_execution_context(task_id=task_id, step_id=step_id, agent_type="test")
    run(pipeline.invoke("example.echo", {"message": "x"}, set(), ctx))
    row = rows_for(db_session, "example.echo")[0]
    assert row.task_id == task_id
    assert row.step_id == step_id
    assert row.agent_type == "test"


def test_row_without_task_context_still_audits(pipeline, db_session) -> None:
    run(pipeline.invoke("example.echo", {"message": "x"}, set(), make_execution_context()))
    row = rows_for(db_session, "example.echo")[0]
    assert row.task_id is None
    assert row.agent_type is None
