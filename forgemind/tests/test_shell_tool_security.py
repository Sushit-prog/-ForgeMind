"""Adversarial tests for ``shell.run_test`` (Phase 8 security).

The core structural guarantee: the tool takes NO command input at all. The
command comes exclusively from ``repositories.test_command`` (validated at
discovery time). ``extra=\"forbid\"`` on the input schema means even a
smuggled extra argument is rejected at validation before anything runs —
proving structurally, not just with a rejected-input test, that there is
no agent-controlled path into the subprocess.

These tests are the regression guard the milestone asks for: they mostly
prove there is nothing to inject into.
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from app.execution.tool_pipeline import ToolInputValidationError, ToolPipeline
from app.tools.base import ExecutionContext
from app.tools.shell_tools import RunTestInput


def test_input_schema_forbids_command_field() -> None:
    """A smuggled ``command`` (or any extra key) is rejected by the schema
    itself — the tool never even sees it."""
    with pytest.raises(ValidationError):
        RunTestInput.model_validate(
            {"worktree_id": str(uuid.uuid4()), "command": "rm -rf /"}
        )
    with pytest.raises(ValidationError):
        RunTestInput.model_validate(
            {
                "worktree_id": str(uuid.uuid4()),
                "args": ["--collect-only"],
                "shell": True,
            }
        )
    # Only the worktree_id is accepted — nothing else.
    parsed = RunTestInput.model_validate({"worktree_id": str(uuid.uuid4())})
    assert parsed.worktree_id is not None


def test_adversarial_proposal_rejected_at_pipeline_validation(db_session) -> None:
    """Even a hostile agent proposing ``shell.run_test`` with an injected
    command is stopped at validation — a contract error, no tool row, no
    subprocess ever spawned."""
    import asyncio

    pipeline = ToolPipeline(db_session)
    ctx = ExecutionContext(task_id=uuid.uuid4(), agent_type="debugger", db=db_session)

    with pytest.raises(ToolInputValidationError):
        # The injected command field must fail FIRST, at schema validation.
        asyncio.run(
            pipeline.invoke(
                "shell.run_test",
                {
                    "worktree_id": str(uuid.uuid4()),
                    "command": "pytest; rm -rf /",
                },
                {"shell.test"},
                ctx,
            )
        )


def test_registry_exposes_run_test_with_no_command_input(db_session) -> None:
    """The registered tool's schema is the same no-command shape the tests
    above exercised — nothing sneaks a command in at registration either."""
    from app.tools import build_runtime_registry

    tool = build_runtime_registry().get("shell.run_test")
    fields = tool.input_schema.model_fields
    assert set(fields) == {"worktree_id"}
