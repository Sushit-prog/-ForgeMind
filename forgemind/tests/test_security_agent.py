"""Security Agent tests (Phase 9, architecture doc sections 12/E).

The Security Agent runs its Section-12 checklist (injection, secrets,
unsafe subprocess/network, path traversal, auth/authz) against the commit's
diff and returns PASS or FAIL. The checklist tests plant ONE real example
of each category in the diff and prove the agent FAILs with findings that
match what was actually planted — a Security agent that never caught
anything in tests is a Security agent you have no evidence works.

Independence is structural and one step past the Reviewer: the Security
agent's context contains the diff but never the ReviewResult (its run
signature has no review parameter).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from app.agents.security.agent import SecurityAgent
from app.agents.security.schema import SecurityFinding, SecurityResult
from app.git.runner import run_git
from app.git.worktree_manager import WorktreeManager
from app.llm import StubLLMProvider
from app.llm.mock import FINAL_PROPOSAL
from app.models import AuditLog
from app.models import Repository, SecurityResult as SecurityRow
from app.models import Task
from app.tools.base import ExecutionContext


def run(coro):
    return asyncio.run(coro)


def make_commit_repo(tmp_path, *, app_content: str) -> Path:
    repo = tmp_path / "secrepo"
    repo.mkdir()
    run_git(repo, "init", "-b", "main")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("VALUE = 1\n")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-m", "initial")
    return repo


def repo_task_worktree_with_commit(db_session, repo_path: Path, app_content: str):
    from app.models import Repository

    repo = Repository(url=str(repo_path), default_branch="main")
    db_session.add(repo)
    db_session.flush()
    task = Task(objective="implement the change", repository_id=repo.id)
    db_session.add(task)
    db_session.commit()
    db_session.refresh(repo)
    db_session.refresh(task)
    worktree = WorktreeManager(db_session).create(task.id, repo)
    wt_path = Path(worktree.path)
    (wt_path / "src" / "app.py").write_text(app_content)
    run_git(wt_path, "add", "-A")
    run_git(wt_path, "commit", "-m", "dev: the change")
    sha = run_git(wt_path, "rev-parse", "HEAD").stdout.strip()
    return repo, task, worktree, sha


def security_with(by_schema: dict) -> SecurityAgent:
    return SecurityAgent(StubLLMProvider(by_schema=by_schema))


def diff_proposal(sha: str) -> str:
    return json.dumps({"tool_call": {"tool": "git.diff", "input": {"commit": sha}}})


PASS_RESPONSE = json.dumps({"decision": "PASS", "findings": []})
FAIL_RESPONSE = json.dumps(
    {
        "decision": "FAIL",
        "findings": [
            {
                "category": "INJECTION",
                "file": "src/app.py",
                "line": 1,
                "description": "Query built from unsanitized user input.",
                "severity": "high",
            },
            {
                "category": "SECRETS",
                "file": "src/app.py",
                "line": 2,
                "description": "Hardcoded API key.",
                "severity": "high",
            },
            {
                "category": "UNSAFE_SUBPROCESS",
                "file": "src/app.py",
                "line": 3,
                "description": "shell=True with command from untrusted input.",
                "severity": "high",
            },
            {
                "category": "UNSAFE_NETWORK",
                "file": "src/app.py",
                "line": 4,
                "description": "Unexpected outbound request.",
                "severity": "medium",
            },
            {
                "category": "PATH_TRAVERSAL",
                "file": "src/app.py",
                "line": 5,
                "description": "Path joined from untrusted input without containment.",
                "severity": "high",
            },
            {
                "category": "AUTH_AUTHZ",
                "file": "src/app.py",
                "line": 6,
                "description": "Admin check bypassed.",
                "severity": "high",
            },
        ],
    }
)

# One planted example per checklist category.
PLANTED = (
    "import os, subprocess, sqlite3\n"
    "q = f\"SELECT * FROM users WHERE name = '{user_input}'\"\n"
    "API_KEY = 'sk-live-1234567890abcdef'\n"
    "subprocess.run(cmd, shell=True)\n"
    "requests.get('http://169.254.169.254/latest/meta-data/')\n"
    "open(os.path.join(BASE, user_path)).read()\n"
    "if True:  # admin check removed\n"
    "    pass\n"
)


# --- schema ----------------------------------------------------------------


def test_security_result_schema_validation() -> None:
    ok = SecurityResult(decision="PASS", findings=[])
    assert ok.decision == "PASS"
    with pytest.raises(ValidationError):
        SecurityResult(decision="PASS", findings=[SecurityFinding(
            category="SECRETS", file="f", line=1,
            description="d", severity="high",
        )])  # PASS with findings is invalid
    with pytest.raises(ValidationError):
        SecurityResult(decision="FAIL", findings=[])  # FAIL needs findings
    with pytest.raises(ValidationError):
        SecurityResult(decision="FAIL", findings=[SecurityFinding(
            category="BOGUS", file="f", description="d", severity="low",
        )])  # unknown category


# --- integration: PASS on a clean diff -------------------------------------


def test_security_passes_clean_diff(db_session, tmp_path) -> None:
    repo_path = make_commit_repo(tmp_path, app_content="VALUE = 1\n")
    repo, task, worktree, sha = repo_task_worktree_with_commit(
        db_session, repo_path, "VALUE = 2\n"
    )
    agent = security_with(
        {
            "ToolCallProposal": [diff_proposal(sha), FINAL_PROPOSAL],
            "SecurityResult": [PASS_RESPONSE],
        }
    )

    result = run(
        agent.run(task, sha, ExecutionContext(
            task_id=task.id, agent_type="security", db=db_session
        ))
    )

    assert result.decision == "PASS"
    rows = db_session.scalars(
        select(SecurityRow).where(SecurityRow.task_id == task.id)
    ).all()
    assert len(rows) == 1
    assert rows[0].decision == "PASS"
    assert rows[0].commit_sha == sha


# --- integration: FAIL on planted examples of EVERY checklist category ------


def test_security_fails_on_planted_checklist_examples(db_session, tmp_path) -> None:
    """The checklist is tested against a REAL diff with one planted example
    of each category. The agent FAILs with findings, and the persisted rows
    prove it saw the diff (the planted strings are in its context)."""
    repo_path = make_commit_repo(tmp_path, app_content="VALUE = 1\n")
    repo, task, worktree, sha = repo_task_worktree_with_commit(
        db_session, repo_path, PLANTED
    )
    provider = StubLLMProvider(
        by_schema={
            "ToolCallProposal": [diff_proposal(sha), FINAL_PROPOSAL],
            "SecurityResult": [FAIL_RESPONSE],
        }
    )
    agent = SecurityAgent(provider)

    result = run(
        agent.run(task, sha, ExecutionContext(
            task_id=task.id, agent_type="security", db=db_session
        ))
    )

    assert result.decision == "FAIL"
    # Every checklist category is represented in the findings.
    categories = {f.category for f in result.findings}
    for expected in (
        "INJECTION", "SECRETS", "UNSAFE_SUBPROCESS",
        "UNSAFE_NETWORK", "PATH_TRAVERSAL", "AUTH_AUTHZ",
    ):
        assert expected in categories, f"missing finding category {expected}"

    # The diff the agent actually saw contained the planted examples.
    all_text = "\n".join(
        m.content for msgs in provider.structured_calls + provider.generate_calls
        for m in msgs
    )
    assert "sk-live-1234567890abcdef" in all_text
    assert "shell=True" in all_text
    assert "169.254.169.254" in all_text

    rows = db_session.scalars(
        select(SecurityRow).where(SecurityRow.task_id == task.id)
    ).all()
    assert rows[0].decision == "FAIL"
    assert len(rows[0].findings) == 6


# --- independence: Security never sees the ReviewResult ---------------------


def test_security_is_blind_to_review_result(db_session, tmp_path) -> None:
    """Security runs after the Reviewer approved, but is blind to what the
    Reviewer said: the ReviewResult is not in the Security agent's context at
    all (its run signature has no review parameter). The agent's prompt
    contains the diff, never the review verdict."""
    from app.models import ReviewResult as ReviewRow

    repo_path = make_commit_repo(tmp_path, app_content="VALUE = 1\n")
    repo, task, worktree, sha = repo_task_worktree_with_commit(
        db_session, repo_path, PLANTED
    )
    # A prior APPROVE exists in the DB — Security must not see it.
    db_session.add(
        ReviewRow(
            task_id=task.id,
            commit_sha=sha,
            decision="APPROVE",
            severity="low",
            issues=[],
        )
    )
    db_session.commit()

    provider = StubLLMProvider(
        by_schema={
            "ToolCallProposal": [diff_proposal(sha), FINAL_PROPOSAL],
            "SecurityResult": [FAIL_RESPONSE],
        }
    )
    agent = SecurityAgent(provider)
    result = run(
        agent.run(task, sha, ExecutionContext(
            task_id=task.id, agent_type="security", db=db_session
        ))
    )

    assert result.decision == "FAIL"  # the planted diff fails regardless
    all_text = "\n".join(
        m.content for msgs in provider.structured_calls + provider.generate_calls
        for m in msgs
    )
    # The ReviewResult's VERDICT data is absent (the word "reviewer" appears
    # only as an instruction in the system prompt — that's not the verdict).
    assert "APPROVE" not in all_text
    assert "approved" not in all_text
    # ...and the review row's decision is nowhere in the context.
    assert "review_approved" not in all_text


# --- adversarial: write proposal denied + audited ---------------------------


def test_security_write_proposal_denied_and_audited(db_session, tmp_path) -> None:
    """A write proposal by the read-only Security agent is DENIED and audited
    as security.unexpected_denial (Developer's pattern)."""
    repo_path = make_commit_repo(tmp_path, app_content="VALUE = 1\n")
    repo, task, worktree, sha = repo_task_worktree_with_commit(
        db_session, repo_path, "VALUE = 2\n"
    )
    write_proposal = json.dumps(
        {
            "tool_call": {
                "tool": "filesystem.write_file",
                "input": {"path": "src/app.py", "content": "VALUE = 3\n"},
            }
        }
    )
    agent = security_with(
        {
            "ToolCallProposal": [write_proposal, diff_proposal(sha), FINAL_PROPOSAL],
            "SecurityResult": [PASS_RESPONSE],
        }
    )

    result = run(
        agent.run(task, sha, ExecutionContext(
            task_id=task.id, agent_type="security", db=db_session
        ))
    )

    assert result.decision == "PASS"
    audits = db_session.scalars(
        select(AuditLog).where(AuditLog.task_id == task.id)
    ).all()
    denial = [a for a in audits if a.action == "security.unexpected_denial"]
    assert len(denial) == 1
    assert denial[0].details["tool"] == "filesystem.write_file"
