"""Developer Agent integration tests (Phase 7).

The write-capable counterpart of Phase 6's research tests: bounded tool-use
loop against a real fixture repo with a mocked LLM. The security headlines
are the write-path traversal defense (see test_filesystem_write.py), the
capability boundary (no shell.*, no github.*), and the one-commit contract —
zero commits is a hard failure, never a degraded-but-usable summary.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select

from app.agents.developer.agent import DeveloperAgent, DeveloperError
from app.agents.developer.prompt import SYSTEM_PROMPT, observation_message
from app.agents.developer.schema import ImplementationSummary
from app.git.runner import run_git
from app.llm import StubLLMProvider
from app.llm.mock import (
    COMMIT_PROPOSAL,
    FINAL_PROPOSAL,
    IMPLEMENTATION_SUMMARY_RESPONSE,
    WRITE_PROPOSAL,
)
from app.models import (
    AuditLog,
    ImplementationSummary as SummaryRow,
    Plan as PlanRow,
    PlanStep,
    ResearchArtifact as ArtifactRow,
    Task,
    TaskStatus,
    ToolCall,
)
from app.runtime.task_lifecycle import advance_task_with_agents, transition_task
from app.tools.base import ExecutionContext, Tool
from app.tools.registry import ToolRegistry

import app.tools as tools_module


def run(coro):
    return asyncio.run(coro)


# --- fixture helpers ---------------------------------------------------------

def make_implement_step(db_session, task, *, description="Implement the fix") -> PlanStep:
    plan = PlanRow(task_id=task.id, status="ACTIVE")
    db_session.add(plan)
    db_session.flush()
    step = PlanStep(
        plan_id=plan.id,
        step_type="implement",
        sequence=1,
        depends_on=None,
        params={"description": description},
    )
    db_session.add(step)
    db_session.commit()
    db_session.refresh(step)
    return step


def make_research_artifact(db_session, task, *, relevant_files=None) -> ArtifactRow:
    row = ArtifactRow(
        task_id=task.id,
        root_cause_hypothesis="The bug is in src/app.py (stub hypothesis).",
        relevant_files=relevant_files or ["src/app.py"],
        relevant_tests=[],
        evidence=["Searched the worktree for 'VALUE'."],
        confidence=0.7,
    )
    db_session.add(row)
    db_session.commit()
    db_session.refresh(row)
    return row


def make_implementing_task(db_session, repo_task):
    """Task at IMPLEMENTING with an active plan (implement step) + artifact."""
    repo, task = repo_task
    for target in (TaskStatus.PLANNING, TaskStatus.RESEARCHING, TaskStatus.IMPLEMENTING):
        transition_task(db_session, task, target)
    db_session.commit()
    db_session.refresh(task)
    step = make_implement_step(db_session, task)
    artifact = make_research_artifact(db_session, task)
    return task, step, artifact


def ctx_for(db_session, task, agent_type="developer") -> ExecutionContext:
    return ExecutionContext(task_id=task.id, agent_type=agent_type, db=db_session)


def tool_calls_for(db_session, task_id) -> list[ToolCall]:
    return list(
        db_session.scalars(
            select(ToolCall).where(ToolCall.task_id == task_id).order_by(ToolCall.created_at)
        )
    )


def summaries_for(db_session, task_id) -> list[SummaryRow]:
    return list(
        db_session.scalars(
            select(SummaryRow).where(SummaryRow.task_id == task_id).order_by(SummaryRow.created_at)
        )
    )


def audits_for(db_session, task_id, action: str) -> list[AuditLog]:
    return list(
        db_session.scalars(
            select(AuditLog).where(
                AuditLog.task_id == task_id, AuditLog.action == action
            )
        )
    )


def worktree_path_for(db_session, task_id) -> str:
    from app.models import Worktree

    wt = db_session.scalar(
        select(Worktree).where(Worktree.task_id == task_id, Worktree.status == "active")
    )
    assert wt is not None
    return wt.path


def commit_count(db_session, task_id) -> int:
    from app.git.operations import GitOperations

    ops = GitOperations(worktree_path_for(db_session, task_id))
    return len(ops.log(limit=50))


# --- the happy-path loop -----------------------------------------------------

def test_full_loop_read_write_commit_produces_grounded_summary(
    db_session, repo_task, source_repo
) -> None:
    """Default script: write src/app.py -> commit once -> final -> grounded,
    persisted summary backed by a real commit. Main is never touched."""
    repo, task = repo_task
    step = make_implement_step(db_session, task)
    artifact = make_research_artifact(db_session, task)
    provider = StubLLMProvider(
        by_schema={
            "ToolCallProposal": [WRITE_PROPOSAL, COMMIT_PROPOSAL, FINAL_PROPOSAL],
            "ImplementationSummaryDraft": [IMPLEMENTATION_SUMMARY_RESPONSE],
        }
    )
    agent = DeveloperAgent(provider, max_tool_calls=10)

    main_head_before = run_git(source_repo, "rev-parse", "HEAD").stdout.strip()

    summary = run(agent.run(task, step, artifact, ctx_for(db_session, task)))

    assert isinstance(summary, ImplementationSummary)
    assert summary.files_changed == ["src/app.py"]
    assert summary.commit_sha  # runtime-injected, real sha
    assert len(summary.commit_sha) == 40

    # Persisted exactly once, COMPLETE, with the real commit sha.
    rows = summaries_for(db_session, task.id)
    assert len(rows) == 1
    assert rows[0].status == "COMPLETE"
    assert rows[0].commit_sha == summary.commit_sha
    assert rows[0].files_changed == ["src/app.py"]
    assert rows[0].step_id == step.id

    # The worktree really changed and really committed (branch agent/task-...).
    wt_path = worktree_path_for(db_session, task.id)
    assert (Path(wt_path) / "src" / "app.py").read_text() == "VALUE = 2\n"
    from app.git.operations import GitOperations

    ops = GitOperations(wt_path)
    commits = ops.log(limit=10)
    assert [c.summary for c in commits[:1]] == ["fix: bump VALUE to 2"]
    status = ops.status()
    assert status.branch == f"agent/task-{task.id}"
    assert status.clean is True

    # The tool loop ran write_file then commit, both EXECUTED.
    calls = tool_calls_for(db_session, task.id)
    assert [c.tool_name for c in calls] == ["filesystem.write_file", "git.commit"]
    assert all(c.status == "EXECUTED" for c in calls)
    assert calls[0].agent_type == "developer"

    # Never touches main: the source repo's default branch is untouched.
    assert run_git(source_repo, "rev-parse", "HEAD").stdout.strip() == main_head_before


def test_developer_reads_research_flagged_file_before_writing(
    db_session, repo_task
) -> None:
    repo, task = repo_task
    step = make_implement_step(db_session, task)
    artifact = make_research_artifact(db_session, task)

    read_proposal = json.dumps(
        {"tool_call": {"tool": "repository.read_file", "input": {"path": "src/app.py"}}}
    )
    provider = StubLLMProvider(
        by_schema={
            "ToolCallProposal": [read_proposal, WRITE_PROPOSAL, COMMIT_PROPOSAL, FINAL_PROPOSAL],
            "ImplementationSummaryDraft": [IMPLEMENTATION_SUMMARY_RESPONSE],
        }
    )
    agent = DeveloperAgent(provider, max_tool_calls=10)

    summary = run(agent.run(task, step, artifact, ctx_for(db_session, task)))

    calls = tool_calls_for(db_session, task.id)
    assert [c.tool_name for c in calls] == [
        "repository.read_file",
        "filesystem.write_file",
        "git.commit",
    ]
    assert summary.files_changed == ["src/app.py"]
    assert len(summaries_for(db_session, task.id)) == 1


# --- capability boundary (adversarial) ----------------------------------------

from pydantic import BaseModel, Field  # noqa: E402


class _PrInput(BaseModel):
    title: str = Field(min_length=1)


class _PrOutput(BaseModel):
    url: str


class _FakePrTool(Tool):
    """A registered github.* tool that must NEVER be invocable by Developer."""

    name = "github.create_pr"
    description = "Open a pull request."
    input_schema = _PrInput
    output_schema = _PrOutput
    capabilities: list[str] = ["github.write"]
    risk = "HIGH"

    async def execute(self, input, ctx):  # pragma: no cover — never reached
        raise AssertionError("github.create_pr must never execute")


def _registry_with_pr(monkeypatch):
    """The runtime registry plus a registered github.create_pr tool."""
    original = tools_module.build_runtime_registry

    def patched() -> ToolRegistry:
        registry = original()
        registry.register(_FakePrTool())
        return registry

    monkeypatch.setattr(tools_module, "build_runtime_registry", patched)


def test_shell_proposal_unknown_tool_audited_as_unexpected(
    db_session, repo_task
) -> None:
    """shell.run_test is outside Developer's capability set (Phase 8 gave it
    to the Test Agent, not Developer) — for the DEVELOPER it signals the LLM
    is reaching outside its job and is audited as developer.unexpected_denial
    (distinct from Research's benign case)."""
    repo, task = repo_task
    step = make_implement_step(db_session, task)
    artifact = make_research_artifact(db_session, task)

    shell_proposal = json.dumps({"tool_call": {"tool": "shell.run_test", "input": {}}})
    provider = StubLLMProvider(
        by_schema={
            "ToolCallProposal": [shell_proposal, WRITE_PROPOSAL, COMMIT_PROPOSAL, FINAL_PROPOSAL],
            "ImplementationSummaryDraft": [IMPLEMENTATION_SUMMARY_RESPONSE],
        }
    )
    agent = DeveloperAgent(provider, max_tool_calls=10)

    summary = run(agent.run(task, step, artifact, ctx_for(db_session, task)))

    assert isinstance(summary, ImplementationSummary)
    audits = audits_for(db_session, task.id, "developer.unexpected_denial")
    assert len(audits) == 1
    assert audits[0].details["tool"] == "shell.run_test"
    # shell.run_test is now a registered tool (Phase 8), so the proposal is
    # denied at the capability gate rather than surfacing as an unknown tool.
    assert audits[0].details["surfaced_as"] == "denied"
    # The loop survived and still produced a real commit.
    assert summary.files_changed == ["src/app.py"]


def test_capability_boundary_denied_and_audited_as_unexpected(
    db_session, repo_task, monkeypatch
) -> None:
    """A REGISTERED tool outside Developer's capability set is DENIED by the
    pipeline's capability gate and audited as developer.unexpected_denial —
    the adversarial capability-boundary test (Phase 6 pattern, write-flavored)."""
    _registry_with_pr(monkeypatch)
    repo, task = repo_task
    step = make_implement_step(db_session, task)
    artifact = make_research_artifact(db_session, task)

    pr_proposal = json.dumps(
        {"tool_call": {"tool": "github.create_pr", "input": {"title": "ship it"}}}
    )
    provider = StubLLMProvider(
        by_schema={
            "ToolCallProposal": [pr_proposal, WRITE_PROPOSAL, COMMIT_PROPOSAL, FINAL_PROPOSAL],
            "ImplementationSummaryDraft": [IMPLEMENTATION_SUMMARY_RESPONSE],
        }
    )
    agent = DeveloperAgent(provider, max_tool_calls=10)

    summary = run(agent.run(task, step, artifact, ctx_for(db_session, task)))

    calls = tool_calls_for(db_session, task.id)
    assert [c.tool_name for c in calls] == [
        "github.create_pr",
        "filesystem.write_file",
        "git.commit",
    ]
    assert calls[0].status == "DENIED"
    assert "github.write" in (calls[0].denial_reason or "")
    audits = audits_for(db_session, task.id, "developer.unexpected_denial")
    assert len(audits) == 1
    assert audits[0].details["tool"] == "github.create_pr"
    assert audits[0].details["surfaced_as"] == "denied"
    assert isinstance(summary, ImplementationSummary)


# --- the one-commit contract --------------------------------------------------

def test_zero_commit_is_hard_failure_with_incomplete_marker(
    db_session, repo_task
) -> None:
    """LLM writes but never commits, then says final: hard failure — no
    summary, an explicit INCOMPLETE marker, and a loud audit."""
    repo, task = repo_task
    step = make_implement_step(db_session, task)
    artifact = make_research_artifact(db_session, task)

    provider = StubLLMProvider(
        by_schema={
            "ToolCallProposal": [WRITE_PROPOSAL, FINAL_PROPOSAL],
            "ImplementationSummaryDraft": [IMPLEMENTATION_SUMMARY_RESPONSE],
        }
    )
    agent = DeveloperAgent(provider, max_tool_calls=10)

    with pytest.raises(DeveloperError):
        run(agent.run(task, step, artifact, ctx_for(db_session, task)))

    rows = summaries_for(db_session, task.id)
    assert len(rows) == 1
    assert rows[0].status == "INCOMPLETE"
    assert rows[0].commit_sha is None
    assert audits_for(db_session, task.id, "developer.no_commit")
    # Nothing was committed on the worktree.
    assert commit_count(db_session, task.id) == 1  # only the initial commit


def test_budget_exhaustion_without_commit_is_hard_failure(
    db_session, repo_task
) -> None:
    """The budget hard-stops the loop with no commit -> hard failure, NOT
    forced synthesis (research's outcome). The exhaustion is audited."""
    repo, task = repo_task
    step = make_implement_step(db_session, task)
    artifact = make_research_artifact(db_session, task)

    provider = StubLLMProvider(
        by_schema={
            "ToolCallProposal": [WRITE_PROPOSAL],  # never commits, never final
            "ImplementationSummaryDraft": [IMPLEMENTATION_SUMMARY_RESPONSE],
        }
    )
    agent = DeveloperAgent(provider, max_tool_calls=3)

    with pytest.raises(DeveloperError):
        run(agent.run(task, step, artifact, ctx_for(db_session, task)))

    calls = tool_calls_for(db_session, task.id)
    assert len(calls) == 3  # bounded
    assert all(c.tool_name == "filesystem.write_file" for c in calls)
    assert audits_for(db_session, task.id, "developer.budget_exhausted")
    rows = summaries_for(db_session, task.id)
    assert len(rows) == 1
    assert rows[0].status == "INCOMPLETE"


def test_empty_diff_after_write_handled_gracefully(db_session, repo_task) -> None:
    """Writing content identical to the existing file makes git.commit refuse
    (Phase 4) — the loop must treat that as 'nothing to do', not crash or
    loop forever, and a subsequent real change commits cleanly."""
    repo, task = repo_task
    step = make_implement_step(db_session, task)
    artifact = make_research_artifact(db_session, task)

    identical_write = json.dumps(
        {
            "tool_call": {
                "tool": "filesystem.write_file",
                "input": {"path": "src/app.py", "content": "VALUE = 1\n"},
            }
        }
    )
    real_write = json.dumps(
        {
            "tool_call": {
                "tool": "filesystem.write_file",
                "input": {"path": "src/app.py", "content": "VALUE = 99\n"},
            }
        }
    )
    provider = StubLLMProvider(
        by_schema={
            "ToolCallProposal": [
                identical_write,
                COMMIT_PROPOSAL,  # refused: empty diff
                real_write,
                COMMIT_PROPOSAL,  # succeeds
                FINAL_PROPOSAL,
            ],
            "ImplementationSummaryDraft": [IMPLEMENTATION_SUMMARY_RESPONSE],
        }
    )
    agent = DeveloperAgent(provider, max_tool_calls=10)

    summary = run(agent.run(task, step, artifact, ctx_for(db_session, task)))

    calls = tool_calls_for(db_session, task.id)
    assert [c.tool_name for c in calls] == [
        "filesystem.write_file",
        "git.commit",
        "filesystem.write_file",
        "git.commit",
    ]
    # The empty-diff commit is a FAILED call — handled, not a crash.
    assert calls[1].status == "FAILED"
    assert "nothing to commit" in calls[1].output["error"]
    assert calls[3].status == "EXECUTED"
    assert len(summaries_for(db_session, task.id)) == 1
    assert summary.files_changed == ["src/app.py"]
    assert commit_count(db_session, task.id) == 2  # initial + the ONE real commit


def test_post_commit_writes_and_commits_denied_structurally(
    db_session, repo_task
) -> None:
    """The one-commit contract is enforced structurally: after the first
    commit, further write_file/commit proposals are denied at the agent
    level (audited), so the commit provably represents the full change."""
    repo, task = repo_task
    step = make_implement_step(db_session, task)
    artifact = make_research_artifact(db_session, task)

    extra_write = json.dumps(
        {
            "tool_call": {
                "tool": "filesystem.write_file",
                "input": {"path": "src/app.py", "content": "VALUE = 3\n"},
            }
        }
    )
    provider = StubLLMProvider(
        by_schema={
            "ToolCallProposal": [WRITE_PROPOSAL, COMMIT_PROPOSAL, extra_write, FINAL_PROPOSAL],
            "ImplementationSummaryDraft": [IMPLEMENTATION_SUMMARY_RESPONSE],
        }
    )
    agent = DeveloperAgent(provider, max_tool_calls=10)

    summary = run(agent.run(task, step, artifact, ctx_for(db_session, task)))

    # The post-commit write was denied by the agent-level guard, audited.
    assert audits_for(db_session, task.id, "developer.post_commit_proposal")
    # Still exactly one commit; the denied write never reached the disk.
    assert commit_count(db_session, task.id) == 2
    assert Path(worktree_path_for(db_session, task.id), "src/app.py").read_text() == "VALUE = 2\n"
    assert summary.commit_sha


def test_two_developer_runs_build_on_prior_commit(db_session, repo_task) -> None:
    """A second developer run on the same worktree (future replan path) must
    build on the prior commit, not discard it — two commits, two summaries."""
    repo, task = repo_task
    step = make_implement_step(db_session, task)
    artifact = make_research_artifact(db_session, task)

    provider1 = StubLLMProvider(
        by_schema={
            "ToolCallProposal": [WRITE_PROPOSAL, COMMIT_PROPOSAL, FINAL_PROPOSAL],
            "ImplementationSummaryDraft": [IMPLEMENTATION_SUMMARY_RESPONSE],
        }
    )
    first = DeveloperAgent(provider1, max_tool_calls=10)
    summary1 = run(first.run(task, step, artifact, ctx_for(db_session, task)))

    second_write = json.dumps(
        {
            "tool_call": {
                "tool": "filesystem.write_file",
                "input": {"path": "src/app.py", "content": "VALUE = 3\n"},
            }
        }
    )
    second_commit = json.dumps(
        {"tool_call": {"tool": "git.commit", "input": {"message": "fix: bump VALUE to 3"}}}
    )
    provider2 = StubLLMProvider(
        by_schema={
            "ToolCallProposal": [second_write, second_commit, FINAL_PROPOSAL],
            "ImplementationSummaryDraft": [IMPLEMENTATION_SUMMARY_RESPONSE],
        }
    )
    second = DeveloperAgent(provider2, max_tool_calls=10)
    summary2 = run(second.run(task, step, artifact, ctx_for(db_session, task)))

    assert summary1.commit_sha != summary2.commit_sha
    # Both commits are on the same branch, second on top of the first.
    from app.git.operations import GitOperations

    ops = GitOperations(worktree_path_for(db_session, task.id))
    commits = ops.log(limit=10)
    assert len(commits) == 3  # initial + run1 + run2
    assert commits[0].sha == summary2.commit_sha
    assert commits[1].sha == summary1.commit_sha
    rows = summaries_for(db_session, task.id)
    assert len(rows) == 2
    assert all(r.status == "COMPLETE" for r in rows)


# --- grounding: files-changed cross-check ------------------------------------

def test_files_changed_fabrication_corrected_once_then_grounded(
    db_session, repo_task
) -> None:
    """Claiming a file that was never written is rejected, the LLM is told
    exactly why, and the corrected (grounded) summary is persisted."""
    repo, task = repo_task
    step = make_implement_step(db_session, task)
    artifact = make_research_artifact(db_session, task)

    fabricated = json.dumps(
        {
            "files_changed": ["src/app.py", "src/never_written.py"],
            "summary": "changed things",
            "tests_added": [],
            "deviations_from_research": None,
        }
    )
    provider = StubLLMProvider(
        by_schema={
            "ToolCallProposal": [WRITE_PROPOSAL, COMMIT_PROPOSAL, FINAL_PROPOSAL],
            "ImplementationSummaryDraft": [fabricated, IMPLEMENTATION_SUMMARY_RESPONSE],
        }
    )
    agent = DeveloperAgent(provider, max_tool_calls=10)

    summary = run(agent.run(task, step, artifact, ctx_for(db_session, task)))

    # The fabricated path never survives: the final summary is grounded.
    assert summary.files_changed == ["src/app.py"]
    rows = summaries_for(db_session, task.id)
    assert len(rows) == 1
    assert rows[0].files_changed == ["src/app.py"]
    # No warning was needed — the correction retry succeeded.
    assert not audits_for(db_session, task.id, "implementation.files_unverified")


def test_files_changed_persistently_fabricated_accepted_with_warning(
    db_session, repo_task
) -> None:
    """Worst case: the LLM keeps fabricating after the correction retry. The
    accept-with-warning policy (Phase 6, applied to writes) persists the
    summary and audits the discrepancy loudly — the committed diff remains
    verifiable ground truth downstream."""
    repo, task = repo_task
    step = make_implement_step(db_session, task)
    artifact = make_research_artifact(db_session, task)

    fabricated = json.dumps(
        {
            "files_changed": ["src/app.py", "src/never_written.py"],
            "summary": "changed things",
            "tests_added": [],
            "deviations_from_research": None,
        }
    )
    provider = StubLLMProvider(
        by_schema={
            "ToolCallProposal": [WRITE_PROPOSAL, COMMIT_PROPOSAL, FINAL_PROPOSAL],
            "ImplementationSummaryDraft": [fabricated],  # repeats: both attempts lie
        }
    )
    agent = DeveloperAgent(provider, max_tool_calls=10)

    summary = run(agent.run(task, step, artifact, ctx_for(db_session, task)))

    # Accepted-with-warning: the fabricated claim is persisted but audited.
    assert summary.files_changed == ["src/app.py", "src/never_written.py"]
    audits = audits_for(db_session, task.id, "implementation.files_unverified")
    assert len(audits) == 1
    assert audits[0].details["mismatch"] == ["src/never_written.py"]
    rows = summaries_for(db_session, task.id)
    assert len(rows) == 1
    assert rows[0].status == "COMPLETE"


def test_deviations_from_research_prompted_and_persisted(db_session, repo_task) -> None:
    """Changing files research never flagged must be EXPLAINED, not noted
    silently — the retry prompts for it, the corrected summary carries it."""
    repo, task = repo_task
    step = make_implement_step(db_session, task)
    artifact = make_research_artifact(db_session, task, relevant_files=["src/app.py"])

    write_test = json.dumps(
        {
            "tool_call": {
                "tool": "filesystem.write_file",
                "input": {
                    "path": "tests/test_app.py",
                    "content": "def test_v():\n    assert VALUE == 2\n",
                },
            }
        }
    )
    commit = json.dumps(
        {"tool_call": {"tool": "git.commit", "input": {"message": "fix: add test"}}}
    )
    unexplained = json.dumps(
        {
            "files_changed": ["tests/test_app.py"],
            "summary": "added a regression test",
            "tests_added": ["tests/test_app.py"],
            "deviations_from_research": None,
        }
    )
    explained = json.dumps(
        {
            "files_changed": ["tests/test_app.py"],
            "summary": "added a regression test",
            "tests_added": ["tests/test_app.py"],
            "deviations_from_research": "The research focused on src/app.py, but the fix "
            "needed a regression test, so I also touched tests/test_app.py.",
        }
    )
    provider = StubLLMProvider(
        by_schema={
            "ToolCallProposal": [write_test, commit, FINAL_PROPOSAL],
            "ImplementationSummaryDraft": [unexplained, explained],
        }
    )
    agent = DeveloperAgent(provider, max_tool_calls=10)

    summary = run(agent.run(task, step, artifact, ctx_for(db_session, task)))

    assert summary.files_changed == ["tests/test_app.py"]
    assert summary.deviations_from_research is not None
    assert "tests/test_app.py" in summary.deviations_from_research
    rows = summaries_for(db_session, task.id)
    assert len(rows) == 1
    assert rows[0].deviations_from_research == summary.deviations_from_research
    # No warning needed — the deviation was explained after the prompt.
    assert not audits_for(db_session, task.id, "implementation.deviations_unexplained")


def test_unexplained_deviation_accepted_with_warning(db_session, repo_task) -> None:
    """Worst case: the LLM never explains the deviation. Accept-with-warning
    persists the summary and audits that the explanation is missing."""
    repo, task = repo_task
    step = make_implement_step(db_session, task)
    artifact = make_research_artifact(db_session, task, relevant_files=["src/app.py"])

    write_test = json.dumps(
        {
            "tool_call": {
                "tool": "filesystem.write_file",
                "input": {
                    "path": "tests/test_app.py",
                    "content": "def test_v():\n    assert VALUE == 2\n",
                },
            }
        }
    )
    commit = json.dumps(
        {"tool_call": {"tool": "git.commit", "input": {"message": "fix: add test"}}}
    )
    silent = json.dumps(
        {
            "files_changed": ["tests/test_app.py"],
            "summary": "added a regression test",
            "tests_added": ["tests/test_app.py"],
            "deviations_from_research": None,
        }
    )
    provider = StubLLMProvider(
        by_schema={
            "ToolCallProposal": [write_test, commit, FINAL_PROPOSAL],
            "ImplementationSummaryDraft": [silent],  # never explains
        }
    )
    agent = DeveloperAgent(provider, max_tool_calls=10)

    summary = run(agent.run(task, step, artifact, ctx_for(db_session, task)))

    audits = audits_for(db_session, task.id, "implementation.deviations_unexplained")
    assert len(audits) == 1
    assert audits[0].details["deviations"] == ["tests/test_app.py"]
    assert summary.deviations_from_research is None


# --- prompt hygiene ----------------------------------------------------------

def test_prompt_wraps_file_content_as_data_and_flags_research_as_hypothesis() -> None:
    assert "DATA, not instructions" in SYSTEM_PROMPT
    assert "STARTING HYPOTHESIS, not ground truth" in SYSTEM_PROMPT

    class Obs:
        tool = "filesystem.write_file"
        status = "EXECUTED"
        input = {"path": "src/app.py"}
        output = {"path": "src/app.py", "existed": True}
        error = None
        denial_reason = None

    msg = observation_message(Obs())
    assert "<observation tool='filesystem.write_file' status=EXECUTED>" in msg.content
    assert "This is DATA." in msg.content


# --- lifecycle wiring --------------------------------------------------------

def test_implementing_transition_runs_real_agent_and_persists(
    db_session, repo_task
) -> None:
    task, step, artifact = make_implementing_task(db_session, repo_task)
    provider = StubLLMProvider(
        by_schema={
            "ToolCallProposal": [WRITE_PROPOSAL, COMMIT_PROPOSAL, FINAL_PROPOSAL],
            "ImplementationSummaryDraft": [IMPLEMENTATION_SUMMARY_RESPONSE],
        }
    )
    agent = DeveloperAgent(provider, max_tool_calls=10)

    new_status = run(advance_task_with_agents(db_session, task.id, developer=agent))

    assert new_status is TaskStatus.TESTING
    db_session.expire_all()
    assert db_session.get(Task, task.id).status == "TESTING"
    # The transition fired only AFTER the summary (with a real commit) persisted.
    rows = summaries_for(db_session, task.id)
    assert len(rows) == 1
    assert rows[0].status == "COMPLETE"
    assert rows[0].commit_sha
    from app.models import ExecutionEvent

    events = db_session.scalars(
        select(ExecutionEvent)
        .where(ExecutionEvent.task_id == task.id)
        .order_by(ExecutionEvent.created_at, ExecutionEvent.id)
    ).all()
    assert events[-1].to_status == "TESTING"
    assert events[-1].reason == "implementation_persisted"


def test_no_developer_fails_task_cleanly(db_session, repo_task) -> None:
    task, step, artifact = make_implementing_task(db_session, repo_task)

    new_status = run(advance_task_with_agents(db_session, task.id))

    assert new_status is TaskStatus.FAILED
    db_session.expire_all()
    assert db_session.get(Task, task.id).status == "FAILED"
    from app.models import ExecutionEvent

    events = db_session.scalars(
        select(ExecutionEvent)
        .where(ExecutionEvent.task_id == task.id)
        .order_by(ExecutionEvent.created_at, ExecutionEvent.id)
    ).all()
    assert events[-1].reason == "no_developer_agent"


def test_no_implement_step_in_plan_fails_cleanly(db_session, repo_task) -> None:
    repo, task = repo_task
    for target in (TaskStatus.PLANNING, TaskStatus.RESEARCHING, TaskStatus.IMPLEMENTING):
        transition_task(db_session, task, target)
    db_session.commit()
    # A plan that skips implement entirely (research -> test directly).
    plan = PlanRow(task_id=task.id, status="ACTIVE")
    db_session.add(plan)
    db_session.flush()
    db_session.add(
        PlanStep(
            plan_id=plan.id, step_type="test", sequence=1,
            depends_on=None, params={},
        )
    )
    db_session.commit()

    agent = DeveloperAgent(StubLLMProvider(), max_tool_calls=10)
    new_status = run(advance_task_with_agents(db_session, task.id, developer=agent))

    assert new_status is TaskStatus.FAILED
    db_session.expire_all()
    from app.models import ExecutionEvent

    events = db_session.scalars(
        select(ExecutionEvent)
        .where(ExecutionEvent.task_id == task.id)
        .order_by(ExecutionEvent.created_at, ExecutionEvent.id)
    ).all()
    assert events[-1].reason == "no_implement_step"


def test_no_research_artifact_fails_cleanly(db_session, repo_task) -> None:
    task, step, _ = make_implementing_task(db_session, repo_task)
    # Remove the artifact: developer has no hypothesis to build on.
    from app.models import ResearchArtifact

    for row in db_session.scalars(
        select(ResearchArtifact).where(ResearchArtifact.task_id == task.id)
    ):
        db_session.delete(row)
    db_session.commit()

    agent = DeveloperAgent(StubLLMProvider(), max_tool_calls=10)
    new_status = run(advance_task_with_agents(db_session, task.id, developer=agent))

    assert new_status is TaskStatus.FAILED
    db_session.expire_all()
    from app.models import ExecutionEvent

    events = db_session.scalars(
        select(ExecutionEvent)
        .where(ExecutionEvent.task_id == task.id)
        .order_by(ExecutionEvent.created_at, ExecutionEvent.id)
    ).all()
    assert events[-1].reason == "no_research_artifact"


def test_developer_hard_failure_fails_task_not_escalated(db_session, repo_task) -> None:
    """A DeveloperError (e.g. no commit) fails the task — never ESCALATED
    (reserved for replan-budget exhaustion) — and the INCOMPLETE marker is
    persisted before the FAILED transition."""
    task, step, artifact = make_implementing_task(db_session, repo_task)
    # Never commits: responds final immediately.
    provider = StubLLMProvider(
        by_schema={
            "ToolCallProposal": [FINAL_PROPOSAL],
            "ImplementationSummaryDraft": [IMPLEMENTATION_SUMMARY_RESPONSE],
        }
    )
    agent = DeveloperAgent(provider, max_tool_calls=10)

    new_status = run(advance_task_with_agents(db_session, task.id, developer=agent))

    assert new_status is TaskStatus.FAILED
    db_session.expire_all()
    assert db_session.get(Task, task.id).status == "FAILED"
    rows = summaries_for(db_session, task.id)
    assert len(rows) == 1
    assert rows[0].status == "INCOMPLETE"
    from app.models import ExecutionEvent

    events = db_session.scalars(
        select(ExecutionEvent)
        .where(ExecutionEvent.task_id == task.id)
        .order_by(ExecutionEvent.created_at, ExecutionEvent.id)
    ).all()
    assert events[-1].reason == "developer_failed"
