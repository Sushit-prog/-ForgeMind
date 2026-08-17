"""ImplementationSummary schema + grounding cross-check unit tests (Phase 7).

Shape validation is only half the contract: ``files_changed`` in a final
summary must match the files the tool-use loop ACTUALLY wrote via
``filesystem.write_file`` — the developer-flavored version of Phase 6's
relevant-files grounding check.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.developer.schema import (
    ImplementationSummary,
    ImplementationSummaryDraft,
    files_changed_mismatch,
    normalize_path,
    research_deviations,
    written_paths,
)


def draft(**overrides) -> ImplementationSummaryDraft:
    base = {
        "files_changed": ["src/app.py"],
        "summary": "Bumped VALUE per the research hypothesis.",
        "tests_added": [],
        "deviations_from_research": None,
    }
    base.update(overrides)
    return ImplementationSummaryDraft.model_validate(base)


# --- schema validation ------------------------------------------------------

def test_draft_valid() -> None:
    d = draft()
    assert d.files_changed == ["src/app.py"]
    assert d.summary


def test_draft_requires_summary() -> None:
    with pytest.raises(ValidationError):
        draft(summary="")


def test_draft_normalizes_paths() -> None:
    """Path spelling is canonicalized so the cross-check is not fooled by
    cosmetic differences (./src/./app.py vs src/app.py)."""
    d = draft(files_changed=["./src/./app.py", "tests//test_app.py"])
    assert d.files_changed == ["src/app.py", "tests/test_app.py"]


def test_summary_injects_commit_sha() -> None:
    d = draft()
    summary = ImplementationSummary(commit_sha="a" * 40, **d.model_dump())
    assert summary.commit_sha == "a" * 40
    assert summary.files_changed == ["src/app.py"]


def test_summary_requires_commit_sha() -> None:
    with pytest.raises(ValidationError):
        ImplementationSummary(**draft().model_dump())


def test_deviations_from_research_optional() -> None:
    d = draft(deviations_from_research="research missed the config file")
    assert d.deviations_from_research
    assert draft().deviations_from_research is None


# --- files-changed cross-check ----------------------------------------------

def test_normalize_path_handles_dots_and_backslashes() -> None:
    assert normalize_path("./src/app.py") == "src/app.py"
    assert normalize_path("src\\app.py") == "src/app.py"
    assert normalize_path("src/./app.py") == "src/app.py"


class Obs:
    def __init__(self, tool, status, input):
        self.tool = tool
        self.status = status
        self.input = input


def test_written_paths_only_counts_executed_writes() -> None:
    observations = [
        Obs("filesystem.write_file", "EXECUTED", {"path": "src/app.py"}),
        Obs("filesystem.write_file", "EXECUTED", {"path": "tests/test_app.py"}),
        # traversal — never written
        Obs("filesystem.write_file", "FAILED", {"path": "../../evil.py"}),
        # post-commit guard
        Obs("filesystem.write_file", "DENIED", {"path": "src/new.py"}),
        # read, not a write
        Obs("repository.read_file", "EXECUTED", {"path": "README.md"}),
        Obs("git.commit", "EXECUTED", {}),
    ]
    assert written_paths(observations) == {"src/app.py", "tests/test_app.py"}


def test_files_changed_mismatch_accepts_consistent() -> None:
    summary = ImplementationSummary(commit_sha="a" * 40, **draft().model_dump())
    assert files_changed_mismatch(summary, {"src/app.py"}) == []


def test_files_changed_mismatch_rejects_fabricated() -> None:
    """Claiming a file that was never written is a grounding violation."""
    claimed = ["src/app.py", "src/never_written.py"]
    summary = ImplementationSummary(
        commit_sha="a" * 40, **draft(files_changed=claimed).model_dump()
    )
    assert files_changed_mismatch(summary, {"src/app.py"}) == ["src/never_written.py"]


def test_files_changed_mismatch_rejects_underreporting() -> None:
    """Omitting a file that WAS written also mismatches — the summary must
    reflect the full change, not just a convenient subset."""
    summary = ImplementationSummary(commit_sha="a" * 40, **draft().model_dump())
    assert files_changed_mismatch(summary, {"src/app.py", "tests/test_app.py"}) == [
        "tests/test_app.py"
    ]


def test_research_deviations_flags_unflagged_files() -> None:
    assert research_deviations({"src/app.py"}, ["src/app.py"]) == []
    assert research_deviations(
        {"src/app.py", "config/settings.py"}, ["src/app.py"]
    ) == ["config/settings.py"]
