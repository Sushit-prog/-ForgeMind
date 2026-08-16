"""ResearchArtifact schema (Section 7) + grounding cross-check.

The cross-check is the research-equivalent of the planner's DAG validation:
an LLM may not claim a file is relevant if the tool-use loop never actually
observed it. These tests exercise the pure functions — no I/O.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.researcher.schema import (
    ResearchArtifact,
    observed_paths,
    unobserved_files,
)


def make_artifact(**overrides) -> ResearchArtifact:
    base = dict(
        root_cause_hypothesis="the bug is in src/app.py",
        relevant_files=["src/app.py"],
        relevant_tests=["tests/test_app.py"],
        evidence=["searched the worktree"],
        confidence=0.7,
    )
    base.update(overrides)
    return ResearchArtifact(**base)


def test_valid_artifact_passes() -> None:
    artifact = make_artifact()
    assert artifact.confidence == 0.7


def test_confidence_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        make_artifact(confidence=-0.1)
    with pytest.raises(ValidationError):
        make_artifact(confidence=1.1)


def test_empty_hypothesis_rejected() -> None:
    with pytest.raises(ValidationError):
        make_artifact(root_cause_hypothesis="")


def test_empty_evidence_rejected() -> None:
    with pytest.raises(ValidationError):
        make_artifact(evidence=[])


def test_confidence_clamped_to_two_decimals() -> None:
    assert make_artifact(confidence=0.6666).confidence == 0.67


def test_relevant_files_optional() -> None:
    artifact = make_artifact(relevant_files=[])
    assert artifact.relevant_files == []


# --- grounding cross-check --------------------------------------------------


def test_unobserved_files_empty_when_fully_grounded() -> None:
    artifact = make_artifact()
    observed = {"src/app.py", "tests/test_app.py"}
    assert unobserved_files(artifact, observed) == []


def test_unobserved_files_flags_fabricated_path() -> None:
    artifact = make_artifact(
        relevant_files=["src/app.py", "src/never_seen.py"], relevant_tests=[]
    )
    observed = {"src/app.py"}
    assert unobserved_files(artifact, observed) == ["src/never_seen.py"]


def test_unobserved_files_checks_tests_too() -> None:
    artifact = make_artifact(relevant_tests=["tests/fake_test.py"])
    observed = {"src/app.py"}
    assert unobserved_files(artifact, observed) == ["tests/fake_test.py"]


def test_observed_paths_from_executed_tools() -> None:
    class Obs:
        def __init__(self, tool, status, input=None, output=None):
            self.tool = tool
            self.status = status
            self.input = input or {}
            self.output = output

    observations = [
        Obs("repository.read_file", "EXECUTED", {"path": "src/app.py"}, {}),
        Obs(
            "repository.search",
            "EXECUTED",
            {"query": "VALUE"},
            {"matches": [{"path": "src/app.py"}, {"path": "tests/test_app.py"}]},
        ),
        Obs("repository.list_files", "EXECUTED", {}, {"files": ["README.md"]}),
        # DENIED and FAILED calls contribute nothing — the agent never saw
        # a result, so it cannot claim the path as ground truth.
        Obs("repository.read_file", "DENIED", {"path": "secrets.env"}, {}),
        Obs("git.commit", "DENIED", {}, None),
    ]
    assert observed_paths(observations) == {
        "src/app.py",
        "tests/test_app.py",
        "README.md",
    }


def test_observed_paths_ignores_git_tools_for_grounding() -> None:
    """git.* output is evidence text, not a file claim (deliberate)."""

    class Obs:
        def __init__(self, tool, status, output):
            self.tool = tool
            self.status = status
            self.input = {}
            self.output = output

    observations = [
        Obs("git.log", "EXECUTED", {"commits": [{"message": "fix src/app.py"}]}),
        Obs("git.diff", "EXECUTED", {"diff": "--- a/src/app.py\n+++ b/src/app.py"}),
    ]
    assert observed_paths(observations) == set()
