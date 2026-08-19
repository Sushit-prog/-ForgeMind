"""Capability registry tests (architecture doc section H)."""

from app.capabilities import (
    AGENT_CAPABILITIES,
    all_capability_names,
    capabilities_for_agent,
    known_agent_types,
)
from app.capabilities.models import Capability

EXPECTED_NAMES = {
    "repo.read",
    "repo.write",
    "git.read",
    "git.write",
    "shell.test",
    "shell.build",
    "github.read",
    "github.write",
}

# Section H, verbatim.
EXPECTED_ASSIGNMENTS = {
    "research": {"repo.read", "git.read", "github.read"},
    "developer": {
        "repo.read",
        "repo.write",
        "git.read",
        "git.write",
        "shell.test",
        "shell.build",
    },
    "test": {"repo.read", "shell.test"},
    "debugger": {"repo.read", "git.read"},
    "reviewer": {"repo.read", "git.read"},
    "security": {"repo.read", "git.read"},
    "github": {"github.read", "github.write", "git.write"},
}


def test_all_capability_names_match_section_h() -> None:
    assert all_capability_names() == EXPECTED_NAMES


def test_known_agents_are_the_section_h_lineup() -> None:
    assert set(known_agent_types()) == set(EXPECTED_ASSIGNMENTS)


def test_every_agent_matches_section_h_assignment() -> None:
    for agent, expected in EXPECTED_ASSIGNMENTS.items():
        assert capabilities_for_agent(agent) == frozenset(expected), agent


def test_unknown_agent_gets_empty_set_not_crash() -> None:
    assert capabilities_for_agent("not-an-agent") == frozenset()


def test_enum_values_are_the_string_names() -> None:
    assert Capability.REPO_READ.value == "repo.read"
    assert Capability.GITHUB_WRITE.value == "github.write"
    # The registry stores enum members; the lookup layer returns strings
    # so the pipeline only ever sees plain strings.
    assert all(isinstance(c, Capability) for c in AGENT_CAPABILITIES["developer"])
    assert all(isinstance(c, str) for c in capabilities_for_agent("developer"))
