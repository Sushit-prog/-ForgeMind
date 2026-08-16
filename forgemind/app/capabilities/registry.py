"""Capability registry: which capabilities exist and which agent gets what.

The per-agent assignments mirror architecture doc section H exactly. Tools
are never granted capabilities here — this is a lookup for the calling
agent's set, consumed by the capability check in the pipeline.
"""

from __future__ import annotations

from app.capabilities.models import Capability

# Section H: capability set assigned per agent, never all-at-once.
AGENT_CAPABILITIES: dict[str, frozenset[Capability]] = {
    "research": frozenset({Capability.REPO_READ, Capability.GIT_READ, Capability.GITHUB_READ}),
    "developer": frozenset(
        {
            Capability.REPO_READ,
            Capability.REPO_WRITE,
            Capability.GIT_READ,
            Capability.GIT_WRITE,
            Capability.SHELL_TEST,
            Capability.SHELL_BUILD,
        }
    ),
    "test": frozenset({Capability.REPO_READ, Capability.SHELL_TEST}),
    "debugger": frozenset({Capability.REPO_READ, Capability.GIT_READ}),
    "reviewer": frozenset({Capability.REPO_READ, Capability.GIT_READ}),
    "security": frozenset({Capability.REPO_READ, Capability.GIT_READ}),
    "github": frozenset({Capability.GITHUB_READ, Capability.GITHUB_WRITE}),
}


def all_capability_names() -> set[str]:
    """Every capability that exists in this domain (as strings)."""
    return {c.value for c in Capability}


def known_agent_types() -> list[str]:
    """Agent types that have a capability assignment (section H lineup)."""
    return list(AGENT_CAPABILITIES)


def capabilities_for_agent(agent_type: str) -> frozenset[str]:
    """The capability set for ``agent_type``, as strings (pipeline input).

    Unknown agents get an EMPTY set — safe default: a capability-gated tool
    is denied rather than crashing or being implicitly granted anything.
    """
    return frozenset(c.value for c in AGENT_CAPABILITIES.get(agent_type, frozenset()))
