"""Harmless dummy tools that prove the pipeline (Phase 3).

- ``example.echo``      LOW risk, no capability — happy path.
- ``example.read_file`` requires ``repo.read`` — capability enforcement.
- ``example.denied``    HIGH risk, explicitly denied by policy — DENY path.

Real tools (repository.*, git.*, shell.*, github.*) arrive in later phases
and register through the same ``ToolRegistry``.
"""

from __future__ import annotations

from app.tools.base import Tool
from app.tools.examples.denied_tool import DeniedTool
from app.tools.examples.echo_tool import EchoTool
from app.tools.examples.read_only_tool import ReadOnlyTool
from app.tools.registry import ToolRegistry

EXAMPLE_TOOLS: list[Tool] = [EchoTool(), ReadOnlyTool(), DeniedTool()]


def build_default_registry() -> ToolRegistry:
    """Registry pre-loaded with the example tools."""
    registry = ToolRegistry()
    for tool in EXAMPLE_TOOLS:
        registry.register(tool)
    return registry
