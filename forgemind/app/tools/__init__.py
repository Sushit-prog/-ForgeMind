from app.tools.base import ExecutionContext, RiskLevel, Tool
from app.tools.examples import build_default_registry
from app.tools.git_tools import GIT_TOOLS
from app.tools.registry import DuplicateToolError, ToolNotFoundError, ToolRegistry
from app.tools.repository_tools import REPOSITORY_TOOLS


def build_runtime_registry() -> ToolRegistry:
    """The full Phase-4 registry: example tools + repository.* + git.*.

    Real tools (shell.*, github.*) register the same way in later phases.
    """
    registry = build_default_registry()
    for tool in (*REPOSITORY_TOOLS, *GIT_TOOLS):
        registry.register(tool)
    return registry


__all__ = [
    "DuplicateToolError",
    "ExecutionContext",
    "RiskLevel",
    "Tool",
    "ToolNotFoundError",
    "ToolRegistry",
    "build_default_registry",
    "build_runtime_registry",
]
