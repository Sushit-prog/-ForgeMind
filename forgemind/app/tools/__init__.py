from app.tools.base import ExecutionContext, RiskLevel, Tool
from app.tools.examples import build_default_registry
from app.tools.filesystem_tools import FILESYSTEM_TOOLS
from app.tools.git_tools import GIT_TOOLS
from app.tools.registry import DuplicateToolError, ToolNotFoundError, ToolRegistry
from app.tools.repository_tools import REPOSITORY_TOOLS
from app.tools.shell_tools import SHELL_TOOLS


def build_runtime_registry() -> ToolRegistry:
    """The full registry: example tools + repository.* + git.* +
    filesystem.* (Phase 7) + shell.* (Phase 8). Real github.* tools
    register the same way in later phases.
    """
    registry = build_default_registry()
    for tool in (*REPOSITORY_TOOLS, *GIT_TOOLS, *FILESYSTEM_TOOLS, *SHELL_TOOLS):
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
