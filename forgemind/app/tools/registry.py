"""Tool registry (architecture doc section F).

The registry is a plain map — it has NO hardcoded tool list. Tools are
registered explicitly (the examples module does so at import time via
``build_default_registry``); real tools (``repository.*``, ``git.*``,
``shell.*``, ``github.*``) register the same way in later phases.
"""

from __future__ import annotations

from app.tools.base import Tool


class ToolNotFoundError(LookupError):
    """Raised when a tool name is not registered — never a silent no-op."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Tool not found in registry: {tool_name!r}")


class DuplicateToolError(ValueError):
    """Raised when registering a tool whose name is already taken."""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        """Register ``tool`` under its name; reject duplicates loudly."""
        if not isinstance(tool, Tool):
            raise TypeError(f"expected a Tool instance, got {type(tool).__name__}")
        if tool.name in self._tools:
            raise DuplicateToolError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool
        return tool

    def get(self, tool_name: str) -> Tool:
        """Look up a tool by name, or raise ``ToolNotFoundError``."""
        try:
            return self._tools[tool_name]
        except KeyError as exc:
            raise ToolNotFoundError(tool_name) from exc

    def list(self) -> list[Tool]:
        """All registered tools (registration order)."""
        return list(self._tools.values())

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
