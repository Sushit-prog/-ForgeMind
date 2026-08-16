"""Capability value objects (architecture doc section H).

Capabilities are the unit of access control. Each agent is assigned a subset
(see ``registry.py``); a tool declares the capabilities REQUIRED to invoke it
(``Tool.capabilities``), and the pipeline denies the call if any are missing
from the calling agent's set — checked in code, never by asking the model.
"""

from __future__ import annotations

import enum


class Capability(str, enum.Enum):
    """The capability set for this domain (section H)."""

    REPO_READ = "repo.read"
    REPO_WRITE = "repo.write"
    GIT_READ = "git.read"
    GIT_WRITE = "git.write"
    SHELL_TEST = "shell.test"
    SHELL_BUILD = "shell.build"
    GITHUB_READ = "github.read"
    GITHUB_WRITE = "github.write"

    def __str__(self) -> str:  # ergonomic: str(Capability.REPO_READ) == "repo.read"
        return self.value
