"""ImplementationSummary schema (architecture doc section 7) and content checks.

``ImplementationSummary`` is the Developer Agent's final artifact: the commit
sha, the files that changed, a plain-text summary, tests touched, and an
optional explanation of any divergence from the research hypothesis.

As with Research's artifact, shape validation is not enough: the agent
CROSS-CHECKS ``files_changed`` against what the tool-use loop actually wrote
(``filesystem.write_file`` EXECUTED observations). An LLM may not claim it
changed a file it never wrote. The same accept-with-warning-after-one-retry
policy as Phase 6 applies (see ``agent._synthesize``): a mismatch is corrected
once, then accepted with a loud audit, because the committed diff remains
independently verifiable ground truth downstream.

``commit_sha`` is deliberately NOT part of what the LLM produces — the runtime
records the sha of the actual commit server-side and injects it. The LLM
parses into ``ImplementationSummaryDraft``; the runtime builds the final
summary with the real sha, so the artifact can never claim a fabricated commit.
"""

from __future__ import annotations

import posixpath

from pydantic import BaseModel, Field, field_validator


def normalize_path(path: str) -> str:
    """Canonical worktree-relative form: ``./src/./app.py`` -> ``src/app.py``.

    Applied on BOTH sides of the files-changed cross-check so a cosmetic
    difference in the LLM's path spelling is not treated as a mismatch.
    """
    cleaned = posixpath.normpath(path.replace("\\", "/"))
    if cleaned == ".":
        return "."
    return cleaned.lstrip("./")


class ImplementationSummaryDraft(BaseModel):
    """What the LLM produces during synthesis — no commit_sha (runtime fills it)."""

    files_changed: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1, max_length=20_000)
    tests_added: list[str] = Field(default_factory=list)
    deviations_from_research: str | None = None

    @field_validator("files_changed", "tests_added")
    @classmethod
    def _no_blank_paths(cls, value: list[str]) -> list[str]:
        return [normalize_path(v) for v in value if v.strip()]


class ImplementationSummary(ImplementationSummaryDraft):
    """The final, persisted artifact — commit_sha injected by the runtime."""

    commit_sha: str = Field(min_length=7, max_length=64)


def written_paths(observations: list) -> set[str]:
    """Worktree-relative paths actually written by EXECUTED write_file calls.

    Only EXECUTED writes count: a FAILED write (e.g. a traversal attempt)
    never touched the filesystem, and a DENIED write (post-commit guard)
    never reached it either.
    """
    written: set[str] = set()
    for obs in observations:
        if obs.status != "EXECUTED" or obs.tool != "filesystem.write_file":
            continue
        path = (obs.input or {}).get("path")
        if path:
            written.add(normalize_path(path))
    return written


def files_changed_mismatch(summary: ImplementationSummary, written: set[str]) -> list[str]:
    """Files the summary claims but never wrote, or wrote but never claimed.

    Returns the symmetric-difference, sorted. Empty = the summary is fully
    grounded in what the loop actually wrote.
    """
    claimed = {normalize_path(f) for f in summary.files_changed if f}
    return sorted(claimed ^ written)


def research_deviations(written: set[str], research_files: list[str]) -> list[str]:
    """Files the developer wrote that research never flagged — the changes a
    downstream reviewer cannot take on faith and must see explained."""
    researched = {normalize_path(f) for f in research_files if f}
    return sorted(written - researched)
