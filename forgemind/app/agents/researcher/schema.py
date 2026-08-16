"""ResearchArtifact schema (architecture doc section 7) and content checks.

The schema matches Section 7 exactly. Beyond shape validation, the agent
CROSS-CHECKS the artifact's ``relevant_files``/``relevant_tests`` against
what the tool-use loop actually observed: an LLM may not claim a file is
relevant if it never read/searched/listed it. That is the research-
equivalent of the planner's DAG validation — verify content, not just
shape, against what is verifiably true.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ResearchArtifact(BaseModel):
    root_cause_hypothesis: str = Field(min_length=1, max_length=20_000)
    relevant_files: list[str] = Field(default_factory=list)
    relevant_tests: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list, min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, value: float) -> float:
        return round(value, 2)


def unobserved_files(artifact: ResearchArtifact, observed: set[str]) -> list[str]:
    """Relevant files/tests the artifact claims but the loop never saw.

    ``observed`` is the set of worktree-relative paths actually touched by
    EXECUTED tool calls during the loop. Returns the claimed-but-unseen
    paths (empty = artifact is fully grounded in observation).
    """
    claimed = {f for f in [*artifact.relevant_files, *artifact.relevant_tests] if f}
    return sorted(claimed - observed)


def observed_paths(observations: list) -> set[str]:
    """Worktree-relative paths proven to exist by EXECUTED tool results.

    repository.read_file -> the path read; repository.search -> matched
    paths; repository.list_files -> the returned listing. git.* results
    contribute evidence text, not file claims (diff/status parsing is
    deliberately not used to ground file references).
    """
    seen: set[str] = set()
    for obs in observations:
        if obs.status != "EXECUTED":
            continue
        if obs.tool == "repository.read_file":
            path = (obs.input or {}).get("path")
            if path:
                seen.add(path)
        elif obs.tool == "repository.search":
            for match in (obs.output or {}).get("matches", []):
                seen.add(match.get("path"))
        elif obs.tool == "repository.list_files":
            seen.update((obs.output or {}).get("files", []))
    return seen
