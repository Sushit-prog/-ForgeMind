"""ReviewResult schema (architecture doc section 11, Phase 9).

The Reviewer Agent's verdict on the developer's commit: APPROVE /
REQUEST_CHANGES / REJECT, with per-issue detail (description, severity,
file, optional line) and an overall severity. ``issues`` may be empty for
APPROVE; REQUEST_CHANGES/REJECT must carry at least one issue.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Severity = Literal["low", "medium", "high"]
ReviewDecision = Literal["APPROVE", "REQUEST_CHANGES", "REJECT"]


class ReviewIssue(BaseModel):
    description: str = Field(min_length=1, max_length=4_000)
    severity: Severity
    file: str = Field(min_length=1, max_length=512)
    line: int | None = Field(default=None, ge=1)


class ReviewResult(BaseModel):
    decision: ReviewDecision
    issues: list[ReviewIssue] = Field(default_factory=list)
    severity: Severity

    @model_validator(mode="after")
    def _issues_consistent_with_decision(self) -> "ReviewResult":
        if self.decision == "APPROVE":
            if self.issues:
                raise ValueError("APPROVE must carry no issues")
        elif not self.issues:
            raise ValueError(
                f"{self.decision} must carry at least one issue"
            )
        return self
