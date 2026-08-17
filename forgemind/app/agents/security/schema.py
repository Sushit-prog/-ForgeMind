"""SecurityResult schema (architecture doc section 12, Phase 9).

The Security Agent's checklist verdict on the developer's commit: PASS or
FAIL, with per-finding category (from the Section-12 checklist: injection,
secrets, unsafe subprocess/network, path traversal, auth/authz), file,
line, description, and severity. A FAIL must carry at least one finding; a
PASS must carry none.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Severity = Literal["low", "medium", "high"]
SecurityDecision = Literal["PASS", "FAIL"]

FindingCategory = Literal[
    "INJECTION",
    "SECRETS",
    "UNSAFE_SUBPROCESS",
    "UNSAFE_NETWORK",
    "PATH_TRAVERSAL",
    "AUTH_AUTHZ",
    "OTHER",
]


class SecurityFinding(BaseModel):
    category: FindingCategory
    file: str = Field(min_length=1, max_length=512)
    line: int | None = Field(default=None, ge=1)
    description: str = Field(min_length=1, max_length=4_000)
    severity: Severity


class SecurityResult(BaseModel):
    decision: SecurityDecision
    findings: list[SecurityFinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def _findings_consistent_with_decision(self) -> "SecurityResult":
        if self.decision == "PASS":
            if self.findings:
                raise ValueError("PASS must carry no findings")
        elif not self.findings:
            raise ValueError("FAIL must carry at least one finding")
        return self
