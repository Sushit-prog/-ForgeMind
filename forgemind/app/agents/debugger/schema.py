"""FailureClassification schema (architecture doc section 10).

The Debugger Agent's output: a category for WHY the test run failed, a
root-cause explanation, a CONCRETE fix instruction handed to the
Developer's next run (never "fix the error"), and whether the failure is
code-fixable at all. ``is_flaky`` is NOT an LLM guess — it is set
deterministically by the Debugger when the single re-run passes (the
re-run is the only legitimate source of a flaky label; see ``agent.py``).

Categories (section 10):

- CODE_FAILURE       — the implementation broke a test; Developer can fix it.
- TEST_FAILURE       — the test itself is wrong; Developer can fix it.
- ENVIRONMENT_FAILURE — the environment can't run the suite (missing
                        binary, permissions, no services); NOT code-fixable.
- DEPENDENCY_FAILURE — a dependency can't be installed/resolved; usually
                        not code-fixable (a lockfile fix may be, at best).
- FLAKY_TEST         — observed intermittent behavior via the re-run; does
                        not block the pipeline but is loudly flagged.
- UNKNOWN            — cannot determine the cause; treat as not fixable.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

FailureCategory = Literal[
    "CODE_FAILURE",
    "TEST_FAILURE",
    "ENVIRONMENT_FAILURE",
    "DEPENDENCY_FAILURE",
    "FLAKY_TEST",
    "UNKNOWN",
]

# Categories that a re-run of the Developer can never fix — routing them
# back to IMPLEMENTING would just burn replans on the same failure.
UNFIXABLE_CATEGORIES = frozenset(
    {"ENVIRONMENT_FAILURE", "DEPENDENCY_FAILURE", "UNKNOWN"}
)


class FailureClassification(BaseModel):
    category: FailureCategory
    root_cause: str = Field(min_length=1)
    fix_instruction: str | None = Field(
        default=None,
        description=(
            "Concrete, specific instruction for the Developer's next run — "
            "never 'fix the error'. Must name the file/behavior to change."
        ),
    )
    fixable: bool = Field(
        default=True,
        description=(
            "False -> the task escalates/fails rather than replanning again. "
            "A CODE_FAILURE/TEST_FAILURE is normally fixable; environment/"
            "dependency/unknown failures are not."
        ),
    )
    # Set by the Debugger agent deterministically from the re-run, never by
    # the LLM — the LLM cannot guess flakiness from a single run.
    is_flaky: bool = Field(default=False)

    @model_validator(mode="after")
    def _category_consistency(self) -> "FailureClassification":
        if self.category == "FLAKY_TEST":
            if not self.is_flaky:
                raise ValueError(
                    "category FLAKY_TEST requires is_flaky=True — the "
                    "flaky label comes from the re-run, not a guess"
                )
        if self.is_flaky and self.category != "FLAKY_TEST":
            raise ValueError(
                "is_flaky=True requires category FLAKY_TEST — the flaky "
                "label is the category"
            )
        if self.category in UNFIXABLE_CATEGORIES and self.fixable:
            raise ValueError(
                f"category {self.category} is not code-fixable — fixable must be False"
            )
        if self.fixable and not self.fix_instruction:
            raise ValueError(
                "a fixable failure requires a concrete fix_instruction — "
                "never hand the developer 'fix the error'"
            )
        return self
