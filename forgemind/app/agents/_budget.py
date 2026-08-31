"""Shared tool-budget accounting for the agent tool-use loops.

Two independent limits govern how long an agent's tool-use loop may run:

- ``max_calls`` — the REAL budget: only *executed* tool calls count against
  it. This is the meaningful limit (each executed call does real work: a
  filesystem/git/search/subprocess action).
- ``max_rejections`` — a SMALL fixed allowance for *non-executed* rejections:
  a pre-execution schema-validation error, a capability/policy DENY, or an
  unknown tool. Such calls never reach ``execute`` and do no real work, so
  they must not consume the scarce executed-call budget.

This exists because a model with broken parameter phrasing (observed live: the
Developer proposing ``repository.search`` with ``pattern`` instead of the
required ``query`` ~28 times) otherwise burns its entire budget on call after
call that never executes, starving the agent of the budget it needs to land a
single real call / commit.

Both the Developer and Research agents use this class; ``Note`` is the single
outcome vocabulary so the accounting cannot drift between agents.
"""

from __future__ import annotations

from enum import IntEnum


class Note(IntEnum):
    """Outcome of one tool-use loop iteration."""

    EXECUTED = 1  # tool.execute ran (success or runtime failure) — costs a call
    REJECTED = 2  # never executed: schema-validation error, DENY, unknown tool
    FINAL = 3  # the model signalled {"final": true} — loop ends normally


class ToolBudget:
    """Counts executed calls vs. non-executed rejections under two caps."""

    def __init__(self, max_calls: int, max_rejections: int) -> None:
        self.max_calls = max_calls
        self.max_rejections = max_rejections
        self._executed = 0
        self._rejections = 0

    # -- recording -----------------------------------------------------------

    def note(self, outcome: Note) -> None:
        if outcome is Note.EXECUTED:
            self._executed += 1
        elif outcome is Note.REJECTED:
            self._rejections += 1
        # FINAL is a normal exit — recorded for completeness, no counter change.

    # -- queries -------------------------------------------------------------

    @property
    def executed(self) -> int:
        return self._executed

    @property
    def rejections(self) -> int:
        return self._rejections

    def should_continue(self) -> bool:
        """Keep looping while neither the executed budget nor the rejection
        allowance is exhausted."""
        return self._executed < self.max_calls and (
            self._rejections < self.max_rejections
        )

    def exhausted_by_what(self) -> str:
        """Human-readable description when the loop ends on a limit."""
        if self._executed >= self.max_calls:
            return "executed-tool-call budget exhausted"
        if self._rejections >= self.max_rejections:
            return "rejection allowance exhausted"
        return "budget not exhausted"
