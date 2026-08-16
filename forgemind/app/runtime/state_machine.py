"""Task state machine (architecture doc section D).

Pure logic only — no I/O, no LLM. ``LEGAL_TRANSITIONS`` is the single
source of truth for what a task may do; every transition in the system
(worker, API cancel, future agents) must go through ``StateMachine``.

The doc's illegal-transition rule is enforced here, deterministically —
never by asking the model "are we done?".

States (section D):

    CREATED -> PLANNING -> RESEARCHING -> IMPLEMENTING -> TESTING
        -> REVIEWING -> SECURITY_REVIEW -> VERIFICATION -> PR_CREATION
        -> AWAITING_APPROVAL -> COMPLETED
    TESTING -> DEBUGGING -> IMPLEMENTING            (failures)
    REVIEWING/SECURITY_REVIEW -> IMPLEMENTING        (reject / changes)
    AWAITING_APPROVAL -> REPLANNING                  (user rejects PR)
    ANY state -> FAILED -> RECOVERING -> REPLANNING  (failure path)
    REPLANNING -> RESEARCHING | IMPLEMENTING         (back into pipeline)
    ANY state -> ESCALATED                           (budget/replans exceeded)
    COMPLETED / ESCALATED are terminal.
"""

from __future__ import annotations

from app.models import TaskStatus


class IllegalTransitionError(Exception):
    """Raised when a transition is not in the legal table (section D)."""

    def __init__(self, current: TaskStatus, target: TaskStatus) -> None:
        self.current = current
        self.target = target
        super().__init__(f"Illegal transition: {current.value} -> {target.value}")


# Terminal states: no transitions out, ever.
TERMINAL_STATES = frozenset({TaskStatus.COMPLETED, TaskStatus.ESCALATED})


def _all_transitions() -> dict[TaskStatus, set[TaskStatus]]:
    """Build the Section-D legal-transition table."""
    failure_targets = {TaskStatus.FAILED, TaskStatus.ESCALATED}
    table: dict[TaskStatus, set[TaskStatus]] = {
        TaskStatus.CREATED: {TaskStatus.PLANNING},
        TaskStatus.PLANNING: {TaskStatus.RESEARCHING},
        TaskStatus.RESEARCHING: {TaskStatus.IMPLEMENTING},
        TaskStatus.IMPLEMENTING: {TaskStatus.TESTING},
        TaskStatus.TESTING: {TaskStatus.DEBUGGING, TaskStatus.REVIEWING},
        TaskStatus.DEBUGGING: {TaskStatus.IMPLEMENTING},
        TaskStatus.REVIEWING: {TaskStatus.SECURITY_REVIEW, TaskStatus.IMPLEMENTING},
        TaskStatus.SECURITY_REVIEW: {TaskStatus.VERIFICATION, TaskStatus.IMPLEMENTING},
        TaskStatus.VERIFICATION: {TaskStatus.PR_CREATION},
        TaskStatus.PR_CREATION: {TaskStatus.AWAITING_APPROVAL},
        TaskStatus.AWAITING_APPROVAL: {TaskStatus.COMPLETED, TaskStatus.REPLANNING},
        TaskStatus.COMPLETED: set(),
        TaskStatus.FAILED: {TaskStatus.RECOVERING},
        TaskStatus.RECOVERING: {TaskStatus.REPLANNING},
        TaskStatus.REPLANNING: {TaskStatus.RESEARCHING, TaskStatus.IMPLEMENTING},
        TaskStatus.ESCALATED: set(),
    }
    # Failure path: ANY non-terminal state may fail or escalate — except a
    # state failing *into itself* (FAILED -> FAILED is not a transition).
    for state, targets in table.items():
        if state not in TERMINAL_STATES:
            targets.update(failure_targets - {state})
    return table


LEGAL_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = _all_transitions()


class StateMachine:
    """Deterministic legal-transition enforcement (section D)."""

    def can_transition(self, current: TaskStatus, target: TaskStatus) -> bool:
        """Return True iff ``current -> target`` is legal."""
        return target in LEGAL_TRANSITIONS.get(current, set())

    def transition(self, current: TaskStatus, target: TaskStatus) -> TaskStatus:
        """Return ``target`` if legal, otherwise raise ``IllegalTransitionError``."""
        if not self.can_transition(current, target):
            raise IllegalTransitionError(current, target)
        return target


state_machine = StateMachine()
