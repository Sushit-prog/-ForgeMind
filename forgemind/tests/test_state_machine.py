"""Unit tests for the Section-D state machine (pure logic, no I/O)."""

import pytest

from app.models import TaskStatus
from app.runtime.state_machine import (
    IllegalTransitionError,
    LEGAL_TRANSITIONS,
    TERMINAL_STATES,
    StateMachine,
    state_machine,
)

ALL_STATES = list(TaskStatus)


# --- legal transitions straight from section D -------------------------------

LEGAL_PAIRS = [
    (TaskStatus.CREATED, TaskStatus.PLANNING),
    (TaskStatus.PLANNING, TaskStatus.RESEARCHING),
    (TaskStatus.RESEARCHING, TaskStatus.IMPLEMENTING),
    (TaskStatus.IMPLEMENTING, TaskStatus.TESTING),
    (TaskStatus.TESTING, TaskStatus.DEBUGGING),
    (TaskStatus.TESTING, TaskStatus.REVIEWING),
    (TaskStatus.DEBUGGING, TaskStatus.IMPLEMENTING),
    (TaskStatus.REVIEWING, TaskStatus.SECURITY_REVIEW),
    (TaskStatus.REVIEWING, TaskStatus.IMPLEMENTING),
    (TaskStatus.SECURITY_REVIEW, TaskStatus.VERIFICATION),
    (TaskStatus.SECURITY_REVIEW, TaskStatus.IMPLEMENTING),
    (TaskStatus.VERIFICATION, TaskStatus.PR_CREATION),
    (TaskStatus.PR_CREATION, TaskStatus.AWAITING_APPROVAL),
    (TaskStatus.AWAITING_APPROVAL, TaskStatus.COMPLETED),
    (TaskStatus.AWAITING_APPROVAL, TaskStatus.REPLANNING),
    (TaskStatus.FAILED, TaskStatus.RECOVERING),
    (TaskStatus.RECOVERING, TaskStatus.REPLANNING),
    (TaskStatus.REPLANNING, TaskStatus.RESEARCHING),
    (TaskStatus.REPLANNING, TaskStatus.IMPLEMENTING),
]

# Any non-terminal state may fail or escalate (section D failure path).
FAILURE_TARGETS = [TaskStatus.FAILED, TaskStatus.ESCALATED]


@pytest.mark.parametrize("current,target", LEGAL_PAIRS)
def test_documented_transitions_are_legal(current, target) -> None:
    assert state_machine.can_transition(current, target)
    assert state_machine.transition(current, target) == target


def test_any_non_terminal_state_can_fail_or_escalate() -> None:
    for state in ALL_STATES:
        if state in TERMINAL_STATES:
            continue
        for target in FAILURE_TARGETS:
            if target is state:
                continue  # FAILED -> FAILED is not a transition
            assert state_machine.can_transition(state, target), (
                f"{state.value} -> {target.value} should be legal (failure path)"
            )


def test_terminal_states_have_no_outgoing_transitions() -> None:
    for terminal in TERMINAL_STATES:
        for target in ALL_STATES:
            assert not state_machine.can_transition(terminal, target)


def test_no_self_transitions() -> None:
    for state in ALL_STATES:
        assert not state_machine.can_transition(state, state)


# --- illegal transitions -----------------------------------------------------

ILLEGAL_PAIRS = [
    (TaskStatus.PLANNING, TaskStatus.COMPLETED),   # skipped every stage
    (TaskStatus.TESTING, TaskStatus.COMPLETED),    # no review gate
    (TaskStatus.CREATED, TaskStatus.PR_CREATION),  # teleport to PR
    (TaskStatus.CREATED, TaskStatus.AWAITING_APPROVAL),
    (TaskStatus.REVIEWING, TaskStatus.TESTING),    # going backwards
    (TaskStatus.COMPLETED, TaskStatus.PLANNING),   # resurrect a finished task
    (TaskStatus.ESCALATED, TaskStatus.RECOVERING),  # escalated is frozen
    (TaskStatus.REPLANNING, TaskStatus.COMPLETED),
    # NOTE: DEBUGGING -> REVIEWING is legal since Phase 8 (flaky path).
]


@pytest.mark.parametrize("current,target", ILLEGAL_PAIRS)
def test_illegal_transitions_are_rejected(current, target) -> None:
    assert not state_machine.can_transition(current, target)
    with pytest.raises(IllegalTransitionError):
        state_machine.transition(current, target)


def test_illegal_transition_error_carries_state() -> None:
    with pytest.raises(IllegalTransitionError) as exc_info:
        state_machine.transition(TaskStatus.PLANNING, TaskStatus.COMPLETED)
    assert exc_info.value.current is TaskStatus.PLANNING
    assert exc_info.value.target is TaskStatus.COMPLETED
    assert "PLANNING" in str(exc_info.value)


def test_transition_table_covers_every_state() -> None:
    # Every state must appear in the table (terminal ones with empty targets).
    assert set(LEGAL_TRANSITIONS) == set(ALL_STATES)


# --- next_status: the stub pipeline driver ----------------------------------

from app.runtime.task_lifecycle import AUTO_PIPELINE, USER_CANCELLED, next_status  # noqa: E402


def test_happy_path_walks_full_pipeline_to_completed() -> None:
    status = TaskStatus.CREATED
    seen = [status]
    while status is not None:
        status = next_status(
            status, replan_count=0, max_replans=None, last_reason=None
        )
        if status is not None:
            seen.append(status)
    assert seen == AUTO_PIPELINE
    assert seen[-1] is TaskStatus.COMPLETED


def test_user_cancelled_failed_task_is_terminal() -> None:
    assert (
        next_status(
            TaskStatus.FAILED,
            replan_count=0,
            max_replans=None,
            last_reason=USER_CANCELLED,
        )
        is None
    )


def test_failure_path_recovers_and_replans() -> None:
    s = TaskStatus.FAILED
    assert (
        next_status(s, replan_count=0, max_replans=None, last_reason=None)
        is TaskStatus.RECOVERING
    )
    s = TaskStatus.RECOVERING
    assert (
        next_status(s, replan_count=0, max_replans=None, last_reason=None)
        is TaskStatus.REPLANNING
    )


def test_replanning_exhausts_budget_into_escalated() -> None:
    s = TaskStatus.REPLANNING
    assert (
        next_status(s, replan_count=3, max_replans=3, last_reason=None)
        is TaskStatus.ESCALATED
    )


def test_replanning_with_budget_left_goes_back_to_research() -> None:
    s = TaskStatus.REPLANNING
    assert (
        next_status(s, replan_count=1, max_replans=3, last_reason=None)
        is TaskStatus.RESEARCHING
    )


def test_debugging_returns_to_implementing() -> None:
    assert (
        next_status(TaskStatus.DEBUGGING, replan_count=0, max_replans=None, last_reason=None)
        is TaskStatus.IMPLEMENTING
    )


def test_terminal_statuses_return_none() -> None:
    for terminal in TERMINAL_STATES:
        assert (
            next_status(terminal, replan_count=0, max_replans=None, last_reason=None)
            is None
        )
