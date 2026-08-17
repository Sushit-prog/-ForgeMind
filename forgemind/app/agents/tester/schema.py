"""TestResult schema (architecture doc section 9).

The deterministic output of the Test Agent: parse the subprocess outcome —
exit code + structured output — into status/counts/failures. ``status`` is
``passed`` (exit 0), ``failed`` (exit nonzero, tests ran and failed), or
``error`` (the run itself errored: timeout, no tests collected, command
not configured). ``error`` is deliberately distinct from ``failed`` so the
Debugger can tell a hung suite from a clean failing exit code.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

TestRunStatus = Literal["passed", "failed", "error"]

# pytest summary lines: "=== 1 failed, 2 passed in 0.5s ===" / "=== 3 passed ==="
_SUMMARY_RE = re.compile(
    r"===+\s*(?:(?P<failed>\d+)\s+failed(?:,\s*)?)?(?:(?P<passed>\d+)\s+passed)?.*?==="
)
# pytest short-summary failure lines: "FAILED tests/test_app.py::test_v - AssertionError: ..."
_FAILED_LINE_RE = re.compile(r"^FAILED\s+(.+?)\s+-\s+(.*)$")


class FailureDetail(BaseModel):
    test: str
    output: str = ""


class TestResult(BaseModel):
    status: TestRunStatus
    passed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    failures: list[FailureDetail] = Field(default_factory=list)
    duration_ms: int = Field(default=0, ge=0)
    exit_code: int | None = None


def parse_test_run(*, exit_code: int | None, output: str, timed_out: bool) -> TestResult:
    """Deterministic parse of a raw subprocess run into a TestResult.

    Section 41's principle applied most literally: exit code + a structured
    parser decide, never an LLM judgment call.
    """
    output = output or ""

    if timed_out:
        return TestResult(
            status="error",
            failed=0,
            failures=[],
            duration_ms=0,
            exit_code=None,
        )

    if exit_code == 0:
        counts = _counts(output)
        return TestResult(
            status="passed",
            passed=counts["passed"],
            failed=0,
            failures=[],
            duration_ms=0,
            exit_code=0,
        )

    # Nonzero exit: parse pytest-style failures when present.
    failures: list[FailureDetail] = []
    for line in output.splitlines():
        match = _FAILED_LINE_RE.match(line.strip())
        if match:
            failures.append(FailureDetail(test=match.group(1), output=match.group(2)[:2000]))

    counts = _counts(output)
    if counts["failed"] == 0 and counts["passed"] == 0 and not failures:
        # No tests ran / unparseable suite output: an error, not a failure.
        status: TestRunStatus = "error"
    else:
        status = "failed"

    if not failures and output:
        failures = [FailureDetail(test="<unknown>", output=output[:2000])]

    return TestResult(
        status=status,
        passed=counts["passed"],
        failed=counts["failed"] if status == "failed" else 0,
        failures=failures,
        duration_ms=0,
        exit_code=exit_code,
    )


def _counts(output: str) -> dict[str, int]:
    """Extract passed/failed counts from pytest's summary line (best effort).

    pytest's ==== HEADER line ("===== test session starts =====") matches
    the delimiter shape without any counts — a match with neither group is
    skipped, so only the FOOTER summary contributes counts.
    """
    for line in output.splitlines():
        match = _SUMMARY_RE.search(line)
        if match and (match.group("passed") or match.group("failed")):
            failed = int(match.group("failed") or 0)
            passed = int(match.group("passed") or 0)
            return {"passed": passed, "failed": failed}
    return {"passed": 0, "failed": 0}


def result_from_row(row) -> TestResult:
    """Reconstruct the Pydantic ``TestResult`` from a persisted ``TestRun``
    row (plus its ``Failure`` rows) — the Debugger's input at DEBUGGING time."""
    failures = [FailureDetail(test=f.test, output=f.output) for f in row.failures]
    return TestResult(
        status=row.status,
        passed=row.passed,
        failed=row.failed,
        failures=failures,
        duration_ms=row.duration_ms,
        exit_code=row.exit_code,
    )
