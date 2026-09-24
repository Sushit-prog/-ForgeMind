"""TestResult schema (architecture doc section 9).

The deterministic output of the Test Agent: parse the subprocess outcome —
exit code + structured output — into status/counts/failures. ``status`` is
``passed`` (exit 0), ``failed`` (exit nonzero, tests ran and failed), or
``error`` (the run itself errored: timeout, no tests collected, command
not configured). ``error`` is deliberately distinct from ``failed`` so the
Debugger can tell a hung suite from a clean failing exit code.

``install_failed`` (Phase 13) is NOT a test-run outcome at all: it is the
Test Agent's signal when dependency provisioning (``shell.install_deps``)
could not be produced, which the lifecycle maps to
``TESTING -> FAILED(dependency_install_failed)`` — never DEBUGGING. No
TestRun is persisted for it (no test ran); the reason lives on the
transition event and the install output on the install tool's audit row.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

TestRunStatus = Literal["passed", "failed", "error", "install_failed"]

# pytest summary lines: "=== 1 failed, 2 passed in 0.5s ===" / "=== 3 passed ==="
# and pytest 9's bare footer "1 failed, 335 passed in 106.72s (0:01:46)" (no
# surrounding "=" delimiters) — the "=" wrap is optional, and a line only
# counts once it carries a real "passed" group.
_SUMMARY_RE = re.compile(
    r"^(?:=+\s*)?(?:"
    r"(?P<failed>\d+)\s+failed(?:,\s*\d+\s+(?:skipped|error|warning)s?)?(?:,\s*)?"
    r")?(?P<passed>\d+)\s+passed"
    r"(?:,\s*\d+\s+(?:skipped|error|warning)s?)?(?:\s+in\s+[^=]*)?=*$"
)
# pytest short-summary failure lines: "FAILED tests/test_app.py::test_v - ..."
# (legacy) AND pytest 8+'s bare "FAILED tests/x.py::test_y" (no " - reason").
_FAILED_LINE_RE = re.compile(r"^FAILED\s+(.+?)(?:\s+-\s+(.*))?$")
# Python "No module named X" traceback lines (Phase 13 diagnostic aid).
_MISSING_MODULE_RE = re.compile(
    r"ModuleNotFoundError:\s*No module named ['\"]([^'\"]+)['\"]"
)


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
    # Phase 13 diagnostic: module names the output reports as missing. A
    # deterministic helper (not an LLM judgment) that steadies the Debugger's
    # DEPENDENCY_FAILURE classification — it is an AID, never a router.
    missing_modules: list[str] = Field(default_factory=list)


def parse_test_run(
    *, exit_code: int | None, output: str, timed_out: bool
) -> TestResult:
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
            failures.append(
                FailureDetail(test=match.group(1), output=(match.group(2) or "")[:2000])
            )

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
        missing_modules=detect_missing_modules(output),
    )


def detect_missing_modules(output: str) -> list[str]:
    """Module names a ``ModuleNotFoundError`` traceback reports as missing.

    Deterministic and defense-in-depth only: after always-on provisioning
    this should rarely fire, but when it does (a dep the install command
    does not cover) the Debugger gets a concrete name instead of re-reading
    raw output — steering toward DEPENDENCY_FAILURE, never a DEBUGGING loop.
    """
    names: list[str] = []
    for match in _MISSING_MODULE_RE.finditer(output):
        name = match.group(1)
        if name not in names:
            names.append(name)
    return names


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
        missing_modules=detect_missing_modules(getattr(row, "output", None) or ""),
    )
