"""Repeat the hermetic suite until a failure is caught.

The mock-LLM test-isolation bug is order-dependent and does not reliably
reproduce on a single run, so this script runs the suite (or a forced-order
short list) repeatedly and prints which iteration failed together with the
full pytest traceback.

Usage (from the project root, with the venv active):

    python scripts/reproduce_flake.py                # 3 full-suite runs
    python scripts/reproduce_flake.py --runs 5        # 5 full-suite runs
    python scripts/reproduce_flake.py --forced-order  # run the mock-LLM
                                                      # pollutor/target files in
                                                      # one process repeatedly

Exit code 0 = all runs green; 1 = a run failed (traceback printed).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The zone the flake has historically hit, in the suite's alphabetical order:
# lifecycle/llm-provider/migrations run first (polluting), then the agent-loop
# files that fail. The file is doubled so mock/provider exhaustion across
# FILE boundaries can surface, which single-file runs never see.
FORCED_ORDER_FILES = [
    "tests/test_lifecycle_phase9.py",
    "tests/test_lifecycle_phase10.py",
    "tests/test_llm_provider.py",
    "tests/test_migrations.py",
    "tests/test_planning_agent.py",
    "tests/test_planning_lifecycle.py",
    "tests/test_policy_engine.py",
    "tests/test_prompt_injection.py",
    "tests/test_pr_template.py",
    "tests/test_debugger_agent.py",
    "tests/test_developer_agent.py",
    "tests/test_research_agent.py",
    "tests/test_reviewer_agent.py",
    "tests/test_security_agent.py",
    "tests/test_tester_agent.py",
]

BASE = ["-m", "pytest", "-x", "-q", "--tb=long", "-p", "no:cacheprovider"]


def _run(files: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *BASE, *files],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--forced-order",
        action="store_true",
        help="run the mock-LLM pollutor+target files (doubled) instead of the full suite",
    )
    args = parser.parse_args(argv)

    for i in range(1, args.runs + 1):
        files = FORCED_ORDER_FILES * 2 if args.forced_order else ["tests"]
        proc = _run(files)
        kind = "forced-order" if args.forced_order else "full-suite"
        print(f"run {i}/{args.runs} ({kind}) -> exit {proc.returncode}", flush=True)
        if proc.returncode != 0:
            print("=== pytest captured output (failure) ===", flush=True)
            print(proc.stdout)
            print(proc.stderr)
            return 1
    print(f"all {args.runs} runs green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
