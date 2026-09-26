"""ForgeMind CLI — a thin HTTP client around the Task API.

No backend logic lives here: every subcommand maps 1:1 onto an existing
route and prints what the API answers.

    forgemind run   --repo OWNER/NAME --issue N   -> POST /tasks, then poll
                                                     GET /tasks/{id} until a
                                                     terminal state
    forgemind approve TASK_ID [--reason ...]      -> POST /tasks/{id}/approve
    forgemind reject  TASK_ID [--reason ...]      -> POST /tasks/{id}/reject
    forgemind status  TASK_ID                     -> GET  /tasks/{id}
                                                     (+ draft PR link)

Auth: ``FORGEMIND_API_TOKEN`` is sent as ``Authorization: Bearer …`` on the
state-mutating routes (the API's own contract — read-only GETs ignore it).
Base URL: ``--url`` or ``FORGEMIND_API_URL``, default http://localhost:8000.

The draft PR URL is NOT part of ``TaskRead`` — it lives in the
``pull_requests`` row and is rendered only by the read-only HTML trace page
(``GET /tasks/{id}/trace``), so the CLI scrapes the ``…/pull/N`` href from
that page. Until the task reaches AWAITING_APPROVAL there is no PR to show,
and the CLI says so rather than inventing a link.

Exit codes: 0 = success (COMPLETED / AWAITING_APPROVAL / decision recorded /
status printed); 1 = task FAILED/ESCALATED, HTTP error, or missing config;
2 = bad usage (argparse); 3 = ``--max-wait`` expired while still running.

Installed as the ``forgemind`` console script (``[project.scripts]``).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

import httpx

DEFAULT_BASE_URL = "http://localhost:8000"

# Poll stops: the human checkpoint, the happy end, and the two dead ends.
STOP_STATES = frozenset({"AWAITING_APPROVAL", "COMPLETED", "FAILED", "ESCALATED"})

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_TIMEOUT = 3

# The trace template renders the draft PR as
#   <a href="https://github.com/owner/repo/pull/12" …>#12 (draft)</a>
_PR_HREF_RE = re.compile(r'href="(https://github\.com/[^"\s]+/pull/\d+[^"\s]*)"')


def _repo_slug(value: str) -> str:
    """argparse type: ``OWNER/NAME`` — reject anything else at parse time."""
    owner, sep, name = value.partition("/")
    if not sep or not owner or not name or "/" in name or " " in value:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not an OWNER/NAME repository slug"
        )
    return value


def _issue_number(value: str) -> int:
    """argparse type: a positive issue number (the API's ``ge=1``)."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
    if number < 1:
        raise argparse.ArgumentTypeError("issue number must be >= 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forgemind",
        description="Thin client for the ForgeMind task API.",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("FORGEMIND_API_URL") or DEFAULT_BASE_URL,
        help=f"API base URL (env FORGEMIND_API_URL; default {DEFAULT_BASE_URL})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run",
        help="create a task and follow its phase until approval/terminal",
    )
    run.add_argument("--repo", required=True, type=_repo_slug, metavar="OWNER/NAME")
    run.add_argument(
        "--issue",
        required=True,
        type=_issue_number,
        metavar="N",
        help="GitHub issue number (>= 1)",
    )
    run.add_argument(
        "--fork",
        type=_repo_slug,
        metavar="OWNER/NAME",
        help="fork the task pushes to (fork_url; without it PR_CREATION fails closed)",
    )
    run.add_argument(
        "--objective",
        help='task objective (default: "Resolve issue #N in OWNER/NAME")',
    )
    run.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="seconds between GET /tasks/{id} polls (default 5)",
    )
    run.add_argument(
        "--max-wait",
        type=float,
        default=None,
        metavar="SECONDS",
        help="give up after SECONDS of polling (default: wait forever)",
    )

    for name in ("approve", "reject"):
        decision = sub.add_parser(name, help=f"{name} a task awaiting approval")
        decision.add_argument("task_id", metavar="TASK_ID")
        decision.add_argument("--reason", help="recorded on the approvals row")

    status = sub.add_parser("status", help="one-shot status (+ draft PR link)")
    status.add_argument("task_id", metavar="TASK_ID")

    return parser


def _print_http_error(resp: httpx.Response) -> None:
    detail: object | None = None
    try:
        body = resp.json()
        if isinstance(body, dict):
            detail = body.get("detail")
    except ValueError:
        detail = (resp.text or "")[:200] or None
    suffix = f" — {detail}" if detail else ""
    print(f"error: HTTP {resp.status_code}{suffix}", file=sys.stderr)


def _require_token() -> str | None:
    token = os.environ.get("FORGEMIND_API_TOKEN") or None
    if not token:
        print(
            "error: FORGEMIND_API_TOKEN is not set — the API rejects "
            "run/approve/reject without it",
            file=sys.stderr,
        )
    return token


def _scrape_pr_url(client: httpx.Client, task_id: str) -> str | None:
    """Draft PR link from the read-only trace page (see module docstring)."""
    try:
        resp = client.get(f"/tasks/{task_id}/trace")
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    match = _PR_HREF_RE.search(resp.text)
    return match.group(1) if match else None


def _trace_url(client: httpx.Client, task_id: str) -> str:
    return f"{str(client.base_url).rstrip('/')}/tasks/{task_id}/trace"


def _cmd_run(client: httpx.Client, args: argparse.Namespace) -> int:
    payload: dict[str, object] = {
        "objective": args.objective
        or f"Resolve issue #{args.issue} in {args.repo}",
        "repository_url": f"https://github.com/{args.repo}",
        "issue_number": args.issue,
    }
    if args.fork:
        payload["fork_url"] = f"https://github.com/{args.fork}"

    resp = client.post("/tasks", json=payload)
    if resp.status_code != 201:
        _print_http_error(resp)
        return EXIT_ERROR
    task_id = resp.json()["id"]
    last = resp.json()["status"]
    print(f"task {task_id} created ({last})")

    deadline = (
        time.monotonic() + args.max_wait if args.max_wait is not None else None
    )
    while True:
        time.sleep(max(args.poll_interval, 0))
        resp = client.get(f"/tasks/{task_id}")
        if resp.status_code != 200:
            _print_http_error(resp)
            return EXIT_ERROR
        current = resp.json()["status"]
        if current != last:
            print(f"{last} → {current}")
            last = current
        if current in STOP_STATES:
            return _finish_run(client, task_id, current)
        if deadline is not None and time.monotonic() >= deadline:
            print(
                f"giving up after {args.max_wait}s — task still {current} "
                f"(trace: {_trace_url(client, task_id)})",
                file=sys.stderr,
            )
            return EXIT_TIMEOUT


def _finish_run(client: httpx.Client, task_id: str, status: str) -> int:
    pr_url = _scrape_pr_url(client, task_id)
    if status == "AWAITING_APPROVAL":
        print("\nAwaiting your approval.")
        if pr_url:
            print(f"Draft PR: {pr_url}")
        else:
            print(f"Trace:    {_trace_url(client, task_id)}")
        print(f"Next:     forgemind approve {task_id}")
        print(f"          forgemind reject  {task_id}")
        return EXIT_OK
    if status == "COMPLETED":
        if pr_url:
            print(f"Draft PR: {pr_url}")
        print("task COMPLETED")
        return EXIT_OK
    # FAILED / ESCALATED — the reason is on the trace page.
    print(f"task {status} — {_trace_url(client, task_id)}", file=sys.stderr)
    return EXIT_ERROR


def _cmd_status(client: httpx.Client, args: argparse.Namespace) -> int:
    resp = client.get(f"/tasks/{args.task_id}")
    if resp.status_code != 200:
        _print_http_error(resp)
        return EXIT_ERROR
    task = resp.json()
    print(f"{task['id']}  {task['status']}")
    print(f"objective: {task['objective']}")
    print(f"updated:   {task['updated_at']}")
    if task.get("recommended_action"):
        print(
            f"recommend: {task['recommended_action']}"
            + (f" — {task['recommendation_reason']}" if task.get("recommendation_reason") else "")
        )
    pr_url = _scrape_pr_url(client, str(task["id"]))
    print(f"draft PR:  {pr_url}" if pr_url else "draft PR:  (none yet)")
    return EXIT_OK


def _cmd_decide(client: httpx.Client, args: argparse.Namespace) -> int:
    resp = client.post(
        f"/tasks/{args.task_id}/{args.command}",
        json={"reason": args.reason} if args.reason else {},
    )
    if resp.status_code != 200:
        _print_http_error(resp)
        return EXIT_ERROR
    task = resp.json()
    print(f"{args.command} recorded — task {task['id']} is now {task['status']}")
    return EXIT_OK


def main(argv: list[str] | None = None, *, client: httpx.Client | None = None) -> int:
    """Console-script entry point (``[project.scripts] forgemind``).

    ``client`` is injectable so tests drive the in-process ASGI app instead
    of a live server; production builds its own httpx client from ``--url``
    and ``FORGEMIND_API_TOKEN``.
    """
    args = build_parser().parse_args(argv)

    owns_client = client is None
    if client is None:
        token = _require_token() if args.command in {"run", "approve", "reject"} else None
        if args.command in {"run", "approve", "reject"} and not token:
            return EXIT_ERROR
        client = httpx.Client(
            base_url=args.url,
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=30.0,
        )

    try:
        if args.command == "run":
            return _cmd_run(client, args)
        if args.command == "status":
            return _cmd_status(client, args)
        return _cmd_decide(client, args)
    except httpx.HTTPError as exc:
        print(f"error: request to {client.base_url} failed: {exc}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        if owns_client:
            client.close()


if __name__ == "__main__":  # pragma: no cover — `python -m app.cli`
    sys.exit(main())
