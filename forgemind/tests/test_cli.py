"""CLI tests (hermetic): argparse wiring + endpoint routing — no network.

The CLI is a THIN client: these tests drive ``app.cli.main`` against the
in-process ASGI app (the shared ``client`` fixture, mock-mode env from
conftest) through a recording proxy, so an assertion on "the right endpoint
was called" is an assertion on real (method, path) pairs. No worker runs
(queue disabled), so a polled task legitimately stays CREATED — the poll
loop is exercised via ``--max-wait`` instead.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from app.cli import main
from app.models import Approval, PullRequest, Task

PAYLOAD = {
    "objective": "fix a bug",
    "repository_url": "https://github.com/o/r.git",
}


class _RecordingClient:
    """Delegates every call to the real in-process client, recording
    ``(method, path)`` so tests can assert the CLI hit the right endpoints."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.calls: list[tuple[str, str]] = []

    @property
    def base_url(self):
        return self._inner.base_url

    def get(self, url, **kwargs):
        self.calls.append(("GET", url))
        return self._inner.get(url, **kwargs)

    def post(self, url, **kwargs):
        self.calls.append(("POST", url))
        return self._inner.post(url, **kwargs)


@pytest.fixture()
def cli(client) -> _RecordingClient:
    return _RecordingClient(client)


def _create(client) -> str:
    resp = client.post("/tasks", json=PAYLOAD)
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


# --- run ---------------------------------------------------------------------


def test_run_posts_task_then_polls_until_max_wait(
    cli, db_session, capsys, monkeypatch
) -> None:
    monkeypatch.setenv("FORGEMIND_MOCK_LLM", "1")
    monkeypatch.setenv("FORGEMIND_MOCK_GITHUB", "1")

    code = main(
        [
            "run",
            "--repo",
            "octo/demo",
            "--issue",
            "7",
            "--poll-interval",
            "0",
            "--max-wait",
            "0",
        ],
        client=cli,
    )
    out = capsys.readouterr()

    assert code == 3  # --max-wait expired while the task is still CREATED
    # Right endpoints, in order: create, then poll GET /tasks/{id}.
    assert cli.calls[0] == ("POST", "/tasks")
    assert any(
        method == "GET" and path.startswith("/tasks/")
        for method, path in cli.calls[1:]
    ), cli.calls
    assert "task " in out.out and "created (CREATED)" in out.out
    assert "giving up" in out.err

    task_id = uuid.UUID(out.out.splitlines()[0].split()[1])
    task = db_session.get(Task, task_id)
    assert task is not None
    assert task.issue_number == 7
    assert task.objective == "Resolve issue #7 in octo/demo"
    assert task.repository.url == "https://github.com/octo/demo"


def test_run_sends_fork_url_when_flag_given(cli, db_session, capsys) -> None:
    code = main(
        [
            "run",
            "--repo",
            "octo/demo",
            "--issue",
            "2",
            "--fork",
            "me/demo",
            "--objective",
            "custom goal",
            "--poll-interval",
            "0",
            "--max-wait",
            "0",
        ],
        client=cli,
    )
    capsys.readouterr()

    assert code == 3
    task = db_session.scalar(select(Task).where(Task.objective == "custom goal"))
    assert task is not None
    assert task.issue_number == 2
    assert task.repository.url == "https://github.com/octo/demo"
    assert task.repository.fork_url == "https://github.com/me/demo"


def test_run_without_token_fails_before_any_request(capsys, monkeypatch) -> None:
    monkeypatch.delenv("FORGEMIND_API_TOKEN", raising=False)

    code = main(["run", "--repo", "o/n", "--issue", "1"])

    out = capsys.readouterr()
    assert code == 1
    assert "FORGEMIND_API_TOKEN" in out.err


def test_bad_args_exit_usage() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--repo", "not-a-slug", "--issue", "1"])
    assert exc_info.value.code == 2

    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--repo", "o/n", "--issue", "0"])
    assert exc_info.value.code == 2


# --- status ------------------------------------------------------------------


def test_status_prints_state_and_reports_missing_pr(cli, client, capsys) -> None:
    task_id = _create(client)

    code = main(["status", task_id], client=cli)
    out = capsys.readouterr().out

    assert code == 0
    assert task_id in out
    assert "CREATED" in out
    assert "draft PR:  (none yet)" in out
    assert ("GET", f"/tasks/{task_id}") in cli.calls
    assert ("GET", f"/tasks/{task_id}/trace") in cli.calls


def test_status_prints_draft_pr_link_scraped_from_trace(
    cli, client, db_session, capsys
) -> None:
    task_id = uuid.UUID(_create(client))
    task = db_session.get(Task, task_id)
    task.status = "AWAITING_APPROVAL"
    db_session.add(
        PullRequest(
            task_id=task_id,
            repo="me/demo",
            branch="agent/task",
            number=12,
            url="https://github.com/me/demo/pull/12",
            status="draft",
        )
    )
    db_session.commit()

    code = main(["status", str(task_id)], client=cli)
    out = capsys.readouterr().out

    assert code == 0
    assert "AWAITING_APPROVAL" in out
    assert "https://github.com/me/demo/pull/12" in out


def test_status_unknown_task_exits_error(cli, capsys) -> None:
    code = main(["status", str(uuid.uuid4())], client=cli)

    assert code == 1
    assert "404" in capsys.readouterr().err


# --- approve / reject --------------------------------------------------------


def test_approve_wrong_state_then_success_once_awaiting(
    cli, client, db_session, capsys
) -> None:
    task_id = _create(client)

    # Not awaiting approval yet -> the API's 409 surfaces, exit 1.
    code = main(["approve", task_id], client=cli)
    err = capsys.readouterr().err
    assert code == 1
    assert "409" in err
    assert "AWAITING_APPROVAL" in err

    # Park it at the checkpoint, then approve for real.
    task = db_session.get(Task, uuid.UUID(task_id))
    task.status = "AWAITING_APPROVAL"
    db_session.commit()

    code = main(["approve", task_id, "--reason", "lgtm"], client=cli)
    out = capsys.readouterr().out

    assert code == 0
    assert ("POST", f"/tasks/{task_id}/approve") in cli.calls
    assert "COMPLETED" in out
    approval = db_session.scalar(select(Approval).where(Approval.task_id == task.id))
    assert approval is not None
    assert approval.action == "approve"
    assert approval.reason == "lgtm"


def test_reject_wrong_state_reports_error(cli, client, capsys) -> None:
    task_id = _create(client)

    code = main(["reject", task_id], client=cli)
    out = capsys.readouterr()

    assert code == 1
    assert ("POST", f"/tasks/{task_id}/reject") in cli.calls
    assert "409" in out.err
