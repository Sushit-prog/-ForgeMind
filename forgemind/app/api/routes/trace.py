"""Read-only execution trace viewer (Phase 11).

``GET /tasks/{id}/trace`` renders the task's full ``execution_events``
history — plan DAG, agent transitions, tool calls, test results, review and
security verdicts, PR link — as a human-readable HTML timeline instead of
raw JSON.

AUTH NOTE: this route is READ-ONLY and deliberately UNAUTHENTICATED —
the same category as the open ``GET /tasks/{id}`` and
``GET /tasks/{id}/events``. The Phase 10.5 bearer token gates only
state-mutating routes (``POST /tasks``, cancel, approve, reject); the
trace page never mutates anything, so it stays open so a human can watch
a task walk the pipeline without the token.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models import Task
from app.runtime.task_trace import build_task_trace

router = APIRouter(prefix="/tasks", tags=["trace"])

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


@router.get("/{task_id}/trace", response_class=HTMLResponse)
def task_trace(
    task_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_db),
) -> HTMLResponse:
    """Render the human-readable execution trace for one task."""
    task = db.get(Task, task_id)
    if task is None:
        return templates.TemplateResponse(
            request,
            "not_found.html",
            {"title": "Task not found", "task_id": str(task_id)},
            status_code=404,
        )
    return templates.TemplateResponse(request, "trace.html", build_task_trace(db, task))
