# ForgeMind

Autonomous software-engineering agent: **GitHub issue → investigate → implement → test → review → PR**.

Current milestone scope:

- **Phase 1 — foundation layer**: project skeleton, config, database schema, minimal FastAPI Task API.
- **Phase 2 — task runtime**: an enforced Section-D state machine, an `execution_events` trail,
  and an arq + Redis worker that drives tasks `CREATED → PLANNING → … → COMPLETED` entirely off
  the queue.
- **Phase 3 — tool runtime**: typed tool registry, capability model, deterministic policy engine,
  and the full tool pipeline (validate → capability → policy → execute → audit) with three
  harmless example tools. No agents and no real side-effecting tools yet.

## Quick start

```bash
# 1. Postgres + Redis + worker (worker runs migrations on boot)
docker compose up -d --wait

# 2. Environment
cp .env.example .env

# 3. Install
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 4. Migrate (if you didn't let the worker container do it)
alembic upgrade head

# 5. Run the API
uvicorn app.main:app --reload

# (the worker is already running in Docker; or run it locally:)
# arq app.worker.worker.WorkerSettings
```

Then:

```bash
curl -X POST localhost:8000/tasks \
  -H 'Content-Type: application/json' \
  -d '{"objective": "Fix the flaky test in auth", "repository_url": "https://github.com/org/repo.git"}'
# -> 201 {"id": "...", "status": "CREATED", ...}  (advance_task job enqueued)

curl localhost:8000/tasks                # list
curl localhost:8000/tasks/{id}           # fetch one — watch status walk to COMPLETED
curl localhost:8000/tasks/{id}/events    # ordered execution-event trail
curl -X POST localhost:8000/tasks/{id}/cancel   # -> FAILED (user_cancelled); 409 on terminal tasks
curl localhost:8000/health               # {"status": "ok"}
```

## Tests

```bash
pytest              # hermetic suite (SQLite, no services) — state machine, lifecycle, API
pytest tests_e2e/   # end-to-end (needs `docker compose up -d --wait`) — real worker pipeline,
                    # cancel, crash recovery (kill/restart), two-worker concurrency
```

## Layout

```
app/
  api/routes/tasks.py       Task API: create/list/get/cancel/events
  capabilities/             Capability value objects + per-agent assignment (Section H)
  database/                 engine/session + Alembic migrations
  execution/tool_pipeline.py  validate -> capability -> policy -> execute -> audit
  models/                   SQLAlchemy models (Section-G schema) + ExecutionEvent + ToolCall
  policies/                 deterministic PolicyEngine + risk-default & explicit-deny rules
  runtime/
    state_machine.py        Section-D legal-transition table — pure logic, no I/O
    task_lifecycle.py       atomic transitions + execution_events + stub pipeline driver
  schemas/                  Pydantic request/response schemas
  tools/                    Tool ABC + registry + example tools (echo / read_file / denied)
  worker/
    queue.py                arq Redis settings + pool + enqueue helper
    jobs/advance_task.py    one job = one transition (FOR UPDATE, re-enqueue)
    worker.py               arq entrypoint + startup sweep (crash recovery)
  config.py                 env-driven settings (secrets never hardcoded/logged)
  logging.py                logging setup with URL redaction
  main.py                   FastAPI app + /health + fail-fast DB check
tests/                      hermetic suite (SQLite)
tests_e2e/                  end-to-end suite (Postgres + Redis + worker subprocesses)
```

## Tool pipeline (architecture doc sections F/H)

Every tool invocation goes through `app/execution/tool_pipeline.py`:

```
validate input -> capability check -> policy check -> execute -> audit
```

- **Exactly one `tool_calls` row per invocation**, whatever the outcome
  (DENIED / EXECUTED / FAILED; ALLOWED is the transient admit state).
- **Fail-closed policy engine**: pure functions over typed input, no LLM;
  any rule's DENY wins over any ALLOW vote.
- **Secrets redacted** from stored input/output (`redact_sensitive`) — the
  structured-data analogue of the Phase-1 URL redaction.
- Contract errors (unknown tool, malformed input) raise loudly and never
  execute or write a row.

Example tools prove the paths: `example.echo` (executes), `example.read_file`
(denied without `repo.read`), `example.denied` (denied by explicit policy).
Real tools (`repository.*`, `git.*`, `shell.*`, `github.*`) register through
the same `ToolRegistry` in later phases.

## State machine (architecture doc section D)

Transitions are enforced by a deterministic lookup table (`runtime/state_machine.py`),
never by asking the LLM "are we done?" Illegal transitions raise `IllegalTransitionError`
and are logged — `tasks.status` is never silently updated. Every applied transition writes an
`execution_events` row in the same transaction, so a crash mid-run always leaves the task at
its last persisted status; the worker's startup sweep re-enqueues it from there.

## Schema

Tables (architecture doc section G, milestone scope): `tasks`, `plans`, `plan_steps`,
`task_steps`, `capabilities`, `policies`, `audit_logs`, `repositories`, `worktrees`,
`execution_events`, `tool_calls`. Remaining Section-G tables arrive with the phases that
use them (agents/eval).

## Security posture

- Secrets come from `.env` only; connection strings are logged redacted.
- Startup fails fast if the DB is unreachable — no silent hangs.
- The worker uses the same env-driven config as the API — no new secret surface.
- No shell execution, no file writes outside the repo, no external network calls yet.
