# ForgeMind

Autonomous software-engineering agent: **GitHub issue → investigate → implement → test → review → PR**.

This is **Phase 1 — the foundation layer**: project skeleton, config, database schema, and a
minimal FastAPI Task API. No agents, no LLM calls, no tools yet. The only thing this milestone
must prove: *a task can be created via API and persisted.*

## Quick start

```bash
# 1. Local Postgres (prod uses Supabase)
docker compose up -d

# 2. Environment
cp .env.example .env

# 3. Install
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 4. Migrate
alembic upgrade head

# 5. Run
uvicorn app.main:app --reload
```

Then:

```bash
curl -X POST localhost:8000/tasks \
  -H 'Content-Type: application/json' \
  -d '{"objective": "Fix the flaky test in auth", "repository_url": "https://github.com/org/repo.git"}'
# -> 201 {"id": "...", "status": "CREATED", ...}

curl localhost:8000/tasks          # list
curl localhost:8000/tasks/{id}     # fetch one
curl localhost:8000/health         # {"status": "ok"}
```

## Tests

```bash
pytest                      # runs on SQLite — no Postgres needed
```

## Layout

```
app/
  api/routes/tasks.py    Task API (POST/GET /tasks, GET /tasks/{id})
  database/              engine/session + Alembic migrations
  models/                SQLAlchemy models (Section-G schema)
  schemas/               Pydantic request/response schemas
  config.py              env-driven settings (secrets never hardcoded/logged)
  logging.py             logging setup with URL redaction
  main.py                FastAPI app + /health + fail-fast DB check
tests/                   schema validation, API round-trip, migration tests
```

## Schema

Tables (architecture doc section G, milestone scope): `tasks`, `plans`, `plan_steps`,
`task_steps`, `capabilities`, `policies`, `audit_logs`, `repositories`, `worktrees`.
Remaining Section-G tables arrive with the phases that use them (agents/tools/eval).

## Security posture (Phase 1)

- Secrets come from `.env` only; connection strings are logged redacted.
- Startup fails fast if the DB is unreachable — no silent hangs.
- No shell execution, no file writes outside the repo, no external network calls yet.
