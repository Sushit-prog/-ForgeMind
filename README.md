# ForgeMind

ForgeMind is an autonomous software-engineering agent that takes a GitHub
issue and drives it to a reviewable pull request: it plans the work, researches
the codebase, implements a fix, runs the real test suite, reviews and
security-checks its own commit, opens a **draft PR on a fork**, and then stops
for a human to approve or reject it. Every step is persisted as an auditable
trace you can watch in a browser.

One-line architecture pitch: same design lineage as the agentops-style
autonomous coding agents, but with a **capability-gated, deterministic policy
engine** and a **fail-closed state machine** — the LLM proposes, the pipeline
enforces. Nothing merges without a human.

## The journey

A task walks the pipeline as a sequence of typed agents, each with a fixed,
structurally-enforced capability set:

1. **Planning** — turns the objective into a schema-validated plan DAG (the only
   LLM that can't call tools).
2. **Research** — bounded READ-ONLY tool loop (search/read/list + git status/diff)
   ending in a grounded research artifact; file claims are cross-checked against
   what the loop actually observed.
3. **Developer** — the write step: edits files in an isolated per-task worktree,
   commits **exactly once**, and its self-reported summary is cross-checked
   against the files actually written.
4. **Test** — runs the repository's *configured* test command deterministically
   (no LLM judgment). Failures shift to the **Debugger**, which investigates
   read-only, classifies the failure, and — if fixable — re-enters implementation
   through the one shared replan path (bounded by `max_replans`).
5. **Reviewer + Security** — both critique the **same commit** (diff + test
   result only) under read-only capabilities, each blind to the other's verdict.
6. **Verification** — a plain-code staleness check: the reviewed commit must still
   be worktree HEAD with a passing test run.
7. **GitHub** — the verified branch lands as a real **draft PR on a fork**
   (`git.push` is the only push in the system, and there is no merge primitive).
8. **Human checkpoint** — the task parks in `AWAITING_APPROVAL`; you review the
   draft PR and `approve` (→ COMPLETED) or `reject` (→ FAILED). Merging stays
   manual on GitHub.

## Architecture

The pipeline is one task state machine enforced in code, never by asking the
LLM "are we done?":

```
CREATED → PLANNING → RESEARCHING → IMPLEMENTING → TESTING → REVIEWING
        → SECURITY_REVIEW → VERIFICATION → PR_CREATION → AWAITING_APPROVAL → COMPLETED

TESTING → DEBUGGING → IMPLEMENTING           tests failed; fixable, replan (bounded)
REVIEWING / SECURITY_REVIEW → IMPLEMENTING   changes requested / security fail
VERIFICATION → TESTING                       reviewed commit went stale
AWAITING_APPROVAL → COMPLETED | FAILED       human decision
FAILED → RECOVERING → REPLANNING             recoverable failures re-enter the pipeline
any state → FAILED | ESCALATED               unrecoverable / replan budget exhausted
COMPLETED / ESCALATED = terminal
```

What makes it hold together:

- **Capability gates.** Every agent lists the capabilities it may use
  (`repo.read`, `repo.write`, `git.read`, ...). The tool pipeline checks the
  agent's set at call time — a read-only agent *cannot* write, by construction.
- **A deterministic policy engine.** Pure functions over typed input, no LLM:
  any rule's DENY wins over any ALLOW. Unknown tools and malformed input raise
  loudly — never a silent no-op.
- **Strictly-structured LLM output.** Agents emit schema-validated objects; the
  provider either returns a fully-valid parse or raises — malformed output
  retries with a correction prompt, then fails the task. Objectives and repo
  content are treated as **data, not instructions** (prompt-injection tests
  prove injected "instructions" can never persist).
- **Isolation by design.** Each task gets its own git worktree off a
  clone-once cache; file access is worktree-scoped with airtight path-traversal
  defense; `shell.run_test` runs the allowlist-validated test command as an
  argument list, never `shell=True`.
- **A complete audit trail.** Every transition, tool call, test run, review and
  security verdict, PR, and human decision is persisted — and now viewable as a
  rendered trace.

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

Then (mutating calls need the bearer token — dev fallback shown inline):

```bash
# A single shared-secret token gates the mutating routes.
AUTH="Authorization: Bearer ${FORGEMIND_API_TOKEN:-forgemind-dev-token}"

curl -X POST localhost:8000/tasks \
  -H 'Content-Type: application/json' \
  -H "$AUTH" \
  -d '{"objective": "Fix the flaky test in auth", "repository_url": "https://github.com/org/repo.git", "fork_url": "https://github.com/you/org-fork.git"}'
# -> 201 {"id": "...", "status": "CREATED", ...}  (advance_task job enqueued)

curl localhost:8000/tasks                # list        (open — no token needed)
curl localhost:8000/tasks/{id}           # fetch one — watch status walk to AWAITING_APPROVAL
curl localhost:8000/tasks/{id}/events    # ordered execution-event trail
curl localhost:8000/tasks/{id}/trace     # the human-readable trace viewer
curl -X POST localhost:8000/tasks/{id}/cancel -H "$AUTH"   # -> FAILED (user_cancelled); 409 on terminal tasks
curl -X POST localhost:8000/tasks/{id}/approve -H "$AUTH"  # human checks out the draft PR, then approves -> COMPLETED
curl -X POST localhost:8000/tasks/{id}/reject -H "$AUTH"   # human rejects -> FAILED (deliberate stop)
curl localhost:8000/health               # {"status": "ok"}   (open)
```

## Authentication

The mutating API routes (`POST /tasks`, cancel, approve, reject) are gated by a
**single shared-secret bearer token** (`FORGEMIND_API_TOKEN`, sent as
`Authorization: Bearer <token>`), compared in constant time
(`secrets.compare_digest`), with the authenticated identity (`token-holder`)
recorded on the audit trail for every approve/reject/cancel. Read routes
(`GET /tasks`, `GET /tasks/{id}`, `GET /tasks/{id}/events`,
`GET /tasks/{id}/trace`) and `/health` stay open so a human can watch a task
without the token. Unset in development/test the token falls back to
`forgemind-dev-token`; **production fails closed** — without a
`FORGEMIND_API_TOKEN`, the API refuses to start.

## Trace viewer

`GET /tasks/{id}/trace` renders a task's full execution history as a
human-readable HTML timeline: the plan DAG, every agent transition, each tool
call, test runs, review and security verdicts, the draft PR, and the human
approval decision. Away from raw JSON, you get the task header and status, a
prominent PR link at the approval checkpoint, a failure-reason banner on
FAILED/ESCALATED tasks, and an auto-refresh that stops once the task is
terminal. It is read-only — editing or approving stays API-only on the
authenticated routes — so, like the other read routes, it is intentionally
open.

## Testing

```bash
pytest                   # hermetic suite (SQLite, real local git repos, stubbed LLM)
                         # — no services required
pytest tests_e2e/        # end-to-end (needs Postgres + Redis up) — real worker
                         # pipeline, crash recovery, concurrency, GitHub PR flow
```

Note: `tests_e2e/` spawns its OWN worker subprocesses, so stop the compose
worker first (`docker compose stop worker`) — an extra always-on worker races
the test workers (both sweep the same tasks). This is a test-harness
constraint, not a product issue.

`python scripts/reproduce_flake.py` exists as a diagnostic tool: it repeats the
hermetic suite (or a forced-order mock-LLM file list) until the historic
order-dependent flake reproduces, so any recurrence is caught with a traceback
instead of chased blindly.

## Milestone status

Feature-complete through the single-operator milestone — every phase below is
implemented, tested, and wired into the running pipeline:

- **Phase 1–4** — foundation, enforced state machine + arq worker, tool
  registry/policy engine, git runtime (clone-once cache, per-task worktrees,
  traversal-safe file access).
- **Phase 5–6** — LLM provider abstraction + Planning Agent; bounded read-only
  Research Agent with grounding cross-check.
- **Phase 7** — Developer Agent: write-capable loop, exactly-one-commit,
  cross-checked summary.
- **Phase 8** — Test + Debugger Agents: deterministic pass/fail branching and
  flakiness-aware debugging with bounded replans.
- **Phase 9** — Reviewer + Security Agents: independent, blind verdicts plus a
  code-level staleness check.
- **Phase 10** — GitHub Agent + PR runtime: fork draft PR with a human approval
  gate; no merge primitive anywhere.
- **Phase 10.5** — bearer-token auth on all mutating routes.
- **Phase 11** — the read-only trace viewer.

## Security posture

- Secrets come from `.env` only; connection strings are logged redacted.
- Startup fails fast if the DB is unreachable — no silent hangs.
- The worker uses the same env-driven config as the API — no new secret surface.
- Shell execution is confined to ONE tool (`shell.run_test`) running the
  repository's allowlist-validated test command as an argument list (never
  `shell=True`) with a hard timeout — no agent-input path into the command; file
  writes stay inside the task's worktree (traversal-rejected).
- External network is confined to the GitHub REST API and a push to the
  operator's FORK — `repositories.url` is for reads only, the push/PR target
  must come from `fork_url` (fail-closed), and `GITHUB_TOKEN` is embedded
  per-invocation and redacted from logs.

## Project layout

```
app/
  agents/            planner · researcher · developer · tester · debugger ·
                     reviewer · security · github — typed loops, fixed capability sets
  api/routes/        tasks (JSON API) · trace (HTML viewer)
  capabilities/      capability value objects + per-agent assignment
  execution/         the tool pipeline: validate → capability → policy → execute → audit
  git/               subprocess runner (arg lists, fixed identity) + per-task worktrees
  llm/               provider abstraction (OpenRouter / deterministic stub) + parse/validate
  models/            Section-G SQLAlchemy schema
  policies/          deterministic policy engine (fail-closed)
  repository/        clone-once cache + traversal-safe read/list/search
  runtime/           state machine (pure logic) · task lifecycle · trace assembly
  schemas/           Pydantic request/response schemas
  templates/         trace viewer templates (Jinja2, vanilla CSS)
  worker/            arq entrypoint · advance_task job · startup sweep (crash recovery)
tests/               hermetic suite (SQLite + git + stubbed LLM)
tests_e2e/           end-to-end suite (Postgres + Redis + worker subprocesses)
```

## Out of scope / future work

Deliberate, documented scope for today's single-operator milestone — the
things a multi-user build would add first:

- **Multi-user accounts** and per-task granular permissions (today "who" is the
  bearer-token holder).
- **OAuth / GitHub-login approval** and token rotation/expiry.
- **Merge automation** — ForgeMind opens the draft PR and waits; merging is
  intentionally a manual GitHub action.