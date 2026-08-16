# ForgeMind

Autonomous software-engineering agent: **GitHub issue → investigate → implement → test → review → PR**.

Current milestone scope:

- **Phase 1 — foundation layer**: project skeleton, config, database schema, minimal FastAPI Task API.
- **Phase 2 — task runtime**: an enforced Section-D state machine, an `execution_events` trail,
  and an arq + Redis worker that drives tasks `CREATED → PLANNING → … → COMPLETED` entirely off
  the queue.
- **Phase 3 — tool runtime**: typed tool registry, capability model, deterministic policy engine,
  and the full tool pipeline (validate → capability → policy → execute → audit) with three
  harmless example tools.
- **Phase 4 — git/repository runtime**: real filesystem access. Repository discovery (clone-once
  cache), per-task git worktrees (never touching `main`), worktree-scoped file read/list/search
  with airtight path-traversal defense, and `repository.*` / `git.*` tools wired through the
  pipeline (capability-gated, audited, fixed commit identity). No `git.push`, no `shell.*`.
- **Phase 5 — LLM provider + Planning Agent**: the first real LLM call and the first agent whose
  output drives the state machine. `llm/` abstracts providers (OpenRouter now, pluggable later)
  with strict `structured_output` (never a partially-valid object); the Planning Agent turns a
  task objective into a schema-validated plan DAG, persists it, and only then does the worker
  move PLANNING → RESEARCHING. Malformed output retries exactly once; task text is wrapped as
  DATA (prompt-injection defense); the planner has zero tool capabilities.
- **Phase 6 — Research Agent**: the first multi-turn tool-use agent. A bounded loop where the
  LLM proposes `repository.*`/`git.*` tool calls, the Phase-3 pipeline executes them under the
  researcher's fixed READ-ONLY capability set (`repo.read`, `git.read`, `github.read`), and the
  loop ends with a schema-validated `ResearchArtifact` whose file claims are cross-checked
  against what the loop actually observed. RESEARCHING → IMPLEMENTING fires only after the
  artifact is persisted. Read-only is enforced structurally (a write proposal is DENIED and
  audited, never "just tried"); file content and git output are DATA, not instructions.

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
pytest              # hermetic suite (SQLite, no services) — state machine, lifecycle, API,
                    # policy engine, path traversal, git runtime, planner, research agent
pytest tests_e2e/   # end-to-end (needs Postgres + Redis up) — real worker pipeline, cancel,
                    # crash recovery (kill/restart), two-worker concurrency, research loop
```

Note: `tests_e2e/` spawns its OWN worker subprocesses on the same Postgres/Redis, so stop the
compose worker first (`docker compose stop worker`) — an extra always-on worker races the test
workers (both sweep the same tasks). This is a test-harness constraint, not a product issue.

## Layout

```
app/
  agents/
    base.py                 Agent ABC + shared structured_output_with_retries
    planner/                PlanningAgent: prompt (data-not-instructions), Plan schema +
                            DAG validation, retry-once, persistence
    researcher/             ResearchAgent: bounded tool-use loop (read-only), ResearchArtifact
                            schema + grounding cross-check, forced synthesis on budget exhaustion
  api/routes/tasks.py       Task API: create/list/get/cancel/events
  capabilities/             Capability value objects + per-agent assignment (Section H)
  database/                 engine/session + Alembic migrations
  execution/tool_pipeline.py  validate -> capability -> policy -> execute -> audit
  git/
    runner.py               git subprocess runner (arg lists only, fixed identity, no prompts)
    operations.py           status/diff/log/create_branch/commit on a worktree
    worktree_manager.py     per-task worktrees (create/discard/path_for) — the only branch creator
  llm/
    provider.py             LLMProvider ABC + parse/validate (strict structured_output)
    openrouter.py           OpenAI-compatible provider (httpx, timeouts -> LLMTimeoutError)
    mock.py                 deterministic stub provider (tests / key-less dev)
    config.py               role -> model from env (LLM_MODEL_PLANNER, ...)
  models/                   SQLAlchemy models (Section-G schema) + ExecutionEvent + ToolCall
  policies/                 deterministic PolicyEngine + risk-default & explicit-deny rules
  repository/
    discovery.py            clone-once cache + default-branch/base-commit resolution
    file_access.py          worktree-scoped read/list/search — traversal-safe
  runtime/
    state_machine.py        Section-D legal-transition table — pure logic, no I/O
    task_lifecycle.py       transitions + events + stub driver + agent transitions
                            (advance_task_with_agents) with compare-and-swap re-check
  schemas/                  Pydantic request/response schemas
  tools/                    Tool ABC + registry + example tools + repository.* / git.* tools
  worker/
    queue.py                arq Redis settings + pool + enqueue helper
    jobs/advance_task.py    one job = one transition (PLANNING/ RESEARCHING run the real agents)
    worker.py               arq entrypoint + startup sweep (crash recovery)
  config.py                 env-driven settings (secrets never hardcoded/logged)
  logging.py                logging setup with URL redaction
  main.py                   FastAPI app + /health + fail-fast DB check
tests/                      hermetic suite (SQLite + real local git repos + stubbed LLM)
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

## LLM provider + Planning Agent (Phase 5)

- **`llm/` provider abstraction**: `structured_output` either returns a fully-valid object or
  raises `LLMMalformedOutputError` with the raw output attached — malformed JSON and
  valid-JSON-with-wrong-shape are both rejected. Models are per-role env vars
  (`LLM_MODEL_PLANNER`), never hardcoded. `FORGEMIND_MOCK_LLM=1` runs a deterministic stub
  provider (tests / key-less dev); a configured `OPENROUTER_API_KEY` always wins.
- **Planning Agent**: task objective in → schema-validated plan DAG out → persisted to
  `plans`/`plan_steps` (raw output in `plans.raw_llm_output`) → PLANNING → RESEARCHING fires
  only after that. DAG rules: unique ids, no dangling deps, no cycles, every implement step
  has a research ancestor.
- **Retry boundary**: transient errors (timeout, 429/5xx) retry with bounded backoff; the
  malformed/invalid retry happens exactly ONCE with a correction prompt; a second failure
  raises and the task goes FAILED (never ESCALATED), raw output preserved.
- **Prompt injection**: task objective/repo metadata sit in a `<reference_data>` block and the
  system prompt declares them DATA, not instructions. Tests feed injection-style objectives
  and prove the injected payload can never become a persisted plan.
- **Planner capabilities are structurally empty** — it cannot invoke any tool.

## Research Agent (Phase 6)

- **Bounded tool-use loop**: the LLM proposes ONE tool call per turn (`repository.search`,
  `read_file`, `list_files`, `git.status`/`diff`/`log`); the pipeline executes it under the
  researcher's fixed read-only capability set; results are fed back as `<observation>` DATA;
  repeat up to `MAX_RESEARCH_TOOL_CALLS` (default 10). A `{"final": true}` response ends the
  loop and the runtime asks for the `ResearchArtifact`.
- **Read-only is structural**: the capability set is `repo.read`/`git.read`/`github.read` — a
  write proposal (e.g. `git.commit`) is DENIED by the pipeline and audited, and the loop
  survives (the denial becomes an observation).
- **Grounding cross-check**: the artifact's `relevant_files`/`relevant_tests` must have been
  observed during the loop (read/searched/listed). Fabricated claims are rejected once with a
  correction prompt; if the retry still lies, the artifact is accepted WITH a loud
  `artifact.files_unverified` audit entry (a wrong file in a hypothesis is recoverable
  downstream; failing the whole task over it is not).
- **Budget exhaustion**: an LLM that never says `final` is hard-stopped at the budget, audited
  (`research.budget_exhausted`), and forced into synthesis — never an infinite loop.
- **Prompt injection**: file contents and git output are untrusted DATA like the issue text;
  an injection inside a file cannot fabricate grounding (the cross-check rejects it).
- **Concurrency-safe worktree/clone**: two workers racing the same task (research commits
  mid-transaction, releasing the row lock) are handled by compare-and-swap transitions and
  idempotent clone/worktree creation — one transition per state, no spurious failures.

## Git/repository runtime (Phase 4)

- **One clone per repo**, cached at `repositories.local_clone_path` (`--no-checkout`, so no
  default-branch working tree exists on disk at all). Per-task worktrees via
  `git worktree add -b agent/task-{id}` from the default-branch HEAD — the only place a branch
  is created; nothing ever commits to or checks out `main`.
- **Server-side path resolution**: every file/git tool takes a `worktree_id`; paths resolve
  against the worktree root and anything escaping it (`../`, absolute, symlink escape) raises
  `PathTraversalError` (a `SecurityError`) before any read. Traversal is logged as a
  security-relevant event and surfaces as a FAILED tool call.
- **Commits** stage all changes (`git add -A`) and refuse empty trees, with a fixed identity
  (`ForgeMind Agent <agent@forgemind.local>`) — agent input can never set authorship.
- **Discard** (`git worktree remove --force` + branch delete) enables Section J's
  "discard and recreate from base_commit" recovery path — recreated worktrees start byte-
  identical to the original.
- `git.*`/`repository.*` tools are capability-gated (`repo.read`, `git.read`, `git.write`),
  risk-tiered, and fully audited by the Phase 3 pipeline.

## State machine (architecture doc section D)

Transitions are enforced by a deterministic lookup table (`runtime/state_machine.py`),
never by asking the LLM "are we done?" Illegal transitions raise `IllegalTransitionError`
and are logged — `tasks.status` is never silently updated. Every applied transition writes an
`execution_events` row in the same transaction, so a crash mid-run always leaves the task at
its last persisted status; the worker's startup sweep re-enqueues it from there.

## Schema

Tables (architecture doc section G, milestone scope): `tasks`, `plans` (incl.
`raw_llm_output`), `plan_steps`, `task_steps`, `capabilities`, `policies`, `audit_logs`,
`repositories` (incl. `local_clone_path`), `worktrees`, `execution_events`, `tool_calls`,
`research_artifacts`. Remaining Section-G tables arrive with the phases that use them
(agents/eval). Tasks default to a bounded replan budget (`max_replans=3`) — Section D's
"budget exhausted → ESCALATED" is enforced even when the API client sets no budget.

## Security posture

- Secrets come from `.env` only; connection strings are logged redacted.
- Startup fails fast if the DB is unreachable — no silent hangs.
- The worker uses the same env-driven config as the API — no new secret surface.
- No shell execution, no file writes outside the repo, no external network calls yet.
