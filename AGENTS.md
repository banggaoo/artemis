# AGENTS.md

## Commits

- Plain commit messages only: no `Generated with <tool>` footers and no
  `Co-Authored-By` trailers. The commit author is the repo owner; tool
  attribution does not belong in history.
- Prefer `-m` with single quotes (or a quoted heredoc) over unquoted
  `$(cat <<EOF ...)` blocks — unquoted heredocs let the shell eat
  backtick-quoted commands in the message body.

## Build & Test

- Python >= 3.12, managed with `uv`. The project venv lives at `.venv/`;
  run tools directly (`.venv/bin/python -m pytest ...`) or via `uv run`.
- `make test` — deterministic unit suite (no device, credentials, or
  private services needed). Testpaths: `tests/unit`, `tests/tools`,
  `packages/artemis-client/tests` (see `[tool.pytest.ini_options]`).
- `make lint` — `ruff format --check` + `ruff check`. Run on changed
  files before committing.
- `make test-integration` / `make test-device` — need configured model
  credentials and an Android device respectively.
- The full suite is NOT green on a bare checkout: expect failures that
  require `GOOGLE_API_KEY`/`GEMINI_API_KEY`, provider mocks, local config
  values, and tmp-path fixtures. Verify work with targeted tests instead:
  e.g. `pytest -q tests/unit/agents/flash tests/unit/config
  tests/unit/core` for Flash/local-model changes.

## Local Model Setup (Gemma 4 E4B)

- Local OpenAI-compatible endpoint convention: `http://127.0.0.1:8080/v1`,
  model alias `gemma4-e4b`, served via `artemis model
  list/pull/serve/status/stop` (MLX on Apple Silicon).
- Opt-in config: `config/artemis.gemma4.jsonc`. `config/artemis.jsonc` is
  the machine-local deployment config — see "Local-only files" below.
- `thinking_level`/`include_thoughts` in a config are Gemini-provider
  kwargs only. The `custom` provider path (`artemis/llm/router.py`) sends
  only `model`, `temperature`, `max_tokens`, `timeout` (+`reasoning_effort`
  if set) — thinking fields are silently dropped for local models.
- Gemma-family local endpoints can emit malformed tool-call envelopes
  (`call:name{...}`); the Flash runner has relaxed parsing
  (`_GEMMA_CALL_RE`, `_loads_relaxed_args`) — keep new parsing tolerant.

## PR Scope Discipline

- Keep PRs single-topic. The local-model PR is "Artemis Flash improvement
  using the on-device model" — general Artemis bug fixes and improvements
  do not belong in it, even when they are real bugs.
- For a legitimate but unrelated issue: file a GitHub issue on
  google/artemis, preserve the patch/tests outside the PR, and handle it
  in a separate PR. Example: compact-XML `checked`/`enabled` omission is
  issue #187, deliberately excluded from PR #176.
- Shared components (formatter, planner, operator prompts, Android
  driver, graph) are not Flash scope — changes there need their own
  justification.

## Local-only files

- `config/artemis.jsonc` is machine-specific deployment config, marked
  `git update-index --assume-unchanged` so local edits are not committed.
  Do not overwrite it from HEAD or from staging operations; a recovery
  copy lives at `/tmp/artemis-pr176-scope-20261017/local-config-recovery.jsonc`.
- Other expected dirty/untracked files: `uv.lock`, `.angular/`,
  `AGENTS.md` (this file, until committed), `GEMMA_FINETUNE.txt`.

## Flash Runner Architecture (`artemis/agents/flash/runner.py`)

- Hybrid loop (opt-in): `_hybrid_plan` (VLM milestones) → per-turn
  `_hybrid_decide` (constrained action menu, `_HYBRID_MAX_MENU_ELEMENTS`,
  meta-actions) → fallback to the full Gemma tool-call path on
  low-confidence, no goal target, or repeat streak.
- Repeat machinery: `_element_signature` (strict set equality —
  systemui/OCR volatile sources excluded, so a single appended digit on a
  dense screen counts as change; `_signatures_similar` remains Jaccard
  for ping-pong detection only), dead-verbatim-call suppression, A-B-A
  ping-pong detection, streak escalation → honest failure. Before a
  'dead' verdict, `_screens_differ` cross-checks the pre/post screenshots
  — a frozen accessibility tree can lie while pixels move (verified:
  helper tree frozen across a whole dial run); then one re-observation
  covers capture lag.
- Index-target feedback: `target_description` on an index click is
  verified against the resolved element (`_description_mismatch_note`).
  On disagreement the result names the real element — and when a listed
  element confidently matches the description, the tap is refused and
  the right index is handed back (`_matching_element_hint`); a
  known-wrong tap on a dense pad corrupts state, correction beats
  cleanup. Without a confident alternative the tap proceeds with the
  note appended.
- Coordinate-target feedback: `[x, y]` taps are hit-tested against the
  indexed element list (`_coordinate_target_note` +
  `find_element_at_point`). A leaf hit is checked against the
  description like an index resolution; a hit on a layout container or
  on nothing listed is refused when the description identifies one
  element (`_confident_element_hint` — widget words like "switch"
  filter same-labelled nodes by class), else noted as a likely dead
  tap. Container hits that agree with the description stay untouched —
  dial-pad keys are clickable FrameLayouts. Verified live: a tap aimed
  at a DeskClock switch landed on the expanded CardView's dead space
  five times; the check refuses it and names the switch's index.
- Unsatisfied invariants inject a `[GOAL CHECK]` notice with
  `describe_failures` evidence (digit-level prefix diff for text
  attributes) before the next action — the model narrates its own plan
  over the field's real contents without this.
- Declared conditions: `agent.flash.goal_invariants` (config) →
  `execution_setup.goal_invariants` + `state_predicates.py` →
  `satisfied`/`unsatisfied`/`unknown` verdicts. All-satisfied suppresses
  further actions and can close the run; unsatisfied rejects 'completed'
  reports; unknown defers to the checker. Both the pre-action check and
  the completion gate re-observe fresh (`_fresh_invariant_elements`) —
  post-action captures race UI transitions, so lagged elements must
  never decide suppression or release.
- Completion: `_verify_completion` evaluates invariants first, then the
  Pro `run_final_check` when `final_check_enabled` — fail-open on
  checker errors, findings feed back on rejection
  (`_completion_rejections`).

## Evidence From Live Tests (Oct 2026)

Measured on the emulator with real local Gemma inference — treat as the
current empirical baseline before proposing model additions:

- Best arm: Gemma + deterministic state checks (3/4 verified, 4/4
  correct final state). Kev/Laya-style decision models did not improve
  completion and made harmful vetoes.
- Gemma's failures were missing-evidence failures (invisible `checked`
  state, label-vs-control grounding), not shallow-reasoning failures.
- Deployed E4B emits inline prose reasoning, not `<think>` blocks;
  `operator_native_thinking` is always null on the local path.
- Decision models stay optional/shadow until fine-tuned on Artemis
  traces; no cloud decision service (Jev) — on-device only.
- Oct 17: goal_invariants verified live — `already_off` closed in 2
  turns (action suppressed, completion released on `checked=false`),
  `dark_off` toggled once then released deterministically without the
  model checker. Both independently confirmed by ADB oracle.
- Oct 17: vision-token sweep (280/560/1120 × 3 tasks): no accuracy
  gradient at this sample size, but vt1120 `dark_off` exposed a real
  harness bug — post-action XML capture raced the theme transition, so
  the pre-action check evaluated lagged `checked` and never suppressed;
  four taps ping-ponged the switch. Fixed by re-observing fresh before
  every invariant decision (pre-action and completion gate). Post-fix
  vt1120 runs closed in 2 turns. Model addressing varies run to run —
  index clicks and normalized-0-1000 coordinate clicks both resolve
  correctly; latency is turn-count-bound, not token-bound.
- Oct 17: complex-UI milestone — `dial_5550155` on the Android dial pad
  (~50 elements, 3 lookalike entries per key, list that grows as
  suggestions appear) completed and verified: `digits` field reached
  `555-0155`, the text invariant released completion deterministically,
  and a satisfied-verdict suppression stopped post-goal actions.
  Harness: `/tmp/artemis-live-flash-20261016/run_invariant_task.py`
  (task `dial_5550155`, `hierarchy_backend: uiautomator` — the a11y
  helper serves a byte-identical frozen tree for dial-pad text updates;
  UIAutomator2 stays fresh but makes external `uiautomator dump` exit
  137, so the final oracle falls back to the trace DB's last element
  snapshot). Earlier same-task runs showed: frozen helper tree → false
  dead-verdicts; Jaccard signature → real digit taps read as unchanged;
  index drift + narrated-vs-observed divergence → corrupted fields.
  E4B-4B's residual limit is plan-level (digit-sequence transcription,
  never choosing backspace), not grounding once the guards hold.
- Oct 18: expanded-card dead zones — DeskClock alarm toggles aimed by
  coordinates landed on the CardView body, not the Switch inside it
  (verified by trace: 5 taps, `checked` stayed false). Coordinate taps
  are now hit-tested and refused-with-hint like index targets. Also
  observed: a post-run shell `uiautomator dump` can serve stale node
  attributes — it read `checked=false` while the same-moment
  screenshot showed the switch on and the in-run captures had been
  stable true for ~2 min; DB-backed widgets additionally revert if the
  app is force-stopped before their async write flushes. Final-state
  oracles should read the run's last live capture (dumps only as
  fallback) and never follow a `force-stop`.
- Oct 18: 102-run complex-UI campaign (17 task families × 6 seeds,
  resumable driver, trace-primary oracle with dump-vote
  corroboration): **54/102 = 52.9% verified end-to-end** by local
  Gemma 4 E4B. Per-family rates: alarm toggles 6/6 both directions,
  dial_911 5/6, dark theme on/off 5/6 each, stopwatch 5/6,
  appinfo_chrome 5/6, nav_battery 4/6, settings_search 4/6,
  battery_pct_on 2/6 (system-setting oracle), sms_5550155 2/6,
  7-digit dials 1/6 each, nav_sound/nav_storage 1/6, sms_1234567 0/6.
  Failure mix: timeout 25, claimed-without-state 14 (all rejected by
  the oracle — e.g. sms_1234567 claimed done 6/6 while never picking
  the "Send to" suggestion), dead-tap escalations 7, runner_failed 2.
  `oracle_disagreement` fired 29× — post-run dumps genuinely disagree
  with live trace state often enough that a single-source oracle
  would mis-score runs. Harness: `tests/integration/flash_live/`
  (env-parameterized; `python spec.py` once, then `run_batch.py`).
- Oct 18: APC (automatic prefix caching) on the local `mlx_vlm.server`
  — `APC_ENABLED=1`, `APC_EXACT_CACHE_ENTRIES=8`. Stock APC salts every
  block with the request's media hash, so a fresh screenshot each turn
  made reuse exactly 0. Vendored patch adds a media-free companion salt
  (`image_hash=0`) with checkpoints stored at (a) the last Gemma
  turn-start (`<|turn>` id 105) before the first media span and (b) the
  first-media boundary itself; lookups fall back to the media-free salt
  bounded at first-media-start (causally valid: pre-image text KV
  cannot see the image). To make the transcript prefix-stable, the
  observation's screenshot is now the LAST content block — pruning then
  removes a trailing span instead of a mid-message one. Measured on
  real turns: `cached_tokens≈5231/~8000` (~65% reuse; the cap is the
  strip of the second-newest observation each turn), prefill
  16–19s→6–10s, turn cadence ~40–45s→~12–15s. `nav_battery` and a live
  dialer run verified correct on the patched server. Diagnostics:
  `MLXVLM_APC_DEBUG=1` (server) and `ARTEMIS_DEBUG_PROMPTS=<path>`
  (per-turn per-block prompt digests for divergence localization).
  Deeper reuse needs media-aware checkpoints over retained images —
  deferred; the strip-point cap is structural while old observations
  are rewritten.
- Oct 18: APC + 900s-budget re-run (17 families × 2 seeds, same
  suite_spec/oracles): **23/34 = 67.6% verified** vs the 52.9%
  pre-APC baseline. Zero timeouts — with 900s budgets the wall clock
  can never fire before `max_turns=18`; wandering runs now end as
  honest `runner_failed` status (model self-reports after exhausting
  productive actions) instead of being cancelled mid-flight. Per-family
  at 2/2: alarms both directions, appinfo, battery_pct, both dark
  themes, dial_911, nav_battery, settings_search, stopwatch. Residual
  failures concentrate exactly where the plan-level limit already
  showed: 7-digit dials 0/6 (`55500155`/`555055` accumulation),
  sms_1234567 0/2 (claimed while "Send to" suggestion untapped),
  nav_sound/nav_storage 1/2 (post-launch confusion).
