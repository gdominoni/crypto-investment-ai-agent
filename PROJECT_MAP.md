# Project Map

File-by-file guide to what each part of this codebase does. Companion to [README.md](README.md) (what the system does) and [`docs/case_study/methodology-decisions.md`](docs/case_study/methodology-decisions.md) (why each number and design choice is what it is) — this file answers *where*, those two answer *what* and *why*.

---

## `candidates/` — statistical engine

- **`methodology.py`** — the acceptance pipeline every candidate runs through, static or dynamic, production or replay:
  - `build_events()` — causality-safe entry, always the bar after a trigger.
  - `pattern_significance()` — **the acceptance gate**: per-fold horizon selection, then a block-bootstrap significance test.
  - `concentration_check()` — flags a result carried by one coin or one year.
  - `classify_status()` — combines the two into `accepted` / `watch` / `rejected` / `insufficient_data`.
  - `explain_non_acceptance()` — the specific reason a candidate isn't `accepted`.
  - `sortino_ratio()`, `compute_anchors()`, `walk_forward()` — informational risk/TP-SL stats, don't gate acceptance.
  - `format_trigger_summary()` / `format_candidate_details()` — power `/summary` and `/details`.
  - `prune_recommendation()` — offline keep/drop decision (no LLM call).
  - `benjamini_hochberg()` / `apply_fdr_demotion()` — family-level multiplicity control.
  - `SIGNIFICANCE_ALPHA` (0.10), `MAX_GROUP_SHARE` (0.6) — the two acceptance thresholds.
- **`definitions.py`** — the 3 static, non-adaptive control-arm triggers (C1 funding-rate, C2 post-macro-release, C6 efficiency-ratio) plus their Telegram descriptions. `compute_triggers(df, funding, scale)` — `scale=24` reinterprets daily windows as hourly for the live scan.
- **`data_loading.py`** — loads OHLCV + funding-rate series from `data/`. `zscore()` — the rolling z-score used by every stationary indicator.
- **`macro_calendar.py`** — FOMC/CPI release calendar.
- **`macro_vintage.py`** — point-in-time-correct macro lookups (`surprise_series`, `latest_release_with_prior`) — never a later-revised value.
- **`run_battery.py`** — weekly orchestrator: runs every candidate through `methodology.py`, writes `execution/live_battery_state.json` (what "currently accepted" is read from).
- **`status_history.py`** — tracks `candidates_due_for_prune_decision()` (2-year keep/drop safety net) and `candidates_due_for_milestone()` (the CONFIRMED checkpoint, `MILESTONE_N = 20`).

## `data/` — historical & market data

- OHLCV (daily/hourly) for 7 coins (BTC, ETH, BNB, XRP, DOGE, ADA, LTC) from Binance, funding-rate history, and macro series (CPI, Fed funds, jobless claims) from FRED. Kept current by `data_ingestion/market_data/binance_fetcher.py`, not a frozen snapshot.

## `data_ingestion/market_data/` — live data refresh

- **`binance_fetcher.py`** — `update_ohlcv()` / `update_funding()` pull new candles from Binance and append to `data/`. Paginates from a real anchor date rather than a single bounded call. `update_all()` isolates each coin's fetch.

## `data_ingestion/news_sentiment/` — news ingestion

- **`cryptocompare_fetcher.py`** — fetches headlines. Not wired into anything since the Haiku headline path was removed — kept as the starting point for a future sentiment effort, see [methodology-decisions.md](docs/case_study/methodology-decisions.md#haiku-and-the-news-headline-path-removed-because-a-measurement-said-they-couldnt-produce-evidence).

## `llm_pipeline/` — LLM judgment layer

- **`haiku_sonnet_pipeline.py`** — Sonnet's one live escalation role.
  - `sonnet_compression_response()` — Sonnet's judgment on a confirmed compression exit.
  - `prune_recommendation()`'s model-free replacement lives in `candidates/methodology.py`; nothing here calls a model for keep/drop or milestone decisions.
  - `format_spec_clauses()` / `format_compression_message()` — plain-English rendering of a proposed condition.
  - `PROPOSAL_KEYBOARD_TEMPLATE` — Test It / Don't Test It buttons.
  - `run_compression_scan()` — isolates every episode in its own try/except.
  - The Haiku headline-screening path (`haiku_scout`, `sonnet_strategist`, etc.) is deleted — see [methodology-decisions.md](docs/case_study/methodology-decisions.md#haiku-and-the-news-headline-path-removed-because-a-measurement-said-they-couldnt-produce-evidence).
- **`novel_condition_tester.py`** — the proposal grammar.
  - `SUPPORTED_INDICATORS` — the full indicator registry. `proposable_indicators()` — the subset (13) an LLM may actually use.
  - `NON_PROPOSABLE_INDICATORS` — banned from proposals: `shock_zscore`, `vol_compression_zscore`, `is_macro_day`, `daily_range_pct` (each for its own reason, see methodology-decisions.md).
  - `MAX_PROPOSABLE_CLAUSES = 2`, `MAX_PROPOSALS_PER_CALL = 2`.
  - `filter_redundant_proposals()` — drops a second proposal that fires on the same days as the first (behavioral overlap, not shared clauses).
  - `is_testable()` — applies both the raw-occurrence floor (`MIN_HISTORICAL_OCCURRENCES = 120`) and the independent-episode floor (`MIN_HISTORICAL_EPISODES = 40`, via `episode_count()`).
  - `relax_to_testable()` — loosens a too-rare condition toward its neutral point (`RELAXATION_NEUTRAL`), never past it.
  - `ConditionSpec` — declares `clauses`, `direction`, `coins`, `outcome` (`raw`/`market_relative`), `prior_weight` up front, before any test runs.
  - `Clause.within_days` — makes a sequenced hypothesis ("crash, THEN news") expressible.
  - `test_novel_condition()` — runs one proposed condition through the same `methodology.py` pipeline as the static battery.
  - `clause_to_dict()` / `clause_from_dict()`, `spec_to_dict()` / `spec_from_dict()` — the only serializer pair for each.
  - `DAILY_NATIVE_INDICATORS` — indicators that must be computed on the daily frame even during an hourly scan (not distributionally comparable at `scale=24`).
- **`compression_detector.py`** — the trigger. `current_compression_exit()` / `scan_for_compression_exits()` call `candidates/methodology.py::compression_exit()` — one definition, not two. Ledger keyed on exit date (`already_escalated`) stops re-asking the same question hourly.
- **`context_builder.py`** — `build_context_summary()` (accepted + already-tested candidate names, uncapped — Sonnet needs the full list to avoid duplicate proposals). `build_live_test_summary(top_n=15)` — capped by `|mean_return|`, see [Cost Optimization](#cost-optimization-what-actually-gets-sent-to-anthropic) below.
- **`dynamic_candidates.py`** — persistent registry of live-discovered conditions, re-tested weekly alongside the static ones.
- **`pending_tests.py`** — FIFO queue of proposals, each with its own short id on its Test It / Don't Test It buttons. Entries expire after 48h.

## `replay/` — historical replay

A day-by-day walker through real 2017-2026 history, isolated from production's own state. Same statistical methodology and live-test model as production. **Completed run:** 2017-08-26 → 2026-09-05, 159 tracked candidates, 23,495 live tests opened. Full result and caveats: [methodology-decisions.md](docs/case_study/methodology-decisions.md#the-real-result-2017-08-26-to-2026-09-05-nine-years-day-by-day).

- **`battery.py`** — `run_replay_battery(as_of)` — mirrors `candidates/run_battery.py`, sourced from as-of data only.
- **`judgment.py`** — `judge_event()` (Sonnet's judgment for one compression exit), `format_compression_event()` / `format_compression_report()` (the two renderings — one for Sonnet, one for a human), `answer_market_question()` (free-text Q&A).
- **`engine.py`** — `advance(chunk_days)`, the day-by-day walker. See the [code navigation diagram](docs/case_study/methodology-decisions.md#2-one-simulated-day-replayenginepyadvance) for its full per-day flow. Halts and waits for a human whenever a novel condition is proposed.
- **`state.py`** — replay's isolated state store: `checkpoint.json`, `battery_status.json`, `trade_log.json`, `dynamic_candidates.json`, `pending_test.json`, `horizons.json`, `parked_proposals.json`, `confirmation_priors.json`.
- **`status_history.py`** — same two mechanisms as `candidates/status_history.py`, keyed off the simulated date.
- **`post_replay_hyperopt.py`** — the Freqtrade cross-check, run once after a replay finishes (not inline — see Cost Optimization Part 3).
- **`time_sandbox.py`** — `daily_as_of()` / `funding_as_of()`, the mechanism that makes the replay causality-safe.
- **`orchestrator.py`** — `run_to_completion()`, the batch driver that repeatedly calls `advance()`.

## Cost Optimization — what actually gets sent to Anthropic

Pricing: Sonnet 5 $2 / $10 per million tokens (input/output), Haiku 4.5 $1 / $5. Every figure below is measured against this project's own real calls, not a generic estimate.

**Part 1 — building the replay (one-time cost, not a recurring expense):**
- Per-event judgment (`judge_event()`) — ~1,230 tokens/call, ~$0.01/call. Flat regardless of registry size: no history, no per-trade detail, just the one event plus current readings.
- Human Q&A (`answer_market_question()`) — capped at the top 15 rows by `|mean_return|` (`build_live_test_summary()`) and a compacted `insufficient_data` bucket. Bounded at ~2,700 tokens regardless of how many candidates exist.
- `/summary` and `/replay_summary` cost **$0** — local computation, no LLM call, no cap.
- A cheaper model (Haiku) was measured, not assumed, as a judge substitute — see [`forecast/model_comparison.py`](#forecast-offline-experiments) below. Verdict: no, it proposes a quarter of the usable hypotheses at the same cost per hypothesis.
- Every Sonnet call uses `max_tokens=4000` (a real, measured failure mode below 3000 — see methodology-decisions.md).

**Part 2 — running live, ongoing (Anthropic API only, excludes hosting):**
- Escalation call: ~1,400 tokens/call (~$0.01/call), grows slowly (name-only lists, not full detail).
- Estimated **$5-10/month** in moderate activity, likely under $20/month even in a volatile month.

**Part 3 — server/hosting (kept off the live host):**
- The live host only needs the hourly mechanical scan, the hourly compression scan, and the Telegram bot loop — all cheap, no meaningful CPU/memory.
- `execution/hyperopt_runner.py` runs **local-only**, never on the live host: it's the one real compute cost (minutes per candidate), its output is never time-sensitive (purely informational, see methodology-decisions.md), and only its output file (`execution/hyperopt_results.json`) needs to reach the live host.

## `execution/` — trade execution

This project never opens a funded position ([why](docs/case_study/methodology-decisions.md#live-testing-hold-for-the-horizon-no-tpsl)).

- **`live_testing.py`** — production's counterpart to `replay/engine.py`'s walker:
  - `_open_live_test()` / `_check_live_tests()` — hold for the horizon, resolve by measuring real forward return/MFE/MAE.
  - `_scan_mechanical_triggers()` — unattended, no LLM, hourly.
  - `find_backdated_entry()` — anchors a newly-discovered condition to the real hour it first became true.
  - `check_n50_milestones()` — production's CONFIRMED checkpoint.
  - `_check_consecutive_failures()` — fast, informational-only losing-streak alert, CONFIRMED candidates only.
  - `run_once()` — the scheduled-job entry point.
- **`live_test_state.py`** — persistent state for the above (`live_tests.json`, `horizons.json`).
- **`hyperopt_runner.py`** / **`freqtrade_bridge.py`** — the local-only Freqtrade cross-check. `python3 -m execution.hyperopt_runner` from the project root; writes `execution/hyperopt_results.json`.
- **`signal_store.py`** — `load_battery_state()`, read by `context_builder.py`. Still load-bearing despite its pre-live-testing name.

## `telegram/` — Telegram interface

- **`bot.py`** — the long-polling bot process (`run_bot()`).
  - `_send()` — splits any message over Telegram's 4,096-char limit (`_chunk_message()`), pins with `pin=True`.
  - Commands (none touch an LLM): `/summary`, `/details <name or id>`, `/replay_summary`, `/replay_details`, `/help`, `/start`.
  - `short_id()` / `resolve_candidate()` — 4-character ids for typing a candidate name on a phone.
  - Button callbacks: `handle_propose_callback()` (Test It / Don't Test It), `handle_prune_callback()` (Keep / Drop).
  - Free text falls through to `handle_natural_language()` — Sonnet, grounded only in real computed numbers.
  - Outbox: `drain_outbox()` / `flush_outbox()` — queues a message past a real Telegram rate limit instead of blocking.

## `scheduler/` — scheduling

- **`live_daemon.py`** — the single entry point for running this project live: `python3 -m scheduler.live_daemon`, one process, no external cron. Owns the Telegram poll loop and the hourly/weekly job timers (`live_daemon_state.json`). Each job runs isolated (`_run_isolated`) — one crashing alerts and retries next cycle, never takes the daemon down.
- **`weekly_revalidation.py`** — refreshes market data, re-runs the full battery, diffs every candidate's status, notifies on change, handles the 2-year keep/drop safety net. Any crash past the data refresh sends a Telegram alert before re-raising.

## `docs/case_study/` — documentation

- **`methodology-decisions.md`** — why each threshold and design choice is what it is, organized by topic (not a change log).
- **`how-the-replay-runs.md`** — step-by-step walk-through of one simulated day.
- **`assets/`** — screenshots and diagrams.

## `forecast/` — offline experiments

Answers questions about the SYSTEM, offline and free — no Anthropic calls, no market claims. Results are committed so the numbers in methodology-decisions.md are auditable.

- **`positive_control.py`** — plants a synthetic signal with known ground truth; confirms the pipeline finds it and stays silent on pure noise.
- **`grammar_sweep.py`** — enumerates the full proposal grammar (672 conditions) and tests each — an upper bound on what the replay could find.
- **`control_sweep.py`** — the gate autopsy: which gate actually kills a known-good condition (significance, sample size, concentration, or MFE/MAE).
- **`market_relative.py`** / **`coin_specific_test.py`** — whether market-relative outcome measurement helps (conditional — see methodology-decisions.md).
- **`sentiment_power.py`** — the go/no-go measurement behind deleting the Haiku headline path.
- **`model_comparison.py`** — Haiku vs. Sonnet as judge, scored on proposal quality, not just agreement.
- **`*.json`** — committed result sets; each sweep resumes from where it left off.

## `tests/` — tests

- **`test_methodology.py`** — causality safety, anchor/barrier math, Sortino, concentration detection, status classification, plus a regression test for every defect the statistical audit found (see methodology-decisions.md).
- **`test_status_history.py`** — keep/drop and checkpoint logic, against an isolated temp file.
- **`test_novel_condition_tester.py`** — the indicator/operator/direction whitelist, plain-English rendering.
- **`test_run_battery.py`** — one broken candidate must not cost the others their result.
- **`test_message_volume.py`** — no per-live-test messages, digest ranked by progress not outcome, outbox drains on every exit path.
- **`test_python_compatibility.py`** — every f-string is valid on Python 3.11, not just the dev machine's version.
- 253 tests total, run automatically on every push via `.github/workflows/tests.yml`.

## Partial failures & crashes

- **`candidates/run_battery.py`** — each candidate runs in its own try/except; a failure is `status = "error"`, retried next run, and doesn't cost any other candidate its result.
- **`data_ingestion/market_data/binance_fetcher.py`** — same isolation per coin.
- **`llm_pipeline/haiku_sonnet_pipeline.py::run_compression_scan()`** — each compression exit is isolated.
- **`scheduler/weekly_revalidation.py`** — any exception past the data refresh sends a Telegram alert before re-raising.
- **`execution/hyperopt_runner.py`** — a failed run is persisted as `{"status": "failed", ...}`, distinguishable from "never attempted."
- **`llm_pipeline/pending_tests.py`** — proposals expire after 48h rather than sitting forever unanswered.
- **`scheduler/live_daemon.py`** / **`telegram/bot.py::run_bot()`** — the polling loop backs off and retries on a network blip instead of crashing the process.

## Root files

- **`requirements.txt`** — Python dependencies.
- **`.env.example`** — required environment variables (names only, never real values).
- **`.gitignore`** — keeps secrets and runtime state (including `replay/state/`) out of version control.
