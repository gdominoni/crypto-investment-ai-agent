# Project Map

A file-by-file guide to what each part of this codebase does. Companion reference to [README.md](README.md) — use this to navigate straight to the code behind any specific claim in it.

---

## Backtesting Methodology (Phase 1) — `candidates/`

- **`methodology.py`** — Core statistical engine, shared by every candidate. Encodes every methodology rule as code: `build_events()` (causality-safe entry, always the bar after a trigger), `compute_anchors()` (duration-bucketed TP/SL levels from real historical MFE/MAE, never a flat barrier), `walk_forward()` (expanding-window, out-of-sample validation), `report()` (win rate, strict win rate, and Sortino ratio, always returned together), `concentration_check()` (flags a result carried by a single coin or a single year), `shock_zscore_series()` / `classify_regime()` (statistical isolation of extreme market shocks from ordinary conditions).
- **`definitions.py`** — The three trigger families under test (C1 funding-rate crowding, C2 post-macro-release reaction, C6 efficiency-ratio trend — C3/C4/C5 were part of the prior research's larger set but weren't carried forward), each a pure, independently testable function of price/volume/funding data.
- **`data_loading.py`** — Loads historical OHLCV and funding-rate series from `data/`.
- **`macro_calendar.py`** — FOMC/CPI release calendar, at calendar-day granularity (matching the daily-bar timeframe this project trades on).
- **`run_battery.py`** — Orchestrator: runs every static and dynamic candidate through `methodology.py`, assigns a live status (`validated` / `watch` / `rejected` / `error`), and writes the state file the execution engine reads from. Each candidate is isolated in its own try/except — one candidate's bad data or bug can't cost every other candidate its already-computed result (see "Partial Failures & Crashes" below).
- **`status_history.py`** — Tracks how long each candidate has been monitored, to trigger a keep-or-drop decision after years without validating.

## Historical & Market Data — `data/`

- OHLCV data (daily/hourly) for 7 coins (BTC, ETH, BNB, XRP, DOGE, ADA, LTC) from Binance spot, funding-rate history, and macro series (CPI, Fed funds rate, VIX, gold, S&P 500) from FRED and Yahoo Finance. Coverage: 2017–2019 through the present, depending on the coin's listing date — kept current by `data_ingestion/market_data/binance_fetcher.py` below, not a frozen one-time snapshot.

## Live Data Refresh — `data_ingestion/market_data/`

- **`binance_fetcher.py`** — Pulls new OHLCV candles and funding-rate entries from Binance's public API (no key needed) and appends them to the files in `data/`, incrementally. Runs before every weekly re-validation and every shock scan — this is what makes "re-validated against live data" and "real-time shock detection" actually true rather than aspirational. Each coin's fetch is isolated -- one coin's network failure doesn't block the other six.

## News Ingestion — `data_ingestion/news_sentiment/`

- **`cryptocompare_fetcher.py`** — Fetches headlines from the CryptoCompare News API. The only file in this folder used by the live pipeline.

## LLM Judgment Layer (Phase 2) — `llm_pipeline/`

- **`haiku_sonnet_pipeline.py`** — The adaptive layer. `haiku_scout()` (Claude Haiku screens every headline for asset/sentiment/magnitude/event type), `sonnet_strategist()` (Claude Sonnet's judgment, only for escalated headlines), `execute_routine_trade()` (fires a routine trade unattended — but only after independently re-checking the candidate is actually `validated` in the real state file, regardless of what the model claims), `sonnet_prune_advice()` (a lightweight, explicitly-unverified opinion feeding the keep/drop decision below). `run_once()` isolates each escalated headline in its own try/except -- one malformed model response can't silently stop the rest of that run's headlines from being processed.
- **`novel_condition_tester.py`** — Runs a human-approved novel condition through the same statistical pipeline as `methodology.py`, including the same shock-regime exclusion the static battery uses (except when the condition being tested IS the shock indicator itself). The model can only choose from a fixed whitelist of indicators (`SUPPORTED_INDICATORS`) — never arbitrary generated code.
- **`shock_detector.py`** — Real-time detection of extreme volatility, reusing the same statistical definition of "shock" as Phase 1.
- **`context_builder.py`** — Builds Sonnet's context from live state: real open positions and the last closed trade, read from the Freqtrade database, real candidate statuses — never a hand-maintained file.
- **`dynamic_candidates.py`** — Persistent registry of conditions discovered live, re-tested weekly alongside the static candidates.
- **`pending_tests.py`** — FIFO queue of novel-condition proposals awaiting a human's "test it" reply — a queue, not a single slot, so two proposals flagged close together can't silently overwrite each other. Entries expire after 48h so an unanswered proposal doesn't sit forever and get resolved against long-stale context.

## Cost Optimization — what actually gets sent to Anthropic

Pricing (verified against [platform.claude.com/docs/en/about-claude/pricing](https://platform.claude.com/docs/en/about-claude/pricing), not assumed): **Claude Sonnet 5** $2 / $10 per million tokens (input / output), **Claude Haiku 4.5** $1 / $5. Every number below is measured against this project's own real calls and real accumulated state, not a generic estimate — see `docs/case_study/methodology-decisions.md` for the incident that prompted this section (a Q&A call's context had grown to ~10,400 tokens by the time 96 candidates were tracked, discovered live while burning through this project's own Anthropic credit balance running the historical replay).

This project's LLM calls split into two kinds with deliberately different context budgets, because they answer two different questions:

**1. Automated per-event judgment** — `judge_event()` (`replay/judgment.py`), `sonnet_strategist()` / `sonnet_shock_response()` (`haiku_sonnet_pipeline.py`). Fires on every real macro release, volatility shock, or escalated headline — hundreds of times over any long observation window, so this is the actual cost driver. Its context answers one narrow question ("is *this* event worth escalating") and is built to stay **flat regardless of how much history has accumulated**:
  - the one event description (a headline, a macro release, a shock reading)
  - current indicator readings for the relevant coin(s) only — not a history of past readings
  - macro releases from the last 10 days only — not the full calendar
  - the currently-*accepted* candidate names, one line — not every candidate ever tracked
  - a two-line aggregate stat, not a per-trade history

  Measured real call (a macro event, all 7 coins' indicators): **~1,230 tokens** input, ~$0.01/call at current pricing. This number does not grow as the candidate registry or the live-test log grows — none of its five inputs are keyed to either.

**2. Human-triggered Q&A** — `answer_market_question()` (`replay/judgment.py`), `handle_natural_language()` (`telegram/bot.py`). Fires only when a human asks something free-form on Telegram. This one legitimately needs more breadth, since it has to be ready for *any* question ("what's accepted", "how's candidate X doing", "which coin is dragging things down", "what's the market doing") — but "more breadth" turned out to mean "every candidate ever discovered, and every (candidate, coin) pair's full trade history, in full detail, every single call," which is what actually caused the growth. Fixed by capping the exhaustive parts, not the categories of information:
  - `_trades_by_candidate_summary()` / `_trades_by_candidate_and_coin_summary()` (`replay/judgment.py`) and `build_live_test_summary()` (`context_builder.py`) now keep only the top 15 rows **ranked by `|mean_return|`** — the candidates/pairs most worth a human's attention either way (best or worst), never an arbitrary or alphabetical cut — with a note pointing to `/summary`/`/replay_summary` when the list was truncated.
  - `_all_candidates_status_summary()` (`replay/judgment.py`) collapses the `insufficient_data` bucket (usually the large majority — 67 of 96 tracked candidates in one real measurement) into a single count-plus-name-list line instead of one detailed line each, since there's rarely anything candidate-specific to say about a trigger that simply hasn't fired enough times yet. `accepted`/`watch`/`rejected`/`dropped` candidates — the ones a question is actually likely to be about — keep full detail, uncapped.
  - **Deliberately not capped:** the "already tested via 'test it' and not accepted" name list `build_context_summary()` (`context_builder.py`) gives Sonnet. This one is functional, not just informational — it's how Sonnet avoids re-proposing a condition a human already tested. Capping it risks a real regression (a duplicate proposal), unlike the other two lists, which exist purely for a human's convenience and have a free local alternative.

  Measured on this project's own real, accumulated state (96 tracked candidates, 1,728 logged live tests): **~10,400 tokens before this fix → ~2,700 tokens after (-74%)**, and now bounded — it will not keep growing as more candidates get discovered or more live tests resolve, because every uncapped block above has a hard ceiling.

**Why this covers every question, not just the ones anticipated in advance:** nothing is omitted *categorically* — Sonnet still sees the full picture at summary level (every candidate's status, every candidate's aggregate performance, current prices, open/closed test counts) on every single call. Only the exhaustive *tail* of per-row detail is capped, and only past the point where a human is realistically going to ask about that specific row: a candidate or pair with a huge |mean_return| (good or bad) is exactly the kind of thing worth asking about, and is guaranteed to survive the top-15 cut; a candidate sitting at N=2 with a near-zero return is unlikely to be the subject of a question, and if it genuinely is, the truncation note tells the reader (and Sonnet) exactly where to get the complete list instead of guessing or silently omitting it.

**For genuinely exhaustive questions, the answer is a free command, not a bigger LLM prompt.** `/summary` and `/replay_summary` (`telegram/bot.py`, via `format_trigger_summary()` in `candidates/methodology.py`) recompute the full battery fresh on demand — real statistics, no LLM call at all, no truncation, no cap — and cost **$0** in API terms (measured: ~4.4 seconds of local computation for 38 tracked candidates). The design split is deliberate: Sonnet answers *"what does this mean, why, what should I make of it"*; the local commands answer *"give me everything, exhaustively"* — the two questions don't need the same context budget, so they don't share one.

**A related correctness fix that's also a cost fix:** every Sonnet call in this project sets `max_tokens=2000` (not the 700-800 first used), after live testing showed the model spending an unpredictable, sometimes-large share of its output budget on an unrequested "thinking" block, occasionally leaving too little room for the actual JSON/text answer and truncating it (`stop_reason="max_tokens"`, observed live, not theoretical). A truncated call that then needs a retry effectively doubles its own cost; the fixed budget avoids that failure mode outright rather than paying for it via retries.

**Rough live-production estimate**, built from the numbers above rather than assumed: hourly shock scan (Sonnet only called on an actual detected shock, rare) + hourly headline screening (Haiku, cheap, batched) + a handful of escalations/day + occasional weekly keep/drop or milestone advice ≈ **$8-15/month** in moderate activity, likely under $25/month even in an unusually volatile month — a small fraction of the historical-replay demo's own cost, which compresses years of events into a short, continuous burst and is not representative of live, one-event-at-a-time operation.

## Trade Execution (Phase 2, Module B) — `execution/`

This project never opens a funded position (see [methodology-decisions.md](docs/case_study/methodology-decisions.md)) — the files below are grouped by which of the two live models they belong to.

**Live testing (the actual current model — no TP/SL, no funded position):**
- **`live_testing.py`** — Production's real-data counterpart to `replay/engine.py`'s day-by-day walker: `_open_live_test()`/`_check_live_tests()` (hold for the horizon `pattern_significance` found significant, resolve by measuring real forward return/MFE/MAE), `_scan_mechanical_triggers()` (unattended, no LLM, hourly detection), `find_backdated_entry()` (retroactively anchors a newly-discovered condition's own triggering occurrence to the real hour it first became true, using already-recorded price history — never something a funded order could do, legitimate here since nothing real is ever placed), `check_n50_milestones()`.
- **`live_test_state.py`** — Persistent state for the above (`live_tests.json`, `horizons.json`) — mirrors `replay/state.py`'s equivalent subset against real, not simulated, data.
- **`hyperopt_runner.py`** / **`freqtrade_bridge.py`** / **`freqtrade_userdir/strategies/hyperopt_candidate_strategy.py`** — A periodic, local-only, purely informational cross-check: runs Freqtrade's own (independent) Bayesian hyperopt against this project's real data to re-derive TP/SL multipliers for each tracked candidate, as a second opinion alongside this project's own walk-forward grid search. `freqtrade` is imported lazily (inside function bodies) so nothing else in this project gains a hard dependency on it.

**Superseded (kept, not deleted — the underlying TP/SL/anchor machinery is what the hyperopt cross-check above now reuses):**
- **`strategies/sentiment_agent_strategy.py`** — The Freqtrade `IStrategy` implementation this project's live execution used before the live-testing model above. `populate_indicators()` computes triggers on live data Freqtrade fetches itself; `populate_entry_trend()` decides entries (static battery or an approved manual signal); `custom_exit()` / `custom_exit_price()` implement the duration-bucketed TP/SL ladder. Not currently loaded for live trading.
- **`signal_store.py`** — Bridge between the LLM layer and the strategy above: a pending/active signal split so an approval is consumed exactly once and its anchors stay recoverable at exit time, days later.
- **`config_live.json`** — Freqtrade configuration for the strategy above: futures/isolated margin mode, dry-run, 7-coin pair whitelist.

## Telegram Interface (Phase 4) — `telegram/`

- **`bot.py`** — The long-polling bot process. Routes free-text questions to Sonnet (`handle_natural_language`); routes structured commands and button presses straight to a database query (`handle_command`, `handle_kpi_callback`, `handle_prune_callback`), never through an LLM.
- **`kpi_queries.py`** — Real SQL queries against the Freqtrade trade database: win rate, Sharpe, Sortino, max drawdown (on chronologically-ordered, summed — not compounded — returns, consistent with `methodology.py`'s own convention), filterable by coin, by signal, and by decision type (`signal_class`).

## Scheduling (Phase 3) — `scheduler/`

- **`weekly_revalidation.py`** — Refreshes market/funding data first, then re-runs the full candidate battery, diffs every candidate's status against the previous run, notifies on change, and handles the candidate keep/drop and shutdown flow. Written and tested end-to-end; not yet wired to an actual cron job. Any crash past the data refresh sends a Telegram alert before re-raising, so a silent failure is never indistinguishable from an ordinary "nothing changed" week.

## Documentation — `docs/case_study/`

- **`PLAN.md`** — Full build plan and the reasoning behind every methodology choice.
- **`methodology-decisions.md`** — Dated log of every non-obvious methodology number/design choice (why N=50, why the shock threshold is 3.0, why live tests hold for a fixed horizon instead of a TP/SL ladder, etc.) — each entry says plainly whether it's a statistical justification or a stated cost/data compromise, and why.
- **`assets/`** — Screenshots, the architecture diagram, and the current battery results table.

## Tests — `tests/`

- **`test_methodology.py`** — Causality safety (entry never fills on the trigger bar itself), anchor/barrier math, Sortino's downside-only penalty, concentration detection, status classification thresholds, and the shock z-score's backward-looking guarantee.
- **`test_status_history.py`** — The keep/drop-decision and shutdown logic, against an isolated temp file (never the real, committed `status_history.json`).
- **`test_novel_condition_tester.py`** — The indicator/operator/direction whitelist rejects anything outside it at construction time.
- **`test_run_battery.py`** — Integration test against the real historical data: one deliberately-broken candidate must not cost the other candidates their already-computed result.
- Runs automatically on every push via `.github/workflows/tests.yml`.

## Partial Failures & Crashes

What happens when something breaks mid-run, and what still doesn't handle it:

- **`candidates/run_battery.py`** — each candidate (static or dynamic) runs in its own try/except. A failure shows up as `status = "error"` in that week's table and in `meta["failed_candidates"]`, and is retried automatically next run — it does not touch that candidate's `status_history.json` entry (no real verdict was reached) and does not cost any other candidate its result.
- **`data_ingestion/market_data/binance_fetcher.py`** — same isolation per coin: one coin's OHLCV or funding fetch failing (`None` in the report, not `0`) doesn't block the other six.
- **`llm_pipeline/haiku_sonnet_pipeline.py::run_once()`** — each escalated headline is isolated; a malformed model response for one headline is logged and skipped, the rest of that run's headlines still get processed. `run_once()` and `run_shock_scan()` are also isolated from EACH OTHER in `__main__`, so one failing doesn't prevent the other from running.
- **`scheduler/weekly_revalidation.py`** — the one process-level alert: any exception past the (already-lenient) data refresh sends a Telegram message describing the failure before re-raising. Without this, "no message this week" was ambiguous between "nothing changed" (normal) and "the whole run silently died" — indistinguishable to the human either way.
- **`llm_pipeline/pending_tests.py`** — proposals expire after 48h rather than sitting in the queue indefinitely if nobody replies "test it".

**Known, not yet handled:** `telegram/bot.py::run_bot()`'s long-poll loop only wraps the per-update dispatch in a try/except — the `_get_updates()` call itself (the network request to Telegram) is not, so a transient network blip or a Telegram API error still crashes the whole always-on bot process and requires a manual restart. Documented here rather than silently left as a gap.

## Root files

- **`requirements.txt`** — Python dependencies.
- **`.env.example`** — Required environment variables (API keys, Telegram credentials) — names only, never real values.
- **`.gitignore`** — Keeps secrets and runtime state out of version control.
