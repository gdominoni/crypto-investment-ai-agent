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

## Trade Execution (Phase 2, Module B) — `execution/`

- **`strategies/sentiment_agent_strategy.py`** — The Freqtrade `IStrategy` implementation. `populate_indicators()` computes triggers on live data Freqtrade fetches itself; `populate_entry_trend()` decides entries (static battery or an approved manual signal); `custom_exit()` / `custom_exit_price()` implement the duration-bucketed TP/SL ladder.
- **`signal_store.py`** — Bridge between the LLM layer and live execution: a pending/active signal split so an approval is consumed exactly once and its anchors stay recoverable at exit time, days later.
- **`config_live.json`** — Freqtrade configuration: futures/isolated margin mode, dry-run, 7-coin pair whitelist.

## Telegram Interface (Phase 4) — `telegram/`

- **`bot.py`** — The long-polling bot process. Routes free-text questions to Sonnet (`handle_natural_language`); routes structured commands and button presses straight to a database query (`handle_command`, `handle_kpi_callback`, `handle_prune_callback`), never through an LLM.
- **`kpi_queries.py`** — Real SQL queries against the Freqtrade trade database: win rate, Sharpe, Sortino, max drawdown (on chronologically-ordered, summed — not compounded — returns, consistent with `methodology.py`'s own convention), filterable by coin, by signal, and by decision type (`signal_class`).

## Scheduling (Phase 3) — `scheduler/`

- **`weekly_revalidation.py`** — Refreshes market/funding data first, then re-runs the full candidate battery, diffs every candidate's status against the previous run, notifies on change, and handles the candidate keep/drop and shutdown flow. Written and tested end-to-end; not yet wired to an actual cron job. Any crash past the data refresh sends a Telegram alert before re-raising, so a silent failure is never indistinguishable from an ordinary "nothing changed" week.

## Documentation — `docs/case_study/`

- **`PLAN.md`** — Full build plan and the reasoning behind every methodology choice.
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
