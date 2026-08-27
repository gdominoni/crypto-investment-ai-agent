# Project Map

A file-by-file guide to what each part of this codebase does. Companion reference to [README.md](README.md) — use this to navigate straight to the code behind any specific claim in it.

---

## Backtesting Methodology (Phase 1) — `candidates/`

- **`methodology.py`** — Core statistical engine, shared by every candidate. Encodes every methodology rule as code: `build_events()` (causality-safe entry, always the bar after a trigger), `compute_anchors()` (duration-bucketed TP/SL levels from real historical MFE/MAE, never a flat barrier), `walk_forward()` (expanding-window, out-of-sample validation), `report()` (win rate, strict win rate, and Sortino ratio, always returned together), `concentration_check()` (flags a result carried by a single coin or a single year), `shock_zscore_series()` / `classify_regime()` (statistical isolation of extreme market shocks from ordinary conditions).
- **`definitions.py`** — The three trigger families under test (C1 funding-rate crowding, C2 post-macro-release reaction, C6 efficiency-ratio trend), each a pure, independently testable function of price/volume/funding data.
- **`data_loading.py`** — Loads historical OHLCV and funding-rate series from `data/`.
- **`macro_calendar.py`** — FOMC/CPI release calendar, with timezone-aware (US Eastern → UTC) conversion logic.
- **`run_battery.py`** — Orchestrator: runs every static and dynamic candidate through `methodology.py`, assigns a live status (`validated` / `watch` / `rejected`), and writes the state file the execution engine reads from.
- **`status_history.py`** — Tracks how long each candidate has been monitored, to trigger a keep-or-drop decision after years without validating.

## Historical & Market Data — `data/`

- OHLCV data (daily/hourly) for 7 coins (BTC, ETH, BNB, XRP, DOGE, ADA, LTC) from Binance spot, funding-rate history, and macro series (CPI, Fed funds rate, VIX, gold, S&P 500) from FRED and Yahoo Finance. Coverage: 2017–2019 through 2026, depending on the coin's listing date.

## News Ingestion — `data_ingestion/news_sentiment/`

- **`cryptocompare_fetcher.py`** — The one file in this folder used by the live pipeline: fetches headlines from the CryptoCompare News API.
- **`rss_fetcher.py`, `aggregator.py`, `haiku_sentiment.py`** — An earlier multi-source design (CryptoCompare + RSS feeds, deduplicated, then scored). Not currently wired into the live pipeline.

## LLM Judgment Layer (Phase 2) — `llm_pipeline/`

- **`haiku_sonnet_pipeline.py`** — The adaptive layer. `haiku_scout()` (Claude Haiku screens every headline for asset/sentiment/magnitude/event type), `sonnet_strategist()` (Claude Sonnet's judgment, only for escalated headlines), `execute_routine_trade()` (fires a routine trade unattended — but only after independently re-checking the candidate is actually `validated` in the real state file, regardless of what the model claims).
- **`novel_condition_tester.py`** — Runs a human-approved novel condition through the same statistical pipeline as `methodology.py`. The model can only choose from a fixed whitelist of indicators (`SUPPORTED_INDICATORS`) — never arbitrary generated code.
- **`shock_detector.py`** — Real-time detection of extreme volatility, reusing the same statistical definition of "shock" as Phase 1.
- **`context_builder.py`** — Builds Sonnet's context from live state: real open positions read from the Freqtrade database, real candidate statuses — never a hand-maintained file.
- **`dynamic_candidates.py`** — Persistent registry of conditions discovered live, re-tested weekly alongside the static candidates.
- **`pending_tests.py`** — Holds the most recent novel-condition proposal awaiting a human's "test it" reply.

## Trade Execution (Phase 2, Module B) — `execution/`

- **`strategies/sentiment_agent_strategy.py`** — The Freqtrade `IStrategy` implementation. `populate_indicators()` computes triggers on live data Freqtrade fetches itself; `populate_entry_trend()` decides entries (static battery or an approved manual signal); `custom_exit()` / `custom_exit_price()` implement the duration-bucketed TP/SL ladder.
- **`signal_store.py`** — Bridge between the LLM layer and live execution: a pending/active signal split so an approval is consumed exactly once and its anchors stay recoverable at exit time, days later.
- **`config_live.json`** — Freqtrade configuration: futures/isolated margin mode, dry-run, 7-coin pair whitelist.

## Telegram Interface (Phase 4) — `telegram/`

- **`bot.py`** — The long-polling bot process. Routes free-text questions to Sonnet (`handle_natural_language`); routes structured commands and button presses straight to a database query (`handle_command`, `handle_kpi_callback`, `handle_prune_callback`), never through an LLM.
- **`kpi_queries.py`** — Real SQL queries against the Freqtrade trade database: win rate, Sharpe, Sortino, max drawdown, filterable by coin, by signal, and by decision type (`signal_class`).

## Scheduling (Phase 3) — `scheduler/`

- **`weekly_revalidation.py`** — Re-runs the full candidate battery, diffs every candidate's status against the previous run, and notifies on change. Not yet wired to an actual cron job.

## Documentation — `docs/case_study/`

- **`PLAN.md`** — Full build plan and the reasoning behind every methodology choice.
- **`assets/`** — Screenshots, the architecture diagram, and the current battery results table.

## Tests — `tests/`

- Currently empty.

## Root files

- **`requirements.txt`** — Python dependencies.
- **`.env.example`** — Required environment variables (API keys, Telegram credentials) — names only, never real values.
- **`.gitignore`** — Keeps secrets and runtime state out of version control.
