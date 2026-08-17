# Phase 4: Module B — Trend-Following (Freqtrade)

**Goal:** stand up Freqtrade via Docker, write a first trend-following strategy candidate, and build the IS/OOS backtest + dynamic-filter + ranking pipeline defined in the original spec's "Backtest & Strategy Selection Phase" rules.

## The prompt

"continue" plus one explicit instruction: document, in the README, *why* Docker beats pip for this part of the system — framed for a case-study audience that needs to see competence without wading through unnecessary technical depth. That requirement is addressed in the main [README, Section 2](../../README.md#2-cost--infrastructure-architecture) and in [Module B's README](../../modules/module_b_trend_following/README.md#why-docker-not-pip-for-this-module) in more detail; this entry covers the reasoning and build process behind it.

## Decisions made in this phase

1. **Docker over pip, proven, not just asserted.** This decision was actually made back in Phase 2 (when Hummingbot's pip install failed), but Phase 4 is where it was acted on and verified: Freqtrade's official Docker image ran the strategy's TA-Lib-based indicators (EMA, ADX, ATR) immediately, with zero manual setup — the same category of dependency that broke under pip. The case-study writeup leads with that concrete, lived example rather than a generic "Docker is more reproducible" claim, since a reader can verify it against Phase 2's own log.
2. **Freqtrade's own history format, separate from Phase 2's market data pipeline.** Freqtrade's backtesting engine reads OHLCV from its own on-disk format (via `freqtrade download-data`), not the parquet files `data_ingestion/market_data/binance_fetcher.py` produces. Rather than building a converter between the two, Module B pulls its own copy directly — Phase 2's pipeline exists for cross-asset feature engineering (Module C, regime detection), Module B's for what Freqtrade itself needs. Documented so this doesn't read as duplicated, undocumented work later.
3. **`dynamic_min_trade_count()` combines two independent floors.** The spec asks for a *dynamic* minimum trade-count filter without specifying the formula. Implemented as the stricter of: (a) a statistical floor — the standard margin-of-error sample-size formula for estimating a proportion (win rate) at 95% confidence, worst-case variance — and (b) an activity floor — at least 1 trade/week on average over the backtest window, so a strategy can't clear the statistical bar on a handful of lucky trades spread across years. Both floors scale with `backtest_days`, so a longer backtest is genuinely held to a higher bar, matching the spirit of "dynamic."
4. **OOS is reported, never ranked.** `run_backtest.py` computes Out-of-Sample results for every candidate but excludes them entirely from `filter_and_rank()` — matching the spec's framing of OOS as mandatory *validation*, not a second scoring input. Ranking happens on In-Sample results only.
5. **Freqtrade's config schema validation rejected `telegram`/`api_server` blocks with `enabled: false`** unless every other required key in that block was also present (e.g. `token`, `chat_id`, `jwt_secret_key`) — a schema quirk, not documented obviously. Simplest fix: omit both blocks entirely, since Freqtrade defaults them to disabled when absent. Found by actually running `show-config` against the file rather than assuming the config was valid because it looked reasonable.

## What got built and verified (against live Docker + real market data)

- `modules/module_b_trend_following/docker-compose.yml` + `user_data/config.json` — validated via `freqtrade show-config`, exchange credentials confirmed picked up from `.env` through Freqtrade's `FREQTRADE__EXCHANGE__KEY/SECRET` env-var convention.
- `user_data/strategies/trend_ema_adx.py` — first candidate: EMA(20/50) crossover gated by ADX > 25, ATR-informed stoploss.
- 3 years of real BTC/USDT and ETH/USDT 1h history downloaded via `freqtrade download-data` (26,289 candles each).
- A full-history backtest ran cleanly end-to-end, confirming Freqtrade natively computes Sortino (no need to hand-roll it) alongside win rate and net profit.
- `oos_split.py` and `candidate_ranking.py` — pure, unit-tested logic (7 new tests, 24 total passing project-wide).
- `run_backtest.py` — ran the complete pipeline against Docker for real: 2-year In-Sample backtest (81 trades) and 1-year Out-of-Sample backtest (40 trades), both against a live Freqtrade container.

## The honest result

`TrendEmaAdx` was **rejected by the filter**: 81 In-Sample trades against a dynamically-computed requirement of 105, and unprofitable in both periods regardless (-14% IS, -11% OOS). This is the pipeline working as intended, not a setback to fix — a naive first candidate correctly failing both the statistical-significance filter and the profitability bar is exactly the kind of case this infrastructure exists to catch before a strategy ever reaches paper trading, let alone live capital.

## Still pending

- More strategy candidates — one is not a meaningful selection pool.
- Hyperopt parameter sweeps (mentioned in the original spec, not yet built).
- Wiring Module B into the safety kernel (Phase 3) — the circuit breaker and risk limits aren't yet called from anywhere inside this module.
