# Module B — Dynamic Trend-Following

Systematic trend-following strategies on futures/spot with automated parameter optimization. Built on **Freqtrade, run via Docker** (see "Why Docker, not pip" below), starting in `--dry-run` mode. Strategy candidates are filtered by a dynamic minimum trade-count threshold, then ranked by Win Rate → Sortino Ratio → Net Profit after fees — the same hierarchy used for live/dry-run monitoring later.

## Why Docker, not pip, for this module

Freqtrade (and Hummingbot, Module A) aren't ordinary Python packages — they depend on **TA-Lib**, a technical-analysis library written in C that has to be *compiled* for the exact combination of operating system, chip architecture, and Python version on the machine running it. Getting that combination to line up correctly by hand is notoriously fragile.

We hit this firsthand, not hypothetically: back in Phase 2, `pip install hummingbot` failed outright on this machine, because Hummingbot's pinned build tools (a specific Cython version) simply don't support the Python version already installed here. That's the exact class of problem Docker exists to solve.

A Docker "image" is a pre-built, pre-tested package containing the *entire* environment a piece of software needs to run — the operating system, the exact Python version, every compiled C library — assembled once by the framework's own maintainers and shared as a single unit. Running Freqtrade via Docker means running that finished, verified environment directly, instead of trying to reconstruct it by hand and hoping every version lines up.

This buys three concrete things for this project:
1. **It just works.** The strategy in this module uses TA-Lib indicators (EMA, ADX, ATR) that would very likely have hit the same compilation wall pip hit in Phase 2 — inside the Freqtrade Docker container, they worked immediately, no setup.
2. **Identical behavior everywhere.** The same container that runs on this Mac during development will run, unmodified, on the Linux cloud server in production (Phase 10) — removing an entire category of "worked on my machine, broke on the server" bugs before they can happen.
3. **Isolation between modules.** Module A (Hummingbot) and Module B (Freqtrade) can each run in their own container without their dependencies ever colliding — important on a $0–5/month server, where debugging a dependency conflict remotely is far more expensive than avoiding one.

## What's built

- `docker-compose.yml` — runs the official `freqtradeorg/freqtrade:stable` image, mounting `user_data/` and passing Binance credentials through Freqtrade's own env-var convention.
- `user_data/config.json` — dry-run, spot, BTC/USDT + ETH/USDT, 1h timeframe.
- `user_data/strategies/trend_ema_adx.py` — first candidate strategy: EMA(20/50) crossover gated by an ADX trend-strength filter, ATR-informed stoploss, trailing stop. A starting candidate for the ranking pipeline, not a tuned final strategy.
- `oos_split.py` — splits a date range into In-Sample / Out-of-Sample periods, holding out the most recent 12 months per the project's operational rules.
- `candidate_ranking.py` — `dynamic_min_trade_count()` (a statistical floor + an activity floor, whichever is stricter) and `filter_and_rank()` (Win Rate → Sortino → Net Profit, applied only to candidates that clear the trade-count filter).
- `run_backtest.py` — orchestrates the whole pipeline: runs IS and OOS backtests in Docker for every candidate, ranks the IS results, reports OOS separately as a validation check that never influences ranking.

## Running it locally

```
# One-time: pull the image and download history
cd modules/module_b_trend_following
docker compose --env-file ../../.env pull
docker compose --env-file ../../.env run --rm freqtrade download-data \
  --config user_data/config.json --pairs BTC/USDT ETH/USDT --timeframe 1h --days 1095

# Run the full IS/OOS ranking pipeline
cd ../..
python -m modules.module_b_trend_following.run_backtest
```

## First result (honest, not cherry-picked)

The first candidate, `TrendEmaAdx`, was **rejected by the pipeline**: over the 2-year in-sample period it produced 81 trades against a dynamically-computed minimum of 105 — not enough to be statistically trustworthy — and it lost money in both the in-sample and out-of-sample periods anyway. This is the pipeline doing its job: a thin-sample, unprofitable strategy should never reach the ranking stage.

## Phase 9: hyperopt, with a loss function matching the project's own ranking hierarchy

`ema_fast_period`, `ema_slow_period`, and `adx_trend_threshold` are now hyperoptable (`space="buy"`). Freqtrade ships several built-in hyperopt loss functions (`SharpeHyperOptLoss`, `SortinoHyperOptLoss`, `CalmarHyperOptLoss`, etc.), but each optimizes a single metric — none reflect this project's specific **Win Rate → Sortino → Net Profit** lexicographic priority, the same hierarchy `candidate_ranking.py` and the capital allocator use everywhere else. Using a mismatched built-in would mean Module B gets *tuned* against a different standard than the one it's *judged* by. `user_data/hyperopts/project_hierarchy_loss.py` closes that gap: a composite score (`win_rate × 1,000,000 + sortino × 1,000 + net_profit_pct`) that lets win rate dominate by scale, exactly mirroring `candidate_ranking.py`'s sort key, collapsed into one scalar hyperopt's optimizer can search over. It reuses Freqtrade's own `calculate_sortino` (the same function behind the backtest report's "Sortino (closed trades)" figure), so hyperopt's notion of "better" stays consistent with what gets read from the resulting backtest zip afterward.

**A real bug, caught by reading the first result instead of trusting the objective value**: the first 100-epoch run's "best" result had **1 trade** — a single lucky win with a trivial 100% win rate, which the composite score rated above every multi-trade candidate, since win rate dominates the score by scale with nothing to stop a thin sample from claiming a perfect one. The loss function didn't apply the project's own statistical-significance filter (`dynamic_min_trade_count`, from `candidate_ranking.py`) before scoring — exactly the overfitting trap that filter exists to catch, reproduced inside the search itself. Fixed by porting the same significance check into the loss function (duplicated, not imported — this file runs inside the Freqtrade Docker container, which only has `user_data/` mounted, the same boundary that led to `data_ingestion/macro_data/loaders.py` existing alongside Module C's `freqai_utils.py`): any candidate below the dynamic trade-count floor is scored as the worst possible outcome, same as zero trades.

**Honest result after the fix**: 150 properly-filtered epochs, best candidate found — `adx_trend_threshold=20, ema_fast_period=27, ema_slow_period=96` — 139 trades, 50.4% win rate, **still a net loss** (-19.05% IS, Sortino -0.46). Re-run as a full backtest against the genuine OOS holdout (never touched during the search): 71 trades, 46.5% win rate, -9.28% loss, Sortino -0.56 — consistent with the IS result rather than a sharp drop-off, which is itself informative: it suggests the search found the strategy family's real characteristics rather than overfitting to IS noise (an overfit result would show a much bigger IS/OOS gap). **Rigorous hyperopt, done properly, could not find a profitable EMA/ADX-crossover parameterization for BTC/ETH on this timeframe** — a real negative result about this strategy family, not a tooling failure.

Run it yourself:
```
docker compose --env-file ../../.env run --rm freqtrade hyperopt \
  --config user_data/config.json --strategy TrendEmaAdx \
  --hyperopt-loss ProjectHierarchyLoss --spaces buy \
  --timerange 20230820-20250822 -e 150 -j 4
```

Status: 🚧 infrastructure built and verified end-to-end (Phase 4); hyperopt built and run with a project-consistent objective (Phase 9). No candidate has cleared both the significance filter and profitability yet — more candidate strategy *families* (not just parameter sweeps of this one) are the natural next step.
