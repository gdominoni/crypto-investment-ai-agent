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

## Phase 9 continued: three new strategy families, coarse-grid screened (hyperopt strictly deferred)

Rather than keep tuning EMA/ADX crossover, the human director directed a deliberate diversification into three structurally different families, each with its own timeframe-tailored coarse grid — **hand-curated discrete parameter sets, not a search**. A "Phase 2" hyperopt/fine-tuning pass was explicitly deferred until these raw results could be reviewed first.

**Three families**, sharing `dynamic_exit_mixin.py` for what "null" SL/TP means (Freqtrade has no native "disabled" stoploss/ROI state, so null is emulated as -99% stoploss / 10,000%-required-profit ROI — one shared definition, not three slightly different approximations):
- `mean_reversion_bb_rsi.py` — Bollinger Bands + RSI + rolling Z-score. Enter on oversold confluence; exit on reversion to the mid-band or RSI > 50.
- `volatility_breakout_kc_squeeze.py` — Keltner Channels + TTM Squeeze momentum. Enter when a low-volatility squeeze releases with positive momentum; exit on momentum reversal or a close back inside the channel.
- `volume_driven_vwap_cmf.py` — rolling VWAP (a lookback-window approximation of a true anchored VWAP) + volume-surge detection + Chaikin Money Flow. Enter on a VWAP cross-up with a volume surge and positive CMF; exit on a cross back below VWAP or CMF turning negative.

Each family provides a genuine indicator-based exit — required so the grid's "null" SL/TP presets still have *some* way to close a position, rather than riding forever.

**The grid**, per `coarse_grid.py`: 3 entry presets **per family per timeframe** (15m/1h/4h; each timeframe's presets are independently tailored — a 15m mean-reversion RSI threshold and a 4h one aren't scaled versions of each other, they're separately reasoned about) × 4 shared per-timeframe exit presets (null/null, tight, wide, and an SL-only + trailing variant), run across both the In-Sample and true Out-of-Sample periods. 108 combinations × 2 periods = 216 real Docker backtests, `run_coarse_grid.py`.

### Two real bugs caught before trusting any result

1. **A silent no-op.** The first version named each combo's auto-loaded parameter file after the strategy *class* (`MeanReversionBBRSI.json`). Freqtrade actually resolves it from the strategy *file's own path* (`mean_reversion_bb_rsi.json`) — confirmed by reading `freqtrade/strategy/hyper.py` directly, not assumed. Caught because two deliberately different exit presets produced bit-for-bit identical results, which had no innocent explanation — every "different" combo had silently been running on the class defaults the whole time.
2. **A process-tracking mistake, not a code bug.** The first launch attempt double-backgrounded the sweep (shell `&` *and* the tool's own background-execution flag), so the ~90-minute job was only tracked for the few seconds its outer wrapper took to return, then ran on, orphaned and unmonitored. Fixed by relaunching with only the tool's own background tracking, which then correctly reported real progress and completion.

### The result: 216 backtests, zero genuine wins

| Family | Timeframe | Mean OOS net profit | Mean OOS win rate |
|---|---|---|---|
| Mean Reversion (BB+RSI) | 15m | -38.2% | 51% |
| Mean Reversion (BB+RSI) | 1h | -33.2% | 47% |
| Mean Reversion (BB+RSI) | 4h | -15.6% | 53% |
| Volatility Breakout (KC+Squeeze) | 15m | -72.8% | 26% |
| Volatility Breakout (KC+Squeeze) | 1h | -32.5% | 32% |
| Volatility Breakout (KC+Squeeze) | 4h | -15.9% | 45% |
| Volume-Driven (VWAP+CMF) | 15m | -23.8% | 20% |
| Volume-Driven (VWAP+CMF) | 1h | -11.6% | 27% |
| Volume-Driven (VWAP+CMF) | 4h | -6.7% | 29% |

Full 216-row matrix in [`coarse_grid_results.csv`](coarse_grid_results.csv).

**Only 3 of 216 combinations were both statistically significant and profitable — all 3 In-Sample only, and all 3 reversed to clear losses Out-of-Sample.** The starkest: `VolatilityBreakoutKCSqueeze` at 4h looked excellent In-Sample (+91.9% profit, Sortino 1.47, 107 trades) and lost -38.4% on the exact same parameters Out-of-Sample. This is the textbook overfitting/noise signature this project's IS/OOS discipline exists to catch — and it caught it, cleanly, across all three "hits."

**15m is uniformly the worst timeframe across every family** (-24% to -73% mean OOS profit), consistent with a real, predictable mechanism: high trade frequency (400-1,000+ trades per combo) means Binance's ~0.1% spot taker fee is paid far more often, and these strategies' edges (such as they are) aren't large enough to clear that fee drag.

**An unexpected, genuinely interesting finding**: averaged across every family/timeframe/period, the `null_null` exit preset (pure indicator-driven exit, no SL/TP at all) had the *least bad* mean result (-16.7%) of all four exit presets — beating `wide` (-24.7%), `tight` (-26.9%), and especially `sl_only_trailing` (-35.5%, the worst). This suggests the hand-designed indicator-reversal exits aren't the weak point in these strategies; the arbitrary percentage-based emergency exits may be doing more harm than good, cutting winners short or exiting on noise the indicator logic itself would have ridden through correctly.

**Conclusion: none of these three strategy families, in this coarse parameterization, show genuine, OOS-validated edge on BTC/ETH spot.** Consistent with `TrendEmaAdx`'s Phase 4/9 result — this is now the fourth and fifth and sixth strategy shape (nine family×timeframe combinations, really) tested rigorously and found wanting, not a first attempt that just needs more tuning.

Status: 🚧 Phase 1 coarse-grid screening complete and honestly reported (Phase 9). Phase 2 (fine-tuning/hyperopt) remains explicitly deferred pending the human director's review of these results — not started, and not assumed to be the right next step just because it's available.
