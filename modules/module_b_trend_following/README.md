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

## ⚠️ SUPERSEDED: isolated single-family testing (kept for the record, not deleted)

**This section describes an approach that was replaced, not extended.** Testing the three families *in isolation* (each alone, never combined) was a methodological gap the human director caught after reviewing these results: a breakout strategy tested alone will buy into chop with no volume confirmation to filter it out; a mean-reversion strategy tested alone will buy a falling knife with nothing to confirm the reversal is real. The combined design that replaced this (below, "Multi-Factor Confluence") requires all three families to agree before entering — see [decisions-log.md](../../docs/case_study/decisions-log.md) for the full reasoning. The section below is left intact because the negative result it found (no single family, alone, at any timeframe up to 4h, shows OOS edge) is still real, still true, and directly motivated the redesign — a case study should show the wrong turn, not just the corrected one.

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

Status: 🚧 Phase 1 coarse-grid screening complete and honestly reported (Phase 9). Superseded by the multi-factor confluence redesign below.

## ⚠️ SUPERSEDED: hand-curated multi-factor grid, and the exhaustive 5,832-combo grid that followed it (both kept, neither ever run)

The design below (hand-curated presets, 2 options per family, 96 backtests) was replaced before it ran: reviewing it, the human director wanted a true parameter sweep rather than hand-picked "logical" presets. That became `full_grid_1h.py` — an exhaustive Cartesian product (36 x 9 x 9 x 2 = 5,832 combinations, verified programmatically), scoped to 1h only. That in turn was replaced before running either: executing 5,832 combinations (11,664 with OOS) as individual Docker/CLI backtests — the mechanism proven out in Phase 1 — would have cost hours to tens of hours in pure container-startup overhead, a concern raised at the time. The final design (below, "8-Combo Hyperopt Harness") resolves this by using Freqtrade's own hyperopt engine (one persistent process per run, not one container launch per parameter combination) and restructuring Family 2/3 from combined conditions into independent alternatives — which also, as a side effect, raises trade frequency versus the rarer "squeeze release" trigger the multi-factor design depended on. All three designs are kept in this README, in order, because each one was a real, reasoned step -- not a mistake to hide.

## Multi-Factor Confluence — hand-curated grid (superseded, never run)

Two findings from the isolated-family screening directly drove this redesign:
1. **15m dropped entirely.** Every family lost heavily there (-24% to -73% mean OOS profit), consistent with fee drag from hundreds of trades at ~0.1% taker fee per side. Only **1h, 4h, and 1d** remain in scope.
2. **Exit presets simplified to 2 per timeframe** (`null` and one `wide` SL+TP pair, materially wider than Phase 1's tightest tiers) — the `null_null` indicator-driven exit was the *least bad* of Phase 1's four exit presets on average, meaning arbitrary fixed-percentage stops were plausibly cutting real winners short more than they were protecting against real losers.

**The bigger change: `MultiFactorConfluence` (`user_data/strategies/multi_factor_confluence.py`) combines all three families on every candle, in asymmetric roles**, rather than testing them in isolation:
- **Family 2 (volatility breakout) is the timing trigger** — a squeeze release is a naturally rare, punctual event, well suited to being the thing that actually opens a trade.
- **Family 1 (RSI) is a confirming filter**: RSI must not already be overbought at the trigger, guarding against chasing an already-extended move.
- **Family 3 (volume) is a confirming filter**: real volume + positive Chaikin Money Flow must be present at the trigger, guarding against a low-volume fakeout — directly targeting "breakouts buy into chop."

A flat AND of all three families' *original*, full multi-part conditions was considered and rejected: that would multiply 3+3+3 sub-conditions together, likely firing so rarely (especially at 4h/1d, where trade counts were already thin in Phase 1) that no combination would ever clear the statistical significance floor. The trigger/filter split keeps the entry bar high without making it nearly unreachable.

**New indicator, per instruction**: Family 1 offers two variants — `classic` (RSI below a fixed overbought threshold) and `rsi_bb` (Bollinger Bands applied to the **RSI series itself**, not price — RSI below its own upper band, adaptive to how volatile RSI has recently been rather than a fixed level regardless of regime). Verified locally that `pandas_ta.bbands` works correctly on any series, RSI included, before writing this into strategy code.

**Exit is intentionally looser than entry**, not a mirror of the strict confluence required to open a position: exits on *any* of breakout momentum fading, price closing back below VWAP, or the RSI-based signal (matching whichever Family 1 variant is active) crossing back into overbought — protecting a position should be faster than the bar for opening one.

### The proposed grid (per `multi_factor_grid.py`)

2 Family-1 options × 2 Family-2 options × 2 Family-3 options × 2 exit presets = **16 configs per timeframe**, × 3 timeframes × 2 periods (IS+OOS) = **96 backtests total** — fewer than Phase 1's 216, despite testing a combined (not isolated) signal, because each family's own option count was deliberately shrunk from 3 to 2 specifically to keep a 3-way Cartesian product from exploding.

Full literal matrix (all 48 unique configs, no execution): `python -m modules.module_b_trend_following.print_multi_factor_matrix`.

**A risk worth naming before running anything**: requiring 3-way confluence will trade less often than any single-family version, and Phase 1 already showed 4h struggling to clear the significance floor even for single-factor entries. 1d is untested territory entirely (no data downloaded yet). If this grid returns too few trades to be significant anywhere, the honest fallback would be loosening the "all three" requirement to "any two of three," not quietly abandoning the significance filter — a decision to make with real data in hand, not now.

**Status: specification only, never run.**

## ⚠️ SUPERSEDED: exhaustive 5,832-combo grid (never run)

`full_grid_1h.py` defined a true, hand-verified Cartesian product for 1h only: Family 1 (RSI length x threshold pairs x RSI-BB std = 3x3x4=36) x Family 2 (BB std x KC ATR mult = 3x3=9) x Family 3 (volume surge mult x CMF threshold = 3x3=9) x exit variants (2) = **5,832 combinations exactly**, confirmed by generating and counting them programmatically, not just checking the arithmetic. Replaced before running: at 5,832 combos (11,664 with IS+OOS), running each as an individual `docker compose run` invocation -- the mechanism proven out in Phase 1 -- would cost hours to tens of hours in pure Docker/Python startup overhead alone, on top of actual backtest compute. That concern, raised at review time, led directly to the current design.

## 8-Combo Hyperopt Harness (current design, pending approval to launch)

Resolves the exhaustive-grid's efficiency problem by using **Freqtrade's own hyperopt engine** — one persistent process per run, loading data once and evaluating many parameter sets internally — instead of one Docker container launch per combination. Restricted to **1h only**.

**Family 2 and Family 3 are now independent alternatives, not combined conditions.** Earlier designs required Bollinger-inside-Keltner (a "squeeze") for Family 2 and volume-surge-and-CMF together for Family 3 — both comparatively rare, compound events. Now each family offers two standalone options (Price BB *or* Keltner as the breakout trigger; Volume Surge *or* CMF as the confirming filter), which trade independently more often — directly addressing the trade-frequency risk flagged in the multi-factor design above, as a side effect of a change made for a different reason (execution efficiency).

**8 base architectures** (`user_data/strategies/combo1..8_*.py`, sharing indicator logic via `confluence_indicators.py`): every combination of Family 1 (classic RSI *or* RSI-Bollinger-Bands) x Family 2 (Price BB *or* Keltner) x Family 3 (Volume Surge *or* CMF) = 2x2x2 = 8. Each combo declares *only* its own relevant hyperopt parameters (e.g. Combo 1 never sees `rsi_bb_std` or `kc_atr_mult`, since it doesn't use RSI-BB or Keltner) — keeping each run's search space scoped to what actually affects that combo's behavior.

**Coarse, stepped parameter spaces** (`step=5` for integer lengths/thresholds, `step=0.5` for stddevs/multipliers) — deliberately discretizing what would otherwise be a continuous search, so hyperopt can't report false precision (e.g. `rsi_period=17.3`) that isn't really distinguishable from noise at this sample size.

**16 total hyperopt runs** (8 combos x 2 exit modes — `null` vs a flat -15%/+15% SL/TP), each using `ProjectHierarchyLoss` (reused unchanged from the earlier `TrendEmaAdx` hyperopt work — no changes needed, since it already implements this project's Win Rate → Sortino → Net Profit hierarchy generically) with `--spaces buy sell`, 200 epochs, `-j -1`. Exit mode is fixed per run (not searched) via the same auto-loaded parameter-file mechanism used throughout this project — confirmed in earlier work that a space excluded from `--spaces` is preserved from the loaded file through to hyperopt's own exported "best params," not reset to a class default.

After each of the 16 runs: the discovered best parameters are re-backtested cleanly on IS (for a parseable result) and validated against the untouched OOS period — all 16, not a filtered top-K, since 16 runs is cheap enough to validate exhaustively (unlike the 5,832-combo design, where that distinction mattered). `run_hyperopt_harness.py` orchestrates all of this and prints one consolidated IS-vs-OOS matrix at the end.

**Estimated runtime, not yet empirically confirmed**: roughly 4-8 minutes per hyperopt run at `-j -1` (8 cores) x 16 runs ≈ 1-2 hours, plus ~30 minutes for the 32 quick IS/OOS re-backtests. **A real risk worth flagging**: `-j -1` uses all 8 CPU cores, each hyperopt worker holding its own copy of the loaded dataset in memory — on this machine's 8GB RAM (see Phase 9's model-training-feasibility discussion), this may need falling back to fewer workers if memory pressure shows up in practice, not assumed away in advance.

**Status: harness built and import-verified (no Docker/backtest execution). Awaiting approval to launch.**
