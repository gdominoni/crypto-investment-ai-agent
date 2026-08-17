# Phase 9: Model Training Deep-Dive

**Goal:** the human director chose to defer VPS deployment (originally next) and go deeper on the ML/optimization side instead — a rigorous rebuild of Module C's classifier, then hyperopt for Module B's trend-following strategy.

## The prompt, and a real mid-conversation correction

"instead of taking care of the VPN [sic, VPS], let's move directly to the model training phase" — plus two direct questions: whether backtesting should stay local "to save tokens," and whether the ML/backtesting workload would be too heavy for the human director's Mac.

Both got a real answer, not a "sounds good": backtesting and ML training consume **zero LLM tokens** — they're pure local compute (Freqtrade, FreqAI/LightGBM, Hummingbot make no Anthropic API calls). The actual reason for keeping heavy compute local, established back in Phase 0-1, is cloud hosting *cost and capacity* — the VPS is meant to be free-tier or ~$5/month, sized for a lightweight always-on daemon, not CPU-intensive sweeps. On the hardware question: rather than speculate, checked actual specs (Apple M1, 8 cores, 8GB RAM) and pointed to real evidence already in hand — Module B's 3-year backtest and Module C's FreqAI training had already run successfully on this exact machine in earlier phases.

A clarifying question (`AskUserQuestion`) resolved what "model training phase" actually meant: both Module C's ML model and Module B's hyperopt, Module C first.

## The Module C spec update, and why it wasn't rubber-stamped

The human director then specified a much more rigorous methodology: dynamic per-fold threshold calibration, purged expanding-window walk-forward CV, Triple Barrier labeling, `scale_pos_weight`, SHAP feature selection. Asked directly to confirm agreement before proceeding — and the honest answer wasn't "yes to everything." Two things were flagged and resolved *before* writing any code:

1. **The spec silently implies switching Module C from regression to classification** (Phase 6 predicted continuous volatility; every term in the new spec — probability threshold, precision-recall curve, F-beta, `scale_pos_weight` — is classification vocabulary). Confirmed this was intended rather than assuming it.
2. **None of the new requirements fit FreqAI's walk-forward retraining** without heavy, undocumented subclassing. Recommended moving Module C's training entirely off FreqAI, to a bespoke local pipeline, with Freqtrade kept only for realistic trade simulation afterward — a real reversal of the Phase 2 instruction that named FreqAI for Module C, flagged as such rather than quietly changed.

Two further ambiguities were resolved before building: the spec's "maximize F0.5 (or maintain 80% precision)" read as two competing rules, resolved to one (maximize F0.5, with the 80% floor kept as a diagnostic flag); and the purge buffer size (unspecified) was proposed at "= the vertical barrier length," the standard Lopez de Prado sizing rule.

Full reasoning for both decisions in [decisions-log.md](decisions-log.md).

## What got built, in dependency order

1. **`labeling.py`** — Triple Barrier, adapted for a non-directional gate (symmetric ATR barriers, since Module C doesn't pick a direction). 5 tests.
2. **`threshold_calibration.py`** — F0.5-maximizing threshold selection with a precision-floor diagnostic. 4 tests.
3. **`feature_selection.py`** — SHAP-based feature ranking, deliberately run once (not per fold) — a documented deviation from a literal reading of the spec, for stability and compute-cost reasons. 2 tests, one of which confirmed the SHAP list-vs-array normalization branch was genuinely necessary (a library warning fired exactly where expected).
4. **`walk_forward.py`** — purged, expanding-window CV with a three-part fold structure (train → calibrate → trade), not the two-part structure a first reading of the spec might suggest — calibrating and evaluating on the same window would be circular. 7 structural tests, verifying exact purge-gap sizes and non-overlapping trading weeks against a synthetic dataset before ever touching real data.
5. **`train.py`** — orchestrates the above against 3 years of real BTC/USDT 1h data plus VIX.

## Two real findings from running it for real, not two clean passes

**Finding 1 — a real bug, caught before the expensive step.** The first labeling run, with "reasonable-sounding" default barrier parameters, produced an 86% high-risk base rate — the gate was flagging almost everything, which would have made the entire downstream walk-forward evaluation nearly meaningless (a classifier can trivially look "accurate" against a base rate that lopsided). Diagnosed via first-passage-time reasoning (a narrow barrier is touched almost certainly given a long enough window) and fixed by empirically sweeping the ATR multiple while holding the previously-agreed vertical barrier fixed, landing on 4.5x ATR for a ~41% base rate. Caught by checking the label distribution *before* running the full pipeline on it — a cheap check that avoided wasting the expensive one.

**Finding 2 — a genuinely informative model, and a genuinely unprofitable naive strategy, reported as two separate results.** The corrected pipeline produced real signal: 0.51 precision against a 0.41 base rate, out-of-sample, across 57 walk-forward folds — modest, honest, not spectacular. Translating that into an actual Freqtrade backtest (enter when calm, exit when high-risk) lost 54% over 493 real trades. These aren't in tension: the classifier predicts *whether* a large move is coming, not *which direction* — forcing it into a directional long/short rule is an arbitrary simplification done only to get comparable Win Rate/Sortino/Net Profit numbers, and the loss reflects that simplification's weakness, not the classifier's. Reported both findings honestly rather than letting one overshadow the other. Full writeup in [Module C's README](../../modules/module_c_volatility_ml/README.md).

## What this changed elsewhere

Fed into Phase 7's capital allocator, Module C is now excluded for **non-positive net profit** rather than **insufficient sample size** — a qualitatively better rejection, since the allocator now has enough real data (493 trades vs. the previous 2) to judge Module C on its actual economics rather than waiting for more data. `orchestrator/status.py` and `orchestrator/run_allocation.py` both needed a one-line update (the new strategy's name) to pick up the new result — caught by actually re-running the allocator after the change, not assumed to still work.

## Still pending

- Module C's real conclusion (it should function as a *gate*, not a scored fourth strategy) isn't wired into the orchestrator yet — it's still being evaluated as if it were a standalone trader, which this phase's own result argues against. A natural Phase 10+ question once the orchestrator's rebalancing logic is revisited.

## Part 2: Module B hyperopt

The second half of "both, Module C first." Freqtrade ships several built-in hyperopt loss functions, each optimizing a single metric (Sharpe, Sortino, Calmar, ...). None reflect this project's specific Win Rate → Sortino → Net Profit lexicographic priority — the same hierarchy `candidate_ranking.py` already applies everywhere else. Rather than accept that mismatch (tuning Module B against one standard while judging it by another), wrote a custom `IHyperOptLoss` (`project_hierarchy_loss.py`) whose composite score mirrors `candidate_ranking.py`'s sort key directly, verified against the exact `IHyperOptLoss` interface for the installed Freqtrade version (read from the actual source inside the Docker image, not assumed from memory or older documentation) before writing any code against it.

### A second real bug, same category as the first, one layer earlier

The first 100-epoch hyperopt run's reported "best" result had exactly **one trade** — a single lucky win, 100% win rate, rated above every larger, more realistic sample because the composite score let win rate dominate by scale with no floor on sample size. This is precisely the overfitting failure mode Module B's own `dynamic_min_trade_count` filter was built in Phase 4 to catch during candidate *selection* — reproduced, undetected, one layer earlier, inside the *search* itself. Caught by reading the actual "best" result's trade count rather than trusting the optimizer's reported objective value. Fixed by porting the same significance formula into the loss function (duplicated, not imported, due to the same Docker/`user_data`-mount boundary that led to `data_ingestion/macro_data/loaders.py` existing separately from Module C's `freqai_utils.py` in Part 1) — any candidate below the dynamic trade-count floor now scores as the worst possible outcome.

Logged as its own decisions-log entry specifically because it's the *same class of mistake* as Module C's barrier-calibration bug, in a different part of the codebase: a check that already existed elsewhere in the project didn't automatically apply itself in a new context, and had to be re-added deliberately. Worth carrying forward as a general habit: whenever a new consumer of backtest-style results gets built, ask explicitly whether it needs its own copy of the significance filter.

### The honest result, after the fix

150 properly-filtered epochs found: `adx_trend_threshold=20, ema_fast_period=27, ema_slow_period=96` — 139 trades, 50.4% win rate, but **still a net loss** (-19.05% over the In-Sample period, Sortino -0.46). Backtested against the genuine Out-of-Sample holdout (Phase 4's split, never touched during the search): 71 trades, 46.5% win rate, -9.28% loss, Sortino -0.56 — consistent with the IS result rather than sharply worse, which is itself informative: a large IS/OOS performance gap is the signature of overfitting, and its absence here suggests the search found this strategy family's genuine characteristics rather than fitting noise.

**Conclusion: rigorous hyperopt, run correctly, could not find a profitable EMA/ADX-crossover parameterization for BTC/ETH on this timeframe.** A real, informative negative result about this specific strategy family — not a tooling failure, and not hidden or softened to make Phase 9 look more successful than it was. The natural next step for Module B isn't more parameter search on this same strategy shape; it's new candidate strategy *families*.

## Part 3: three new strategy families, coarse-grid screened

Rather than accept Module B's negative EMA/ADX result as final, the human director directed a deliberate diversification: three structurally different strategy families (mean reversion, volatility breakout, volume-driven), each screened with a **hand-curated, timeframe-tailored coarse grid** — explicitly *not* hyperopt. A "Phase 2" fine-tuning/search pass was named up front and deliberately deferred until these raw results could be reviewed, not skipped past.

### The plan got a real critique before any code was written

Asked to estimate runtime and comment on the original (hyperopt-based) version of this plan before proceeding, the honest answer wasn't just a time estimate. Flagged: a full 3-family × 3-timeframe hyperopt sweep (9 independent high-dimensional searches, ~4 hours) is a real multiple-comparisons problem, not just a compute cost — the more independent searches run, the higher the chance *something* looks profitable by chance alone, which is exactly the failure mode this project had already caught twice by that point (the 1-trade "100% win rate" bugs in both Module C's labeling and Module B's first hyperopt loss function). Recommended staging the search instead of committing blind. The human director's actual response went further than that recommendation: replaced the hyperopt-heavy plan with a coarse, hand-curated grid entirely, which turned out both more interpretable *and* cheaper (~20 real minutes for 216 backtests, versus the ~4-hour hyperopt estimate) — a case of a redirected plan beating the original proposal on every axis, not just the one that was pushed back on.

### Two real bugs caught before trusting any result

1. **A silent no-op.** The orchestration script named each combination's auto-loaded Freqtrade parameter file after the strategy *class* (`MeanReversionBBRSI.json`); Freqtrade actually resolves it from the strategy's own *file path* (`mean_reversion_bb_rsi.json`), confirmed by reading `freqtrade/strategy/hyper.py` directly. Every "different" exit preset had silently been running on class defaults. Caught because two deliberately different presets produced bit-for-bit identical results -- treated as a red flag requiring investigation, not a coincidence to shrug off.
2. **A process-tracking mistake**, not a code bug: the sweep was launched with both a shell `&` and the tool's own background-execution flag, so the harness tracked only the outer wrapper and reported the ~90-minute job "complete" after a few seconds, while the real sweep ran on, orphaned. Caught by noticing the "completed" task's log had one progress line where ~216 were expected. Fixed by backgrounding it exactly one way.

Full reasoning for both in [decisions-log.md](decisions-log.md).

### The result: 216 backtests, zero genuine wins

Only 3 of 216 combinations were both statistically significant and profitable -- all 3 In-Sample only, and all 3 reversed to clear losses Out-of-Sample when checked (one swung from +91.9% IS to -38.4% OOS on the exact same parameters). Textbook overfitting/noise, caught cleanly by the same IS/OOS discipline this project has applied since Phase 4. 15m was uniformly the worst timeframe across every family, consistent with fee drag from very high trade frequency (400-1,000+ trades per combo against a real ~0.1% taker fee). An unexpected, genuinely useful finding: the "null" exit preset (pure indicator-driven exit, no SL/TP) had the *least bad* average result of all four exit presets tested -- suggesting the hand-designed indicator exits aren't these strategies' weak point; the arbitrary percentage-based emergency exits may be. Full matrix and narrative in [Module B's README](../../modules/module_b_trend_following/README.md#the-result-216-backtests-zero-genuine-wins).

**None of the three new families show genuine, OOS-validated edge, joining `TrendEmaAdx`'s already-negative result.** Nine family×timeframe combinations tested rigorously now, all wanting -- a real, cumulative negative finding about simple technical-indicator strategies on BTC/ETH spot at these timeframes, not a string of unlucky first attempts.

## Part 4: multi-factor confluence, three design iterations, four real bugs, one clean answer

After Part 3's isolated-family screening, the human director identified the actual methodological gap -- the families were never meant to be tested alone -- and the design went through three full iterations before any code ran: a hand-curated multi-factor grid (2 options/family), replaced by a true exhaustive 5,832-combination Cartesian product after the hand-curated version was judged to still use "hardcoded preset levels," which was itself replaced by an 8-combo Freqtrade-native hyperopt harness once running 5,832 individual Docker backtests was shown to cost hours to tens of hours in pure startup overhead. All three designs stayed in the README, marked superseded, not deleted -- each was a real, reasoned step given what was known at the time.

The final harness didn't run cleanly on the first, second, or third attempt. Four distinct, real issues surfaced only by actually launching it:
1. `DecimalParameter`'s `decimals` defaults to `3` internally, conflicting with the spec's `step=` unless explicitly overridden -- caught on launch one, with a too-broad first fix (also touching `IntParameter`, which has no `decimals` concept) reverted after checking the actual source rather than assuming the fix generalized.
2. The shared indicator module worked for a plain backtest but wasn't importable from hyperopt's *parallel* worker subprocesses -- they don't inherit Freqtrade's runtime `sys.path` modification, only the environment each subprocess starts with. Fixed by inlining each combo's indicator functions into its own file.
3. The spec'd `-j -1` caused *measured*, not hypothetical, memory pressure (swap at 92% of 4GB) -- a risk named during design review, confirmed with `vm.swapusage` rather than assumed away, and fixed by dropping to `-j 4` (already proven safe on this machine from the earlier `TrendEmaAdx` hyperopt run).
4. Combos 5-8 (the RSI-Bollinger-Bands variant) have no "sell"-space parameter at all -- their exit is fully data-driven -- so the harness's originally-unconditional `--spaces buy sell` broke them, discovered only after Combos 1-4 had already completed successfully. Fixed by giving each combo its own space list, and by switching to incremental CSV output after losing those 8 completed runs' in-memory results to the crash.

**The actual result, once it ran clean: 16 runs, zero combos statistically significant and profitable in both periods.** But not a flat "everything failed" -- a specific, genuinely useful finding emerged: the classic fixed-threshold RSI combos that cleared significance were *consistently* negative in both IS and OOS (a real, replicable loss), while the adaptive RSI-Bollinger-Bands combos showed dramatic IS-to-OOS reversals (one swinging from +33.9% to -20.7% on identical parameters) -- overfitting concentrated specifically in the adaptive indicator, plausible because a band fitted to a small in-sample RSI distribution captures that period's idiosyncrasies more readily than a fixed threshold does. Full table and reasoning in [Module B's README](../../modules/module_b_trend_following/README.md#the-result-16-runs-zero-combos-validated-in-both-periods) and [decisions-log.md](decisions-log.md).

## Phase 9 status: all four parts complete

Module C: rebuilt on a rigorous bespoke pipeline, real (if modest) classifier signal found, honestly reported as economically unprofitable when naively translated into trades. Module B: hyperopt on EMA/ADX (negative), three new families coarse-grid screened across three timeframes (negative), and finally an 8-combo multi-factor confluence hyperopt harness -- three design iterations, four real bugs, and a clean, specific negative result with a genuinely useful side-finding about RSI-BB's overfitting tendency. Neither module has a profitable, statistically significant candidate yet. Phase 9 was about building rigorous *methodology*, tested against reality at every step rather than trusted on paper -- the methodology is now trustworthy enough to believe its negative results across more than a dozen tested strategy shapes, which is real progress, even without a winning strategy to show for it yet.
