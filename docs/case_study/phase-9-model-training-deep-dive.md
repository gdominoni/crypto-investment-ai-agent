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

- Module B hyperopt — the second half of this phase, not yet started.
- Module C's real conclusion (it should function as a *gate*, not a scored fourth strategy) isn't wired into the orchestrator yet — it's still being evaluated as if it were a standalone trader, which this phase's own result argues against. A natural Phase 10+ question once the orchestrator's rebalancing logic is revisited.
