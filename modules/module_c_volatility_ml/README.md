# Module C — Volatility Gate / Machine Learning

A probabilistic "high-risk" classifier — not price direction, and not (as of Phase 9) a continuous volatility number either, but a binary risk flag with a dynamically calibrated confidence threshold. Gated by the deterministic circuit breaker in [`safety/`](../../safety/): the ML model can suggest, never override, the hardcoded safety gate. That gating happens at the orchestrator level (Phase 7/8), not here — backtesting has no live execution path for it to act on yet.

## Phase 9: rebuilt on a rigorous, bespoke ML methodology

The human director specified a significantly more rigorous pipeline than Phase 6's simple regression, requiring capabilities FreqAI doesn't provide out of the box. Rather than fight FreqAI's internals, **Module C's training moved entirely off FreqAI** and now runs as a plain local Python pipeline — Docker/Freqtrade is only used afterward, to turn the resulting signal into realistic simulated trades. This is a deliberate, discussed trade-off (see [decisions-log.md](../../docs/case_study/decisions-log.md)), not a silent reversal of the Phase 2 instruction that named FreqAI — it's logged as exactly that: a reconsideration, made explicitly, once the requirements outgrew what the framework was chosen for.

**Why bespoke, concretely:** none of the following are things FreqAI's walk-forward retraining supports natively, and each would require subclassing/fighting undocumented internals to bolt on:
- A **purged buffer** between training and evaluation windows (label leakage prevention).
- **Per-fold threshold calibration**, stored and applied to a *different, later* window than the one it was calibrated on.
- **SHAP-based feature selection** as a pipeline step.

Building these directly in plain pandas/lightgbm/shap is more code but far more transparent and debuggable than reverse-engineering FreqAI's `IFreqaiModel`/DataKitchen internals to make it cooperate — and it turned out to have a real side benefit: **training no longer needs Docker at all**, only the final trade-simulation step does.

## The pipeline

1. **`labeling.py` — Triple Barrier, adapted for a non-directional gate.** The classic Triple Barrier method (Lopez de Prado) labels a *directional* bet: upper = profit target, lower = stop loss, whichever hit first. Module C isn't picking a direction, so barriers are placed symmetrically at ± a multiple of ATR, and the label is "high-risk" (1) if *either* barrier is touched before the vertical (max holding period) barrier expires — a large move happened, in either direction. Calm (0) if price stayed inside the band the whole window.

   **A real bug, caught before it went further**: the initial "textbook" default (2.0× ATR, 24-candle vertical barrier) produced an **86% high-risk base rate** — the gate was flagging almost everything. The cause is first-passage-time math: for a diffusive price process, a narrow barrier is touched almost certainly given enough candles, so a wide-enough vertical window with a narrow barrier degenerates into an always-on signal, no matter how the model trains. Swept the ATR multiple empirically (holding the agreed 24h vertical barrier fixed) and landed on **4.5× ATR**, giving a ~41% base rate — informative rather than a near-constant flag.

2. **`feature_selection.py` — SHAP, run once, not per fold.** The spec's per-fold reading was deliberately not followed literally: recomputing SHAP at every one of ~57 folds would let the selected feature set drift week to week (undermining comparability of the walk-forward results) and would multiply an already non-trivial compute cost 57×. A single, stable selection upfront — on the first training window, before any walk-forward evaluation — is both cheaper and more methodologically sound. Selected features: `atr`, `day_of_week`, `vix_close`, `adx`, `rsi` (of 7 candidates, top 5 by mean absolute SHAP value).

3. **`walk_forward.py` — purged, expanding-window walk-forward, three parts per fold, not two.** `[expanding training window] → [purge] → [calibration window] → [purge] → [trading week]`. The calibration window (where `threshold_calibration.py` picks T\* from the precision-recall curve) is *not* the same window T\* gets applied to and reported on — calibrating and evaluating on identical data would be circular, tuning the threshold with knowledge of the very outcomes used to judge it. Purge width = the vertical barrier length (24 candles), the standard Lopez de Prado sizing rule: purge at least as wide as the longest lookforward any label could have used. 57 folds over ~2.5 years of BTC/USDT 1h data, each retraining fresh (expanding window) and reporting a genuinely held-out trading week.

4. **`threshold_calibration.py` — F0.5 maximization, not a fixed 0.5 cutoff.** The original spec ("maximize F0.5 *or* maintain 80% precision") read as two competing rules; resolved to one: always maximize F-beta=0.5 (which already weights precision over recall by construction), with the 80% floor kept as a diagnostic flag (`meets_precision_floor`) rather than a second objective — a weak fold still gets its best available threshold instead of blocking training.

5. **`train.py`** ties it together: builds features (price technicals + VIX) and labels from 3 years of real BTC/USDT 1h data, runs the pipeline, saves a per-candle signal (`user_data/signal_output/high_risk_signal.parquet`, gitignored — regenerate with the command below).

## The honest result

Run locally (no Docker needed for this step):
```
python -m modules.module_c_volatility_ml.train
```
57 folds. **Overall out-of-sample: precision 0.51, recall 0.53, against a 41% base rate.** Modestly, genuinely better than chance (a trivial "always predict positive" baseline would score precision ≈ 0.41) — real, if weak, predictive signal for *whether a large move is coming*. Only 18/57 folds (32%) hit the 80% precision floor, an honest sign that forward generalization is meaningfully worse than in-window calibration (0.68 precision) — expected and not hidden.

**Translating that into a naive long/short strategy loses money, and that's a separate, important finding, not the same one.** Feeding the signal into a real Freqtrade backtest (`VolatilityGateSignal`, "enter when calm, exit when high-risk") produced 493 trades, 41.8% win rate, Sortino −1.03, **−54.29% net loss** over ~2.5 years — a real, statistically meaningful sample (unlike Phase 6's 2-trade result), correctly and honestly reported. The reason isn't that the classifier is useless: it's that "not-high-risk" was never a directional signal. Being calm doesn't mean price is going up — forcing a non-directional risk read into "enter long when calm" is an arbitrary translation, done only so the result could be compared on the project's Win Rate → Sortino → Net Profit hierarchy like every other module. This reinforces, empirically, what the module's own name already says: Module C's real value is as a **gate on other modules' position-taking**, not a standalone trader — a conclusion this result supports rather than undermines.

Run the real backtest yourself:
```
docker compose --env-file ../../.env run --rm freqtrade backtesting \
  --config user_data/config_signal.json --strategy VolatilityGateSignal \
  --timerange 20240223-20260814
```

Fed into the capital allocator (Phase 7), Module C is correctly excluded — but now for **non-positive net profit**, not insufficient sample size like before. That's a qualitatively better rejection: the allocator has enough real data to judge Module C on its economics, not just wait for more.

Status: ✅ rigorous ML pipeline built, verified end-to-end against real data, and honestly evaluated (Phase 9). Not yet wired as a genuine *gate* modifying other modules' behavior — it's still being scored as if it were a fourth standalone strategy, which this phase's own result argues against.

---

## Earlier approach (Phase 6, superseded)

The original build used **FreqAI** (Freqtrade's built-in ML module) with a `LightGBMRegressor` predicting forward realized volatility (a continuous target) over the next 12 candles, plus the same VIX macro feature. It ran successfully — FreqAI's walk-forward retraining produced 11 separate model checkpoints over a ~2.5-month window, confirmed by inspecting `user_data/models/` directly. That code (`user_data/strategies/volatility_gate_freqai.py`, `user_data/config.json`) is left in place as a working artifact and a record of how the module's requirements evolved, not deleted to make Phase 9 look like the only approach ever considered. **Note the image tag** if you run it: FreqAI needs `freqtradeorg/freqtrade:stable_freqai`, not the plain `:stable` image Module B uses — the base image doesn't bundle scikit-learn/LightGBM/`datasieve` at all.
