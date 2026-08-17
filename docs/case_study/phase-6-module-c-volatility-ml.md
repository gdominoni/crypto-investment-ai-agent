# Phase 6: Module C — Volatility Gate (FreqAI)

**Goal:** build the ML-driven volatility signal on FreqAI, as specified early in the project ("Restrict macro features to these core series to maintain high signal-to-noise ratio and avoid overfitting in Module C (FreqAI)"), and verify it actually trains and backtests end-to-end.

## The prompt

"continue" — a plain continuation after Phase 5's back-and-forth debugging. The specific engineering choices below extend both the FreqAI commitment made explicitly in the Phase 2 spec update and the Docker-first pattern established in Phases 4-5.

## Decisions made in this phase

1. **FreqAI needs a different Docker image tag.** Discovered by trying the obvious thing first (reuse Module B's `:stable` image) and reading the actual error: `ModuleNotFoundError: No module named 'datasieve'`. Switched to `freqtradeorg/freqtrade:stable_freqai` and confirmed via `list-freqaimodels` that `LightGBMRegressor` loads correctly before writing any strategy code against it — verifying the foundation before building on it, the same discipline applied to every prior Docker setup in this project.
2. **The model predicts volatility, not price direction.** `set_freqai_targets()` labels each candle with realized volatility (return std) over the next 12 candles — a regression target — rather than a price-direction classification. This matches the module's actual job per the spec (a *volatility gate*, feeding the orchestrator's caution level) rather than turning it into an unrelated fourth trading strategy competing with Module B.
3. **VIX wired in as a real feature, not just referenced in docs.** The spec's macro-feature restriction explicitly named Module C, so this phase actually merges Phase 2's cached VIX data into the FreqAI feature set (`%-vix_close`) via `pd.merge_asof`, rather than leaving that connection as an aspiration. Verified by inspecting a trained model's `metadata.json` directly and confirming `%-vix_close` is genuinely in `training_features_list` — not just present in the code that was supposed to add it.
4. **Circuit-breaker gating deliberately deferred, and the deferral is explained, not silently skipped.** The module's docstring and README both say explicitly why `safety/circuit_breaker.py` isn't imported here: backtesting has no live execution path, so there's nothing for a live safety gate to act on yet, and wiring it in now would add complexity with nothing to verify. That gating is Phase 7/8's job, at the orchestrator level.
5. **`freqai_utils.py` factored out for local testability**, following the exact pattern set in Module B (`oos_split.py`, `candidate_ranking.py`): the VIX-loading/column-flattening logic has zero Freqtrade dependency, so it can be unit tested locally without Docker, loaded by explicit file path since it lives inside a `user_data/strategies/` tree rather than the `modules` Python package.

## What got built and verified (against live Docker + real market and macro data)

- `modules/module_c_volatility_ml/docker-compose.yml` + `user_data/config.json` — validated cleanly via `show-config` on the first attempt, no schema errors (contrast with Module B's `telegram`/`api_server` schema surprises in Phase 4 — this time the config was written correctly the first time, informed by that earlier experience).
- `user_data/strategies/volatility_gate_freqai.py` — feature engineering (RSI/ATR/ADX at multiple periods, returns, volume, time-of-day, VIX), volatility target, LightGBM regressor.
- A real backtest over a ~2.5-month window (1 year of BTC/USDT 1h data downloaded first) ran cleanly end-to-end.
- **Confirmed genuine walk-forward retraining**, not just a successful exit code: inspected `user_data/models/module_c_volatility_gate/` directly and found 11 separate `sub-train-BTC_*` checkpoints, matching the configured 60-day train / 7-day retrain cadence over the backtest window.
- 4 new local unit tests for `freqai_utils.py` (32 passing project-wide), verified against a synthetic parquet matching yfinance's actual MultiIndex column structure.

## Still pending

- Predictive performance hasn't been evaluated against the project's Win Rate → Sortino → Net Profit ranking hierarchy the way Module B's candidates are — Module C produces a continuous volatility estimate, not discrete trades in the same sense, so how it plugs into that ranking (or whether it's ranked at all, versus consumed directly by the allocator) is a Phase 7 design question, not resolved here.
- Circuit-breaker wiring, as noted above — Phase 7/8.
- Only BTC/USDT and only VIX from the curated macro list so far — ETH/USDT and the remaining FRED series (Fed Funds, unemployment, CPI, industrial production) are natural extensions once there's a concrete reason (a specific signal hypothesis) to add them, consistent with the project's "expand deliberately, not by default" pattern.
