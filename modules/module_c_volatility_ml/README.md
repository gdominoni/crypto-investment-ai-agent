# Module C — Volatility Gate / Machine Learning

Probabilistic ML signal — predicts forward-looking realized volatility, not price direction — gated by the deterministic circuit breaker in [`safety/`](../../safety/). The ML model can suggest, never override, the hardcoded safety gate; that gating happens at the orchestrator level (Phase 7/8), which will check `safety.circuit_breaker.evaluate_circuit_breaker()` before ever acting on this module's output live. Backtesting has no live execution path, so there's nothing for the circuit breaker to gate yet — wiring it in now would be complexity with nothing to verify.

Built on **FreqAI** (Freqtrade's built-in ML module), run via Docker like Modules A and B — reuses Freqtrade's feature pipeline, walk-forward retraining, and prediction plumbing instead of rebuilding it from scratch. **Note the image tag differs from Module B**: FreqAI needs `freqtradeorg/freqtrade:stable_freqai`, not the plain `:stable` image — the base image doesn't bundle scikit-learn/LightGBM/`datasieve` at all (confirmed by trying it and getting `ModuleNotFoundError` before switching tags).

## What it predicts

`set_freqai_targets()` labels each candle with the realized volatility (std of returns) over the *next* 12 candles. The model — a LightGBM regressor — learns to predict that forward volatility from current technicals plus one curated macro feature. Entry/exit rules in the strategy exist only to exercise the pipeline end-to-end (enter when the model expects calm, exit when it expects turbulence); they aren't the module's real decision logic, which is the volatility *read* itself, to be consumed by the orchestrator's allocation logic later.

## Feature set — restricted, per the project spec

- Price-derived technicals: RSI, ATR, ADX (multiple lookback periods), returns, volume, day-of-week/hour-of-day.
- **One curated macro feature: VIX**, merged in directly from Phase 2's `data_ingestion/macro_data` pipeline (`data/macro/yfinance/vix.parquet`, mounted read-only into the container) — the most directly relevant series in the curated list to a *volatility* gate specifically. Confirmed present in the trained model's actual feature list (`%-vix_close`), not just referenced in code.

Kept deliberately small, per the original instruction to restrict Module C's features to protect signal-to-noise and avoid overfitting a small crypto dataset. `freqai_utils.py` (the VIX-loading logic) is factored out of the strategy file specifically so it has zero Freqtrade dependency and can be unit tested locally without Docker — same pattern as Module B's `oos_split.py`/`candidate_ranking.py`.

## Verified running end-to-end

A real FreqAI backtest ran via Docker against 1 year of downloaded BTC/USDT 1h data:
```
docker compose --env-file ../../.env run --rm freqtrade backtesting \
  --config user_data/config.json --strategy VolatilityGateFreqAI \
  --freqaimodel LightGBMRegressor --timerange 20260601-20260817
```
Over the ~2.5-month window (train 60 days / retrain every 7 days), FreqAI **walk-forward retrained 11 separate model checkpoints**, confirmed by inspecting `user_data/models/` directly rather than just trusting the log output — genuine proof the retraining cadence works, not just that the command exits 0. One checkpoint's metadata was inspected directly to confirm `%-vix_close` was actually in the trained feature list.

Status: ✅ built and verified (Phase 6) — full pipeline (feature engineering, macro merge, walk-forward training, backtesting) runs end-to-end. Not yet wired into the orchestrator's circuit-breaker gating (that's Phase 7/8) or tuned for actual predictive performance.
