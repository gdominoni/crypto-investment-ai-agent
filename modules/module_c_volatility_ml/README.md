# Module C — Volatility Gate / Machine Learning

Probabilistic ML signals (regime/volatility classification) gated by the deterministic circuit breaker in [`safety/`](../../safety/) — the ML model can suggest, never override, the hardcoded safety gate.

Built on **FreqAI** (Freqtrade's built-in ML module) rather than a fully bespoke training pipeline — reuses Freqtrade's feature pipeline, walk-forward retraining, and prediction plumbing instead of rebuilding it. The feature set is deliberately restricted to the curated series in [`data_ingestion/macro_data/config.py`](../../data_ingestion/macro_data/config.py) to keep signal-to-noise high and avoid overfitting on a small crypto dataset.

Status: not yet built (Phase 6).
