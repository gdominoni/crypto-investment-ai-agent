# Phase 2 (part 1): Macro & Cross-Asset Data Ingestion

**Goal:** ingest a small, curated set of macro/cross-asset features for regime detection and Module C (FreqAI) — before touching exchange market data or news sentiment.

## The prompt

The human director sent an updated Phase 2 spec locking down the exact feature set, rather than leaving it open-ended:
- 5 `yfinance` tickers: S&P 500, Nasdaq Composite, VIX, Gold futures, Crude Oil futures.
- 4 FRED series: Fed Funds Rate, Unemployment Rate, CPI, Industrial Production.
- An explicit constraint: keep the feature list restricted to these to protect signal-to-noise and avoid overfitting Module C.
- New detail: Module C is built on **FreqAI** (Freqtrade's built-in ML module), not a fully bespoke ML pipeline.

## Decisions made in this phase

1. **Single source of truth for the feature list.** Both ticker sets live in `data_ingestion/macro_data/config.py` as plain dicts, imported by the fetchers. This makes the "restricted feature set" constraint enforceable in code, not just in a doc — adding a series means touching one file, and it's obvious at a glance what's in scope.
2. **Module C re-scoped to FreqAI.** Since the spec named FreqAI directly, `modules/module_c_volatility_ml/` will wrap Freqtrade's FreqAI rather than a custom sklearn/lightgbm training loop — reuses FreqAI's feature pipeline and walk-forward retraining instead of rebuilding it.
3. **Freqtrade and Hummingbot removed from `requirements-server.txt`.** Attempting to `pip install` them during this phase's smoke test failed: Hummingbot pins build dependencies (`cython==3.0.0a10`, `numpy==1.26.4`) incompatible with Python 3.13, and neither project is officially distributed via plain pip. Both will be added as Docker services in Phase 9, matching the deployment architecture instead of fighting pip version pins now — premature to solve in Phase 2.
4. **Parquet for yfinance data, CSV for FRED data.** yfinance data is daily OHLCV (multi-column, benefits from parquet's columnar compression); FRED series are single-column and low-frequency (monthly/weekly) where CSV's readability outweighs any storage benefit from parquet.

## What got built and verified

- `data_ingestion/macro_data/config.py` — the restricted feature list.
- `data_ingestion/macro_data/yfinance_fetcher.py` — fetches all 5 tickers, writes parquet to `data/macro/yfinance/`. Smoke-tested locally: all 5 tickers returned ~2,900 rows of daily history and wrote successfully.
- `data_ingestion/macro_data/fred_fetcher.py` — fetches all 4 series, writes CSV to `data/macro/fred/`. Smoke-tested for correct failure behavior: raises a clear `RuntimeError` pointing at `.env.example` when `FRED_API_KEY` isn't set (expected, since the key hasn't been provisioned yet).
- Added `pyarrow` to `requirements-local.txt` (needed for parquet support, missing from the initial dependency list — caught by actually running the fetcher rather than assuming it would work).

## Still pending

- `FRED_API_KEY` needs to be added to the human director's local `.env` before `fred_fetcher.py` can pull real data (code path is verified, live pull is not yet).
- Exchange market data (Binance via `ccxt`) and news sentiment (CryptoCompare + RSS) ingestion — the other two legs of Phase 2 — not yet built.
