# Macro Data Ingestion

Cross-asset and macroeconomic features used for regime detection, Module C (FreqAI) feature engineering, and the macro-event blackout window in the safety kernel. The series list is intentionally short — see [`config.py`](config.py) — to keep signal-to-noise high and avoid overfitting.

**Financial market tickers** (`yfinance_fetcher.py`):
| Ticker | Series | Role |
|---|---|---|
| `^GSPC` | S&P 500 | Risk-on/tech correlation proxy |
| `^IXIC` | Nasdaq Composite | Risk-on/tech correlation proxy |
| `^VIX` | CBOE Volatility Index | Fear gauge / risk-off detection |
| `GC=F` | Gold futures | Macro hedge / risk-off flows |
| `CL=F` | Crude oil futures | Macro/inflation proxy |

**Macroeconomic indicators** (`fred_fetcher.py`, requires `FRED_API_KEY`):
| Series ID | Series | Role |
|---|---|---|
| `FEDFUNDS` | Federal Funds Effective Rate | Monetary policy |
| `UNRATE` | Unemployment Rate | Labor market |
| `CPIAUCSL` | Consumer Price Index | Inflation |
| `INDPRO` | Industrial Production Index | Real economy |

Run locally with:
```
python -m data_ingestion.macro_data.yfinance_fetcher
python -m data_ingestion.macro_data.fred_fetcher
```
Output lands in `data/macro/yfinance/` and `data/macro/fred/` (gitignored).

Status: ✅ built (Phase 2).
