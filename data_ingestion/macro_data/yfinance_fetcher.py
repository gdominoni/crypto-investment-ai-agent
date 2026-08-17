"""Fetches equity index, volatility, and commodity data via yfinance.

Local-only (see requirements-local.txt) -- these feed backtesting and
Module C feature engineering, not live server decisions. Run as:

    python -m data_ingestion.macro_data.yfinance_fetcher
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

from data_ingestion.macro_data.config import YFINANCE_TICKERS

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "macro" / "yfinance"


def fetch_yfinance_data(start: str = "2015-01-01", end: str | None = None) -> dict[str, pd.DataFrame]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for ticker, name in YFINANCE_TICKERS.items():
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            raise RuntimeError(f"yfinance returned no data for {ticker} ({name})")
        df.to_parquet(DATA_DIR / f"{name}.parquet")
        results[name] = df
    return results


if __name__ == "__main__":
    fetched = fetch_yfinance_data()
    for name, df in fetched.items():
        print(f"{name}: {len(df)} rows -> {DATA_DIR / f'{name}.parquet'}")
