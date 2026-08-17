"""Fetches curated macro series via the FRED API.

Local-only (see requirements-local.txt). Requires FRED_API_KEY in .env
(free key: https://fred.stlouisfed.org/docs/api/api_key.html). Run as:

    python -m data_ingestion.macro_data.fred_fetcher
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

from data_ingestion.macro_data.config import FRED_SERIES

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "macro" / "fred"


def fetch_fred_series(start: str = "2015-01-01") -> dict[str, pd.Series]:
    load_dotenv()
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY is not set -- add it to .env (see .env.example)")

    fred = Fred(api_key=api_key)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for series_id, name in FRED_SERIES.items():
        series = fred.get_series(series_id, observation_start=start)
        series.rename(name).to_frame().to_csv(DATA_DIR / f"{name}.csv")
        results[name] = series
    return results


if __name__ == "__main__":
    fetched = fetch_fred_series()
    for name, series in fetched.items():
        print(f"{name}: {len(series)} observations -> {DATA_DIR / f'{name}.csv'}")
