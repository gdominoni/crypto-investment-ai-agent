"""Loads the raw OHLCV/funding series each candidate needs. All frames are
returned tz-naive UTC, indexed by bar timestamp -- comparisons against
the tz-naive macro calendar (`macro_calendar.py`) are safe by construction,
not by remembering to localize at the call site."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_daily(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "market/binance/spot" / f"{symbol}_1d.parquet")
    df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None).dt.floor("D")
    return df.set_index("date").sort_index().rename(columns=str.lower)


def load_hourly(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "market/binance/spot" / f"{symbol}_1h.parquet")
    df["datetime"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    return df.set_index("datetime").sort_index().rename(columns=str.lower)


def load_funding(symbol: str) -> pd.Series | None:
    path = DATA_DIR / "market/binance/funding" / f"{symbol}_funding.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None).dt.floor("D")
    daily = df.groupby("date")["fundingRate"].mean().astype(float)
    return daily.sort_index()


def zscore(series: pd.Series, window: int = 30) -> pd.Series:
    mu = series.rolling(window, min_periods=window // 2).mean()
    sd = series.rolling(window, min_periods=window // 2).std()
    return (series - mu) / sd
