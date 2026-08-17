"""Pure helper functions for volatility_gate_freqai.py, kept free of any
freqtrade import so they can be unit tested locally without Docker (see
tests/test_freqai_utils.py, which loads this file directly by path).

Freqtrade adds user_data/strategies/ to sys.path when loading strategies,
so `from freqai_utils import load_vix` works as a sibling import from the
strategy file at runtime.
"""

from pathlib import Path

import pandas as pd


def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance saves single-ticker downloads with MultiIndex columns like
    ('Close', '^VIX') -- flatten to just 'Close', 'High', etc."""
    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def load_vix(macro_data_dir: Path) -> pd.DataFrame:
    vix = pd.read_parquet(macro_data_dir / "yfinance" / "vix.parquet")
    vix = flatten_yfinance_columns(vix)
    vix = vix.reset_index().rename(columns={"Date": "date", "Close": "vix_close"})[["date", "vix_close"]]
    vix["date"] = pd.to_datetime(vix["date"], utc=True)
    return vix.sort_values("date")
