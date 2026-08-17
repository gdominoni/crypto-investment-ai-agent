"""Loaders for cached macro data, for consumers that aren't Freqtrade
strategies (see modules/module_c_volatility_ml/user_data/strategies/freqai_utils.py
for the Freqtrade-sibling-import equivalent, kept separate deliberately --
that copy is bound to a Docker-mounted path convention the local, bespoke
training pipeline doesn't use, so unifying them would trade a few
duplicated lines for awkward cross-boundary coupling).
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
