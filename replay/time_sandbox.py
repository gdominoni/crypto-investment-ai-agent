"""As-of data access for the historical replay -- every function here
takes a cutoff date and returns ONLY what was actually knowable by that
date, never a peek at what came after. This is the one piece the whole
replay's honesty rests on: reusing candidates/data_loading.py's readers
but slicing OHLCV/funding by bar date. Vintage macro data reading itself
now lives in candidates/macro_vintage.py, shared with production
(`as_of=None` there means "as of right now"); re-exported here so
existing `from replay.time_sandbox import ...` call sites in this
package are unaffected.
"""
from __future__ import annotations

import pandas as pd

from candidates.data_loading import load_daily, load_funding
from candidates.macro_vintage import VINTAGE_DIR, latest_release_with_prior, release_dates, vintage_releases_as_of  # noqa: F401


def daily_as_of(symbol: str, as_of: pd.Timestamp) -> pd.DataFrame:
    df = load_daily(symbol)
    return df.loc[:as_of]


def funding_as_of(symbol: str, as_of: pd.Timestamp) -> pd.Series | None:
    series = load_funding(symbol)
    return series.loc[:as_of] if series is not None else None
