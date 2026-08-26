"""Trigger definitions for the statistical baseline battery. Two
candidates ported forward as ideas worth continuing to test (C4
stablecoin-supply and C5 basis-unwind are deliberately NOT included here
yet -- their required data source, on-chain stablecoin supply, isn't
wired into this project's data layer; add them once it is, rather than
approximate with something weaker):

  C1 -- funding-rate crowding: an extreme funding rate (perpetual futures
        positioning) split by sign, betting on a squeeze in the opposite
        direction.
  C2 -- post-macro-release reaction: FOMC/CPI day with an unusually wide
        range, betting the day's own initial close direction reverses.
  C6 -- efficiency-ratio trend: a high Kaufman efficiency ratio (a clean,
        low-noise trend) paired with a volume surge, split by the
        direction already in motion.

Every trigger here is defined on bar N's own close/range -- by
construction, `methodology.build_events()` enters at bar N+1's open, so
no trigger definition needs its own leak-avoidance logic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loading import load_daily, load_funding, zscore
from .macro_calendar import macro_release_days


def trend_efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    net_change = (close - close.shift(window)).abs()
    path_length = close.diff().abs().rolling(window).sum()
    return (net_change / path_length).clip(upper=1.0)


def compute_triggers(daily: pd.DataFrame, funding: pd.Series | None = None) -> pd.DataFrame:
    """Pure function of a daily OHLCV frame (+ optional funding series) --
    called identically by the research battery (`run_battery.py`, on a
    fully historical frame) and the live Freqtrade strategy (on its own
    `populate_indicators` dataframe), so there is exactly one
    implementation of each trigger, never two that could drift apart."""
    idx = daily.index
    out = pd.DataFrame(index=idx)
    close, volume = daily["close"], daily["volume"]

    macro_days = macro_release_days()
    daily_range = (daily["high"] - daily["low"]) / close
    range_avg20 = daily_range.rolling(20, min_periods=10).mean()
    is_macro_day = pd.Series(idx.isin(macro_days), index=idx)
    breakout = daily_range > 1.5 * range_avg20
    out["c2_long"] = is_macro_day & breakout & (close < daily["open"])
    out["c2_short"] = is_macro_day & breakout & (close > daily["open"])

    er20 = trend_efficiency_ratio(close, 20)
    vol_surge = volume > 1.8 * volume.rolling(20, min_periods=10).mean()
    prior_ret = close.pct_change(5)
    trend_base = (er20 > 0.40) & vol_surge
    out["c6_long"] = trend_base & (prior_ret > 0)
    out["c6_short"] = trend_base & (prior_ret < 0)

    if funding is not None:
        fz = zscore(funding.reindex(idx).ffill(), window=30)
        out["c1_long"] = fz < -2.0
        out["c1_short"] = fz > 2.0
    else:
        out["c1_long"], out["c1_short"] = False, False

    return out.fillna(False)


def build_triggers(symbol: str) -> pd.DataFrame:
    return compute_triggers(load_daily(symbol), load_funding(symbol))


CANDIDATE_DIRECTIONS = {
    "c1_long": "long", "c1_short": "short",
    "c2_long": "long", "c2_short": "short",
    "c6_long": "long", "c6_short": "short",
}

# Grounds any LLM call that reasons about a static candidate by name --
# e.g. haiku_sonnet_pipeline.sonnet_prune_advice() -- in what the trigger
# ACTUALLY tests, rather than letting the model infer a mechanism from
# the label alone (a label like "c1_long" carries no information about
# funding-rate crowding on its own).
TRIGGER_DESCRIPTIONS = {
    "c1": "funding-rate crowding: an extreme perpetual-futures funding rate, betting on a squeeze in the opposite direction",
    "c2": "post-macro-release reaction: FOMC/CPI day with an unusually wide range, betting the day's own initial close direction reverses",
    "c6": "efficiency-ratio trend: a high Kaufman efficiency ratio (a clean, low-noise trend) paired with a volume surge, in the direction already in motion",
}
