"""Trigger definitions for the statistical baseline battery: C1, C2, and
C6 -- the three candidates selected, out of the prior research's larger
set, as ideas worth re-deriving clean and continuing to test here (the
numbering is inherited from that prior labeling, not sequential by
design -- C3 was never one of the three carried forward). C4
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

import pandas as pd

from .data_loading import load_daily, load_funding, zscore
from .macro_calendar import macro_release_days


def trend_efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    net_change = (close - close.shift(window)).abs()
    path_length = close.diff().abs().rolling(window).sum()
    return (net_change / path_length).clip(upper=1.0)


def compute_triggers(daily: pd.DataFrame, funding: pd.Series | None = None, scale: int = 1) -> pd.DataFrame:
    """Pure function of a daily OHLCV frame (+ optional funding series) --
    called identically by the research battery (`run_battery.py`, on a
    fully historical frame) and the live Freqtrade strategy (on its own
    `populate_indicators` dataframe), so there is exactly one
    implementation of each trigger, never two that could drift apart.

    `scale`: every window below is a bar count, in DAYS by default
    (`scale=1`). Passed `scale=24` against an HOURLY frame instead, the
    same day-defined windows (e.g. "20-day efficiency ratio") get
    reinterpreted as their hour-equivalent (480 hours) -- a deliberate,
    documented approximation used only for LIVE trigger detection (never
    the backtest), see docs/case_study/methodology-decisions.md. C2
    ('is_macro_day') stays calendar-day gated regardless of `scale` --
    macro releases are only known to day granularity in this project's
    calendar, there's no hourly refinement to reinterpret.

    `funding=None` (no funding-rate history available for this coin)
    degrades c1_long/c1_short to always-False rather than raising --
    deliberate, so a coin missing this one data source doesn't break
    every other trigger for it. See `data_ingestion/market_data/`'s
    fetcher for keeping funding-rate coverage current across the full
    coin universe."""
    idx = daily.index
    out = pd.DataFrame(index=idx)
    close, volume = daily["close"], daily["volume"]
    w20, w5, w30 = 20 * scale, 5 * scale, 30 * scale

    macro_days = macro_release_days()
    bar_range = (daily["high"] - daily["low"]) / close
    range_avg = bar_range.rolling(w20, min_periods=max(w20 // 2, 1)).mean()
    is_macro_day = pd.Series(idx.floor("D").isin(macro_days) if scale > 1 else idx.isin(macro_days), index=idx)
    breakout = bar_range > 1.5 * range_avg
    out["c2_long"] = is_macro_day & breakout & (close < daily["open"])
    out["c2_short"] = is_macro_day & breakout & (close > daily["open"])

    er = trend_efficiency_ratio(close, w20)
    vol_surge = volume > 1.8 * volume.rolling(w20, min_periods=max(w20 // 2, 1)).mean()
    prior_ret = close.pct_change(w5)
    trend_base = (er > 0.40) & vol_surge
    out["c6_long"] = trend_base & (prior_ret > 0)
    out["c6_short"] = trend_base & (prior_ret < 0)

    if funding is not None:
        fz = zscore(funding.reindex(idx).ffill(), window=w30)
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

# The exact numeric thresholds behind each TRIGGER_DESCRIPTIONS entry --
# must mirror compute_triggers() above precisely (kept as prose there,
# not re-derived here, since a duplicated number is a number that can
# silently drift out of sync with the real logic). Powers Telegram's
# `/details`/`/replay_details` commands: TRIGGER_DESCRIPTIONS alone
# leaves a reader unable to answer "elevated funding rate -- how
# elevated, exactly?" with an actual number.
TRIGGER_NUMERIC_DEFINITIONS = {
    "c1": "30-day funding-rate z-score below -2.0 (long) or above +2.0 (short) -- extreme relative to that coin's own trailing 30-day funding history, not a fixed absolute rate.",
    "c2": "on an FOMC/CPI release day: that day's own high-low range exceeds 1.5x its trailing 20-day average range, AND the day closes below its open (long) or above its open (short).",
    "c6": "20-day Kaufman efficiency ratio above 0.40, AND that day's volume above 1.8x its trailing 20-day average; fires as 'long' when the 5-day price change is positive, 'short' when it's negative.",
}
