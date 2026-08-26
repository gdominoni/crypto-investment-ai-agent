"""Deterministic circuit breaker: volatility spikes and macro-event
blackout windows force the portfolio to 100% USDT. Nothing here is
probabilistic or LLM-driven -- pure threshold logic against limits.py
constants and a hardcoded macro calendar.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from safety.limits import (
    ATR_BASELINE_WINDOW,
    ATR_LOOKBACK_PERIODS,
    ATR_SPIKE_MULTIPLIER,
    MACRO_BLACKOUT_MINUTES_AFTER,
    MACRO_BLACKOUT_MINUTES_BEFORE,
)
from safety.macro_calendar import MacroEvent


@dataclass(frozen=True)
class CircuitBreakerDecision:
    should_liquidate: bool
    reasons: list[str]


def compute_atr(ohlc: pd.DataFrame, periods: int = ATR_LOOKBACK_PERIODS) -> pd.Series:
    high, low, close = ohlc["high"], ohlc["low"], ohlc["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.rolling(periods).mean()


def check_volatility_spike(
    ohlc: pd.DataFrame,
    baseline_window: int = ATR_BASELINE_WINDOW,
    spike_multiplier: float = ATR_SPIKE_MULTIPLIER,
) -> bool:
    atr = compute_atr(ohlc).dropna()
    if len(atr) < baseline_window + 1:
        # Not enough history to judge what's "normal" -- fail safe to
        # "no trigger" rather than assuming a spike from insufficient data.
        return False
    baseline = atr.iloc[-(baseline_window + 1):-1].median()
    current = atr.iloc[-1]
    if baseline <= 0:
        return False
    return bool(current > baseline * spike_multiplier)


def check_macro_blackout(
    now: datetime,
    macro_events: list[MacroEvent],
    minutes_before: int = MACRO_BLACKOUT_MINUTES_BEFORE,
    minutes_after: int = MACRO_BLACKOUT_MINUTES_AFTER,
) -> MacroEvent | None:
    for event in macro_events:
        window_start = event.timestamp_utc - timedelta(minutes=minutes_before)
        window_end = event.timestamp_utc + timedelta(minutes=minutes_after)
        if window_start <= now <= window_end:
            return event
    return None


def evaluate_circuit_breaker(
    now: datetime,
    ohlc: pd.DataFrame,
    macro_events: list[MacroEvent],
) -> CircuitBreakerDecision:
    reasons = []
    if check_volatility_spike(ohlc):
        reasons.append("volatility_spike")
    blackout_event = check_macro_blackout(now, macro_events)
    if blackout_event:
        reasons.append(f"macro_blackout:{blackout_event.label}")
    return CircuitBreakerDecision(should_liquidate=bool(reasons), reasons=reasons)
