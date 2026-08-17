"""Tests for the deterministic circuit breaker (safety/circuit_breaker.py)."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from safety.circuit_breaker import check_macro_blackout, check_volatility_spike, evaluate_circuit_breaker
from safety.macro_calendar import MacroEvent


def _flat_ohlc(periods: int, price: float = 100.0, high_low_spread: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "high": [price + high_low_spread] * periods,
            "low": [price - high_low_spread] * periods,
            "close": [price] * periods,
        }
    )


def test_no_spike_on_stable_volatility():
    ohlc = _flat_ohlc(periods=120)
    assert check_volatility_spike(ohlc) is False


def test_spike_triggers_when_recent_range_widens():
    ohlc = _flat_ohlc(periods=120)
    ohlc.loc[110:, "high"] = 100 + 20
    ohlc.loc[110:, "low"] = 100 - 20
    assert check_volatility_spike(ohlc) is True


def test_insufficient_history_fails_safe_to_no_trigger():
    ohlc = _flat_ohlc(periods=10)
    assert check_volatility_spike(ohlc) is False


def test_macro_blackout_triggers_inside_window():
    event_time = datetime(2026, 9, 16, 18, 0, tzinfo=timezone.utc)
    event = MacroEvent("Test FOMC", "fomc_decision", event_time)
    now = event_time - timedelta(minutes=10)
    assert check_macro_blackout(now, [event]) == event


def test_macro_blackout_silent_outside_window():
    event_time = datetime(2026, 9, 16, 18, 0, tzinfo=timezone.utc)
    event = MacroEvent("Test FOMC", "fomc_decision", event_time)
    now = event_time - timedelta(hours=5)
    assert check_macro_blackout(now, [event]) is None


def test_evaluate_combines_both_checks():
    ohlc = _flat_ohlc(periods=120)
    now = datetime(2026, 9, 16, 18, 0, tzinfo=timezone.utc)
    decision = evaluate_circuit_breaker(now, ohlc, macro_events=[])
    assert decision.should_liquidate is False
    assert decision.reasons == []


def test_evaluate_flags_macro_blackout_reason():
    ohlc = _flat_ohlc(periods=120)
    event_time = datetime(2026, 9, 16, 18, 0, tzinfo=timezone.utc)
    event = MacroEvent("Test FOMC", "fomc_decision", event_time)
    now = event_time
    decision = evaluate_circuit_breaker(now, ohlc, macro_events=[event])
    assert decision.should_liquidate is True
    assert decision.reasons == ["macro_blackout:Test FOMC"]
