"""Tests for safety/risk_limits.py -- every function must clamp toward
safety, never expand past the hardcoded limits."""

from safety.limits import MAX_LEVERAGE, MAX_PORTFOLIO_DRAWDOWN_PCT, MAX_POSITION_SIZE_PCT
from safety.risk_limits import clamp_leverage, clamp_position_size_pct, is_max_drawdown_breached


def test_clamp_leverage_caps_at_max():
    assert clamp_leverage(MAX_LEVERAGE + 50) == MAX_LEVERAGE


def test_clamp_leverage_passes_through_lower_requests():
    assert clamp_leverage(1.0) == 1.0


def test_clamp_leverage_floors_at_zero():
    assert clamp_leverage(-5) == 0.0


def test_clamp_position_size_caps_at_max():
    assert clamp_position_size_pct(MAX_POSITION_SIZE_PCT + 0.5) == MAX_POSITION_SIZE_PCT


def test_drawdown_breach_detection():
    assert is_max_drawdown_breached(MAX_PORTFOLIO_DRAWDOWN_PCT) is True
    assert is_max_drawdown_breached(MAX_PORTFOLIO_DRAWDOWN_PCT - 0.01) is False
