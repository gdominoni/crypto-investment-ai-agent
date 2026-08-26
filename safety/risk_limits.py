"""Enforcement functions -- always clamp toward safety, never expand it.

This is the only sanctioned way for any module (or the LLM-driven
orchestrator) to touch a risk parameter: every function here can only
return a value at or inside the hardcoded limits.py bounds, never outside
them, regardless of what's requested.
"""

from safety.limits import MAX_LEVERAGE, MAX_PORTFOLIO_DRAWDOWN_PCT, MAX_POSITION_SIZE_PCT


def clamp_leverage(requested_leverage: float) -> float:
    return max(0.0, min(requested_leverage, MAX_LEVERAGE))


def clamp_position_size_pct(requested_pct: float) -> float:
    return max(0.0, min(requested_pct, MAX_POSITION_SIZE_PCT))


def is_max_drawdown_breached(current_drawdown_pct: float) -> bool:
    """True once drawdown reaches the hardcoded ceiling -- caller (the
    portfolio/orchestrator layer, which tracks P&L) must then force-flatten
    to USDT. This function only answers the question; it holds no state."""
    return current_drawdown_pct >= MAX_PORTFOLIO_DRAWDOWN_PCT
