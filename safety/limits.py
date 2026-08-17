"""Hardcoded risk limits (Phase 3).

These are the system's outer bounds. No module, no LLM output, and no
Telegram command can raise these -- they are plain Python constants, not
config that anything writes to at runtime. Changing them requires editing
this file and redeploying, the same as any other code change.
"""

MAX_LEVERAGE = 3.0
MAX_POSITION_SIZE_PCT = 0.25       # max fraction of a module's capital in one position
MAX_PORTFOLIO_DRAWDOWN_PCT = 0.15  # breach -> force flatten to USDT
DEFAULT_STOP_LOSS_PCT = 0.05
DEFAULT_TAKE_PROFIT_PCT = 0.10

# Volatility circuit breaker (safety/circuit_breaker.py)
ATR_LOOKBACK_PERIODS = 14
ATR_BASELINE_WINDOW = 90    # rolling window used as the "normal" baseline
ATR_SPIKE_MULTIPLIER = 2.5  # current ATR > multiplier x baseline -> trigger

# Macro-event blackout window (safety/macro_calendar.py)
MACRO_BLACKOUT_MINUTES_BEFORE = 30
MACRO_BLACKOUT_MINUTES_AFTER = 60
