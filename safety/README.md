# Safety Kernel

Deterministic, hardcoded risk controls shared by all three modules. Nothing here is AI-controlled or AI-editable at runtime — the LLM can read state from this layer but cannot change its limits, disable stop-losses, or raise leverage.

Planned contents (Phase 3):
- Volatility circuit breaker (ATR-based threshold → force-liquidate to USDT).
- Macro-event blackout windows (FOMC, CPI, NFP) sourced from a scheduled macro calendar.
- Hardcoded position-size, leverage, and stop-loss/take-profit ceilings imported by every module.
