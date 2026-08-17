"""Curated symbol list for exchange market data ingestion (Phase 2).

Kept small and liquid: BTC and ETH cover Module A (cash & carry, where
funding-rate yield and depth matter most) and Module B (trend-following,
where liquid pairs minimize slippage). Expand deliberately, not by default.
"""

SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAMES = ["1h", "4h", "1d"]
OHLCV_LOOKBACK_DAYS = 365 * 3  # 3 years of history for backtesting
