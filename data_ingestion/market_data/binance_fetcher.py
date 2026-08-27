"""Keeps this project's local OHLCV and funding-rate snapshots current by
pulling new candles/funding entries from Binance's public API (via ccxt
-- no API key needed for public market data) and appending them to the
existing parquet/CSV files in `data/`.

Without this, every path that reads through `candidates/data_loading.py`
(the research battery, novel-condition testing, live shock detection)
would be reading a frozen, one-time snapshot forever -- this is what
makes "weekly re-validation against live data" and "real-time shock
detection" actually true rather than aspirational. Incremental by
design: each call fetches only what's newer than the last row already on
disk, so re-running this often is cheap and safe.

Run as:
    python -m data_ingestion.market_data.binance_fetcher
"""
from __future__ import annotations

from pathlib import Path

import ccxt
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SPOT_DIR = DATA_DIR / "market/binance/spot"
FUNDING_DIR = DATA_DIR / "market/binance/funding"

COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LTCUSDT"]


def _ccxt_symbol(symbol: str) -> str:
    """'BTCUSDT' -> 'BTC/USDT', ccxt's unified symbol notation."""
    return f"{symbol[:-4]}/USDT"


def update_ohlcv(symbol: str, timeframe: str = "1d") -> int:
    """Appends new candles since the last one already on disk,
    deduplicated by timestamp. Returns the number of rows added."""
    path = SPOT_DIR / f"{symbol}_{timeframe}.parquet"
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame(
        columns=["timestamp", "open", "high", "low", "close", "volume"])
    since = None
    if len(existing):
        since = int(pd.to_datetime(existing["timestamp"], utc=True).max().timestamp() * 1000)

    exchange = ccxt.binance({"enableRateLimit": True})
    candles = exchange.fetch_ohlcv(_ccxt_symbol(symbol), timeframe=timeframe, since=since, limit=1000)
    if not candles:
        return 0

    new_df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms", utc=True)
    existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True) if len(existing) else existing["timestamp"]
    combined = (pd.concat([existing, new_df], ignore_index=True)
                .drop_duplicates(subset="timestamp", keep="last")
                .sort_values("timestamp").reset_index(drop=True))

    added = len(combined) - len(existing)
    SPOT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    return added


def update_funding(symbol: str) -> int:
    """Appends new funding-rate entries since the last one on disk.
    Binance's public funding-rate-history endpoint caps each call at
    ~1000 entries -- fine for keeping an already-backfilled file current
    (a handful of new entries per call), but a symbol with NO prior file
    only gets ~1000 entries' worth of history back (funding posts every
    8h, so roughly the trailing year), not the multi-year depth the
    already-backfilled coins have. Real, honestly-reported partial
    coverage, not silently assumed complete."""
    path = FUNDING_DIR / f"{symbol}_funding.csv"
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(
        columns=["timestamp", "datetime", "symbol", "fundingRate"])
    since = int(existing["timestamp"].max()) + 1 if len(existing) else None

    exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
    try:
        history = exchange.fetch_funding_rate_history(_ccxt_symbol(symbol), since=since, limit=1000)
    except ccxt.BaseError as e:
        print(f"funding fetch failed for {symbol}: {e}")
        return 0
    if not history:
        return 0

    new_df = pd.DataFrame([
        {"timestamp": h["timestamp"], "datetime": h["datetime"], "symbol": h["symbol"], "fundingRate": h["fundingRate"]}
        for h in history
    ])
    combined = (pd.concat([existing, new_df], ignore_index=True)
                .drop_duplicates(subset="timestamp", keep="last")
                .sort_values("timestamp").reset_index(drop=True))

    added = len(combined) - len(existing)
    FUNDING_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return added


def update_all(coins: list[str] | None = None) -> dict[str, dict[str, int | None]]:
    """One coin's fetch failure must not cost every OTHER coin's update --
    each coin's OHLCV and funding fetch is isolated so a single network
    blip or a bad response for e.g. coin #3 of 7 doesn't silently skip
    coins #4-7 for the whole week. `None` in the report (rather than 0)
    marks "failed", distinct from 0 meaning "fetched fine, nothing new"."""
    report = {}
    for symbol in (coins or COINS):
        try:
            ohlcv_added = update_ohlcv(symbol)
        except Exception as e:
            print(f"OHLCV fetch failed for {symbol}, skipping: {e}")
            ohlcv_added = None
        try:
            funding_added = update_funding(symbol)
        except Exception as e:
            print(f"funding fetch failed for {symbol}, skipping: {e}")
            funding_added = None
        report[symbol] = {"ohlcv_added": ohlcv_added, "funding_added": funding_added}
        ohlcv_msg = f"+{ohlcv_added} daily candle(s)" if ohlcv_added is not None else "OHLCV FAILED"
        funding_msg = f"+{funding_added} funding entry(ies)" if funding_added is not None else "funding FAILED"
        print(f"{symbol}: {ohlcv_msg}, {funding_msg}")
    return report


if __name__ == "__main__":
    update_all()
