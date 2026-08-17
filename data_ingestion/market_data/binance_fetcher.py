"""Fetches spot OHLCV, futures OHLCV, and funding-rate history from Binance
via ccxt's public endpoints -- no API key required for market data, even
though live execution (Phase 4/5) will connect to Binance testnet.
Historical data is pulled from mainnet since testnet history is sparse.

Local-only. Run as:
    python -m data_ingestion.market_data.binance_fetcher
"""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import pandas as pd

from data_ingestion.market_data.config import OHLCV_LOOKBACK_DAYS, SYMBOLS, TIMEFRAMES

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "market" / "binance"


def _fetch_ohlcv_history(exchange: ccxt.Exchange, symbol: str, timeframe: str, since_ms: int) -> pd.DataFrame:
    all_candles = []
    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=1000)
        if not candles:
            break
        all_candles.extend(candles)
        since_ms = candles[-1][0] + 1
        if len(candles) < 1000:
            break
        time.sleep(exchange.rateLimit / 1000)
    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.drop_duplicates(subset="timestamp").reset_index(drop=True)


def fetch_spot_ohlcv(lookback_days: int = OHLCV_LOOKBACK_DAYS) -> dict[str, pd.DataFrame]:
    exchange = ccxt.binance({"enableRateLimit": True})
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000)
    out_dir = DATA_DIR / "spot"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            df = _fetch_ohlcv_history(exchange, symbol, timeframe, since_ms)
            name = f"{symbol.replace('/', '')}_{timeframe}"
            df.to_parquet(out_dir / f"{name}.parquet")
            results[name] = df
    return results


def fetch_futures_ohlcv_and_funding(
    lookback_days: int = OHLCV_LOOKBACK_DAYS,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp() * 1000)
    ohlcv_dir = DATA_DIR / "futures"
    funding_dir = DATA_DIR / "funding"
    ohlcv_dir.mkdir(parents=True, exist_ok=True)
    funding_dir.mkdir(parents=True, exist_ok=True)

    ohlcv_results = {}
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            df = _fetch_ohlcv_history(exchange, symbol, timeframe, since_ms)
            name = f"{symbol.replace('/', '')}_{timeframe}"
            df.to_parquet(ohlcv_dir / f"{name}.parquet")
            ohlcv_results[name] = df

    funding_results = {}
    for symbol in SYMBOLS:
        history = []
        fetch_since = since_ms
        while True:
            batch = exchange.fetch_funding_rate_history(symbol, since=fetch_since, limit=1000)
            if not batch:
                break
            history.extend(batch)
            fetch_since = batch[-1]["timestamp"] + 1
            if len(batch) < 1000:
                break
            time.sleep(exchange.rateLimit / 1000)
        df = pd.DataFrame(history)[["timestamp", "datetime", "symbol", "fundingRate"]]
        name = symbol.replace("/", "")
        df.to_csv(funding_dir / f"{name}_funding.csv", index=False)
        funding_results[name] = df

    return ohlcv_results, funding_results


if __name__ == "__main__":
    spot = fetch_spot_ohlcv()
    for name, df in spot.items():
        print(f"spot/{name}: {len(df)} rows")

    futures, funding = fetch_futures_ohlcv_and_funding()
    for name, df in futures.items():
        print(f"futures/{name}: {len(df)} rows")
    for name, df in funding.items():
        print(f"funding/{name}: {len(df)} rows")
