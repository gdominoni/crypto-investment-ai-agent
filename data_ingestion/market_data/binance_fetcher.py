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

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SPOT_DIR = DATA_DIR / "market/binance/spot"
FUNDING_DIR = DATA_DIR / "market/binance/funding"

COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LTCUSDT"]

# A safely-before-any-real-listing anchor for a from-scratch backfill (a
# new coin added to COINS, or this repo cloned somewhere `data/` isn't
# already populated) -- asking Binance for data "since" a date before a
# pair's real listing isn't an error, it just returns from wherever real
# history actually begins, so one fixed anchor works for every coin,
# matching this project's own "earliest available real data" convention
# used elsewhere (see replay/engine.py::advance()).
BACKFILL_SINCE = "2017-01-01"

# Hard cap on pagination loops per call -- real safety net against an
# unbounded loop if some future API quirk ever stops `since` from
# advancing, not expected to bind in practice (2500 pages x 1000 daily
# candles is centuries of history).
MAX_FETCH_PAGES = 2500


def _ccxt_symbol(symbol: str) -> str:
    """'BTCUSDT' -> 'BTC/USDT', ccxt's unified symbol notation."""
    return f"{symbol[:-4]}/USDT"


def update_ohlcv(symbol: str, timeframe: str = "1d") -> int:
    """Appends new candles since the last one already on disk,
    deduplicated by timestamp. Returns the number of rows added.

    Paginates until genuinely caught up, rather than one bounded call --
    verified live (2026-08-29): `since=None` does NOT return a coin's
    earliest history, it returns Binance's MOST RECENT candles (the last
    5 days, when tested with limit=5). A from-scratch backfill (a new
    coin, or this repo cloned somewhere `data/` isn't already populated)
    would otherwise silently truncate to ~1000 candles (~2.7 years of
    daily bars) of the *newest* history instead of this project's full
    multi-year depth -- exactly backwards from what every downstream
    walk-forward/pattern_significance computation assumes, with no error
    or warning. `since` now always anchors to BACKFILL_SINCE when there's
    no existing file (Binance simply returns from wherever a pair's real
    listing history begins if asked for an earlier date, so one fixed
    anchor is safe for every coin), and the fetch loop keeps paging until
    a page comes back short of `limit` (genuinely caught up to "now") or
    MAX_FETCH_PAGES is hit (safety net, not expected to bind)."""
    path = SPOT_DIR / f"{symbol}_{timeframe}.parquet"
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame(
        columns=["timestamp", "open", "high", "low", "close", "volume"])
    if len(existing):
        since = int(pd.to_datetime(existing["timestamp"], utc=True).max().timestamp() * 1000) + 1
    else:
        since = int(pd.Timestamp(BACKFILL_SINCE, tz="utc").timestamp() * 1000)

    # Imported HERE, not at module level. `ccxt` is an exchange client with a
    # large dependency tree, and importing it eagerly made it a hard requirement
    # of everything downstream: scheduler/weekly_revalidation.py imports this
    # module, scheduler/live_daemon.py imports that, so a test that merely
    # checks live_daemon's SOURCE for a scheduled job name could not run without
    # an exchange client installed. CI deliberately omits ccxt and three tests
    # failed on it -- the same failure this project already hit once and fixed
    # the same way (see .github/workflows/tests.yml's own note).
    import ccxt

    exchange = ccxt.binance({"enableRateLimit": True})
    candles: list = []
    for _ in range(MAX_FETCH_PAGES):
        page = exchange.fetch_ohlcv(_ccxt_symbol(symbol), timeframe=timeframe, since=since, limit=1000)
        if not page:
            break
        candles.extend(page)
        if len(page) < 1000:
            break  # short page -- caught up to "now", nothing more to fetch
        since = page[-1][0] + 1  # advance past the last candle received
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
    (a handful of new entries per call), which is why this paginates the
    same way update_ohlcv does: `since=None` returns Binance's MOST
    RECENT entries, not the earliest (verified live, same as OHLCV
    above), so a from-scratch backfill anchors to BACKFILL_SINCE and
    pages until caught up, rather than silently keeping only the last
    ~1000 entries (funding posts every 8h, so roughly the trailing year)
    of a symbol with no prior file. Coverage still genuinely varies by
    coin -- DOGE/ADA/LTC's perpetual futures listed later than
    BTC/ETH/BNB/XRP's, a real difference in what Binance actually has,
    not an artifact of this function's own call limit anymore."""
    path = FUNDING_DIR / f"{symbol}_funding.csv"
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(
        columns=["timestamp", "datetime", "symbol", "fundingRate"])
    if len(existing):
        since = int(existing["timestamp"].max()) + 1
    else:
        since = int(pd.Timestamp(BACKFILL_SINCE, tz="utc").timestamp() * 1000)

    import ccxt  # lazy -- see update_ohlcv

    exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
    history: list = []
    try:
        for _ in range(MAX_FETCH_PAGES):
            page = exchange.fetch_funding_rate_history(_ccxt_symbol(symbol), since=since, limit=1000)
            if not page:
                break
            history.extend(page)
            if len(page) < 1000:
                break  # short page -- caught up to "now"
            since = page[-1]["timestamp"] + 1
    except ccxt.BaseError as e:
        print(f"funding fetch failed for {symbol}: {e}")
        if not history:
            return 0
        # Partial pages already fetched before the error are still worth
        # keeping (real data, isolated the same way update_all() isolates
        # one coin's failure from the others) -- fall through to save them.
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
