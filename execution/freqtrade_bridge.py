"""Bridges this project's own OHLCV data (data/market/binance/spot/) into
Freqtrade's own feather format, so its (independent, third-party) hyperopt
engine can run against the exact same real price history the rest of
this project uses -- no re-download, no separate data source. Local-only,
periodic use (see execution/hyperopt_runner.py): never wired into live
execution, never gates acceptance. See
docs/case_study/methodology-decisions.md for why this cross-check exists
alongside this project's own walk-forward grid search.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from candidates.data_loading import load_daily
from candidates.run_battery import COINS

# freqtrade itself is imported lazily, inside sync_data() below -- NOT at
# module level. This module is imported (for build_config()/pair_for(),
# neither of which needs freqtrade) by callers like replay/engine.py that
# must not gain a hard dependency on the heavy freqtrade package just to
# show a hyperopt cross-check line; only actually calling sync_data()
# requires it installed.

FT_USERDIR = Path(__file__).resolve().parent / "freqtrade_userdir"
FT_DATADIR = FT_USERDIR / "data" / "binance"


def pair_for(coin: str) -> str:
    """'BTCUSDT' -> 'BTC/USDT:USDT' -- the futures-pair spelling this
    project's own live strategy already uses (can_short=True,
    trading_mode='futures')."""
    base = coin[:-4] if coin.endswith("USDT") else coin
    return f"{base}/USDT:USDT"


def sync_data(coins: list[str] | None = None, timeframe: str = "1d") -> None:
    """Converts every coin's current OHLCV (kept fresh by
    data_ingestion/market_data/binance_fetcher.py, same as everywhere
    else in this project) into Freqtrade's feather format. Idempotent --
    overwrites, safe to call before every hyperopt run so it's never
    stale relative to what the rest of the project sees."""
    from freqtrade.data.history.datahandlers.featherdatahandler import FeatherDataHandler
    from freqtrade.enums import CandleType

    FT_DATADIR.mkdir(parents=True, exist_ok=True)
    handler = FeatherDataHandler(FT_DATADIR)
    for coin in coins or COINS:
        df = load_daily(coin).reset_index()[["date", "open", "high", "low", "close", "volume"]]
        df["date"] = pd.to_datetime(df["date"], utc=True)
        handler.ohlcv_store(pair_for(coin), timeframe, df, CandleType.FUTURES)


def build_config(coins: list[str] | None = None) -> dict:
    """Minimal config for LOCAL, dry-run-only backtesting/hyperopt --
    never a live/real exchange connection. `use_order_book: true` on
    both pricing blocks is required for Binance specifically: Freqtrade
    hardcodes `tickers_have_price=False` for this exchange, so ticker-
    based pricing fails its own config validation without this (a real,
    non-obvious issue hit and confirmed while building this)."""
    return {
        "max_open_trades": 10,
        "stake_currency": "USDT",
        "stake_amount": 100,
        "tradable_balance_ratio": 1,
        "dry_run": True,
        "trading_mode": "futures",
        "margin_mode": "isolated",
        "timeframe": "1d",
        "dataformat_ohlcv": "feather",
        "exchange": {
            "name": "binance",
            "pair_whitelist": [pair_for(c) for c in (coins or COINS)],
            "ccxt_config": {},
            "ccxt_async_config": {},
        },
        "pairlists": [{"method": "StaticPairList"}],
        "entry_pricing": {"price_side": "same", "use_order_book": True, "order_book_top": 1},
        "exit_pricing": {"price_side": "same", "use_order_book": True, "order_book_top": 1},
    }
