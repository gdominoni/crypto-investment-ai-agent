"""Freqtrade execution engine for the candidate battery + Sonnet-approved
manual signals. Runs on the daily timeframe -- the same granularity the
candidate battery is backtested on, so no informative-pair merge or
cross-timeframe shift is needed: Freqtrade's own default behavior (a
signal set on a fully-formed candle fills at the NEXT candle's open) is
already the causally correct entry timing this project requires, with
nothing extra to get wrong.

Two independent entry sources, both gated:
  1. A candidate from `candidates/definitions.py` whose pooled status is
     'accepted' in `execution/live_battery_state.json` (refreshed
     weekly) fires on its own trigger condition for a given pair.
  2. A Sonnet-approved manual signal (`execution/signal_store.py`) for
     that exact pair+direction -- covers both a routine Sonnet trade
     proposal and a real-time shock-detector escalation
     (`llm_pipeline/shock_detector.py`), tagged `enter_tag=manual:<class>`
     so KPI reporting can separate the two. `populate_entry_trend` only
     checks for a pending manual signal on the LAST (current) row, since
     it's a live-only concept with nothing to check for past backtest
     dates; `confirm_trade_entry` consumes it exactly once, moving it
     into an ACTIVE store `custom_exit`/`custom_exit_price` read from
     (never re-consume) to recover this trade's own anchors later.

No third path exists. A 'watch' or 'rejected' candidate cannot place a
trade no matter what its own trigger condition does.

Exit is the anchor-based, duration-bucketed ladder from
`candidates/methodology.py` -- SL logic lives entirely in `custom_exit`,
never `custom_stoploss` (which can only tighten a stop over a trade's
life, incompatible with a ladder that widens with duration), and
`custom_exit_price` overrides Freqtrade's default open-price fill on a
CUSTOM_EXIT so the trade closes at the actual barrier level, not
wherever the triggering candle happened to open.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from freqtrade.strategy import IStrategy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from candidates.data_loading import load_funding
from candidates.definitions import compute_triggers
from candidates.methodology import barrier_prices, bucket_for_elapsed
from execution.signal_store import (
    consume_manual_signal, get_active_manual_signal, load_battery_state, peek_pending_manual_signal,
)

MANUAL_TAG_PREFIX = "manual:"


def _manual_tag(signal_class: str, direction: str) -> str:
    """Format: 'manual:<signal_class>:<direction>', e.g. 'manual:shock_reactive:long'.
    `direction` is always re-derived independently from `trade.is_short`/
    `side` wherever this strategy needs it (confirm_trade_entry, custom_exit,
    custom_exit_price) -- the tag's role is only to (a) mark an entry as
    manual (the `MANUAL_TAG_PREFIX` check) and (b) carry `signal_class`
    through into Freqtrade's own trade record for KPI reporting
    (`telegram/kpi_queries.py`), which DOES parse it back out."""
    return f"{MANUAL_TAG_PREFIX}{signal_class}:{direction}"


class SentimentAgentStrategy(IStrategy):
    timeframe = "1d"
    can_short = True
    trading_mode = "futures"
    margin_mode = "isolated"
    process_only_new_candles = True
    use_custom_stoploss = False
    minimal_roi = {"0": 10.0}  # effectively disabled -- custom_exit owns all exit decisions
    stoploss = -0.99  # emergency-only floor, custom_exit is expected to fire long before this

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        symbol = metadata["pair"].split("/")[0] + "USDT"
        # Freqtrade's `date` column is tz-aware UTC; every other daily
        # series in this project (data_loading.load_daily, the macro
        # calendar) is tz-naive -- comparing tz-aware vs tz-naive via
        # `.isin()` doesn't raise, it silently returns all-False, so the
        # tz is stripped here rather than assumed compatible.
        naive_date = dataframe["date"].dt.tz_localize(None)
        daily = dataframe.set_index(naive_date)[["open", "high", "low", "close", "volume"]]
        funding = load_funding(symbol)
        triggers = compute_triggers(daily, funding)
        for col in triggers.columns:
            dataframe[col] = triggers[col].to_numpy()
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        symbol = metadata["pair"].split("/")[0] + "USDT"
        battery = load_battery_state()
        dataframe["enter_long"], dataframe["enter_short"] = 0, 0
        dataframe["enter_tag"] = ""  # Freqtrade's own convention -- populates trade.enter_tag, read back in custom_exit

        for variant, spec in battery.get("candidates", {}).items():
            if variant not in dataframe.columns:
                continue
            col = "enter_long" if spec["direction"] == "long" else "enter_short"
            fires = dataframe[variant].astype(bool) & (dataframe[col] == 0)
            dataframe.loc[fires, col] = 1
            dataframe.loc[fires, "enter_tag"] = variant

        # Manual/Sonnet-approved signals are a live-only concept (there is
        # nothing to check for arbitrary past dates in a backtest), so
        # only the LAST row -- "now" -- is ever checked, not the whole
        # historical frame.
        last = dataframe.index[-1]
        for direction, col in (("long", "enter_long"), ("short", "enter_short")):
            pending = peek_pending_manual_signal(symbol, direction) if dataframe.loc[last, col] == 0 else None
            if pending is not None:
                dataframe.loc[last, col] = 1
                dataframe.loc[last, "enter_tag"] = _manual_tag(pending.get("signal_class", "manual"), direction)

        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                             time_in_force: str, current_time, entry_tag, side: str, **kwargs) -> bool:
        """A manual (Sonnet-approved) signal is consumed here -- exactly
        once, at the moment the order is actually about to be placed --
        and moved into the ACTIVE store so `custom_exit` can recover this
        same trade's anchors later without re-touching the pending queue."""
        if not entry_tag.startswith(MANUAL_TAG_PREFIX):
            return True  # a battery candidate already claimed this entry
        symbol = pair.split("/")[0] + "USDT"
        direction = "short" if side == "sell" else "long"
        manual = consume_manual_signal(symbol, direction)
        return manual is not None

    def custom_exit(self, pair: str, trade, current_time, current_rate: float, current_profit: float, **kwargs):
        symbol = pair.split("/")[0] + "USDT"
        tag = trade.enter_tag if hasattr(trade, "enter_tag") and trade.enter_tag else None
        direction = "short" if trade.is_short else "long"

        if tag and tag.startswith(MANUAL_TAG_PREFIX):
            spec = get_active_manual_signal(symbol, direction)  # read-only -- must not re-consume a newer, unrelated signal
        elif tag:
            spec = load_battery_state().get("candidates", {}).get(tag)
        else:
            spec = None
        if spec is None:
            return None  # no known anchor set for this trade -- leave it to the emergency stoploss floor

        anchors = {int(h): v for h, v in spec["anchors"].items()}
        horizons = tuple(sorted(anchors.keys()))
        elapsed_days = (current_time.normalize() - trade.open_date_utc.normalize()).days
        if elapsed_days < 1:
            return None  # entry candle itself, excluded from exit checks by construction

        horizon = bucket_for_elapsed(elapsed_days, horizons)
        if horizon is None:
            return "duration_ladder_timeout"

        tp_price, sl_price, _, _ = barrier_prices(trade.open_rate, direction, anchors, horizon,
                                                    spec["tp_mult"], spec["sl_mult"])
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        today = dataframe.loc[dataframe["date"] <= current_time]
        if len(today) == 0:
            return None
        hi, lo = today["high"].iloc[-1], today["low"].iloc[-1]

        sl_hit = (lo <= sl_price) if direction == "long" else (hi >= sl_price)
        tp_hit = (hi >= tp_price) if direction == "long" else (lo <= tp_price)
        if sl_hit:  # SL-priority tie-break, matching methodology.simulate_trade
            return "duration_ladder_sl"
        if tp_hit:
            return "duration_ladder_tp"
        return None

    def custom_exit_price(self, pair: str, trade, current_time, proposed_rate: float,
                           current_profit: float, exit_tag: str | None, **kwargs) -> float:
        if exit_tag not in ("duration_ladder_tp", "duration_ladder_sl"):
            return proposed_rate
        symbol = pair.split("/")[0] + "USDT"
        tag = trade.enter_tag if hasattr(trade, "enter_tag") and trade.enter_tag else None
        direction = "short" if trade.is_short else "long"
        if tag and tag.startswith(MANUAL_TAG_PREFIX):
            spec = get_active_manual_signal(symbol, direction)
        elif tag:
            spec = load_battery_state().get("candidates", {}).get(tag)
        else:
            spec = None
        if spec is None:
            return proposed_rate
        anchors = {int(h): v for h, v in spec["anchors"].items()}
        horizons = tuple(sorted(anchors.keys()))
        elapsed_days = (current_time.normalize() - trade.open_date_utc.normalize()).days
        horizon = bucket_for_elapsed(elapsed_days, horizons) or horizons[-1]
        tp_price, sl_price, _, _ = barrier_prices(trade.open_rate, direction, anchors, horizon,
                                                    spec["tp_mult"], spec["sl_mult"])
        return tp_price if exit_tag == "duration_ladder_tp" else sl_price
