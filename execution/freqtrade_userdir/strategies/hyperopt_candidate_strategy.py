"""Generic Freqtrade strategy for the periodic hyperopt cross-check
(execution/hyperopt_runner.py) -- ONE class, not one per candidate: which
candidate to test is read from the FT_HYPEROPT_CANDIDATE environment
variable at class-definition time (set by the runner before each
subprocess invocation), covering both a static candidate
(candidates/definitions.py) and a dynamic one
(llm_pipeline/dynamic_candidates.py) with the same code path. Entry is
this project's own real trigger condition -- never invented for this
strategy. Exit is this project's own duration-bucketed anchor ladder
(candidates/methodology.py::barrier_prices/bucket_for_elapsed), with
tp_mult/sl_mult exposed as Freqtrade's own DecimalParameter so its
(independent) optimizer -- not this project's own grid search -- picks
them. Purely a LOCAL, periodic, informational run: this strategy is
never loaded for live trading (see docs/case_study/methodology-decisions.md).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from freqtrade.strategy import DecimalParameter, IStrategy

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from candidates.data_loading import load_daily, load_funding
from candidates.definitions import CANDIDATE_DIRECTIONS, compute_triggers
from candidates.methodology import (
    barrier_prices, bucket_for_elapsed, build_events, compute_anchors, shock_zscore_series,
)
from candidates.run_battery import COINS, HORIZONS_DAYS, SHOCK_ZSCORE_THRESHOLD
from llm_pipeline.dynamic_candidates import registered_specs
from llm_pipeline.novel_condition_tester import SUPPORTED_INDICATORS, ConditionSpec, _OPERATORS, clause_from_dict

CANDIDATE = os.environ["FT_HYPEROPT_CANDIDATE"]
_dynamic_spec = next((s for s in registered_specs() if s.label == CANDIDATE), None)
if _dynamic_spec is None:
    # Not in production's own registry -- most of this project's real
    # dynamic-candidate discovery happened through the replay (its own,
    # isolated registry, see replay/state.py), so a candidate only
    # existing there must ALSO be checked, or the cross-check crashes on
    # (nearly) every real dynamic candidate this case study has actually
    # produced. The underlying market data is the same real history
    # either way -- only the spec's SOURCE differs, not its validity.
    from replay.state import load_dynamic_candidates as _load_replay_dynamic_candidates
    _replay_spec_dict = _load_replay_dynamic_candidates().get(CANDIDATE)
    if _replay_spec_dict is not None:
        _dynamic_spec = ConditionSpec(
            label=_replay_spec_dict["label"], direction=_replay_spec_dict["direction"],
            clauses=tuple(clause_from_dict(c) for c in _replay_spec_dict["clauses"]),
            horizons=tuple(_replay_spec_dict["horizons"]),
        )
DIRECTION = CANDIDATE_DIRECTIONS[CANDIDATE] if _dynamic_spec is None else _dynamic_spec.direction


def _trigger_for(daily: pd.DataFrame, funding) -> pd.Series:
    if _dynamic_spec is None:
        return compute_triggers(daily, funding)[CANDIDATE]
    trig = pd.Series(True, index=daily.index)
    for clause in _dynamic_spec.clauses:
        signal = SUPPORTED_INDICATORS[clause.indicator](daily, funding)
        trig &= _OPERATORS[clause.op](signal, clause.threshold).fillna(False)
    return trig


def _compute_anchors_for_candidate() -> dict:
    """Same real anchors (mean MFE/MAE per horizon) this project's own
    battery would compute -- pooled across the whole coin universe, real
    historical occurrences only, no invented data."""
    all_events = []
    for coin in COINS:
        daily = load_daily(coin)
        funding = load_funding(coin)
        trigger = _trigger_for(daily, funding)
        shock_z = shock_zscore_series(daily)
        ev = build_events(daily, trigger, DIRECTION, HORIZONS_DAYS, shock_z=shock_z, shock_threshold=SHOCK_ZSCORE_THRESHOLD)
        if len(ev):
            all_events.append(ev[ev["regime"] == "normal"])
    if not all_events or not any(len(e) for e in all_events):
        raise ValueError(f"No historical events found for candidate '{CANDIDATE}' -- nothing to hyperopt.")
    events = pd.concat(all_events, ignore_index=True)
    return compute_anchors(events, HORIZONS_DAYS)


ANCHORS = _compute_anchors_for_candidate()


class HyperoptCandidateStrategy(IStrategy):
    timeframe = "1d"
    can_short = True
    trading_mode = "futures"
    margin_mode = "isolated"
    process_only_new_candles = True
    use_custom_stoploss = False
    use_exit_signal = False
    minimal_roi = {"0": 10.0}  # effectively disabled -- custom_exit owns all exit decisions, same as the live strategy
    stoploss = -0.99  # emergency-only floor, custom_exit is expected to fire long before this
    startup_candle_count = 60

    tp_mult = DecimalParameter(0.5, 2.0, default=1.0, space="sell")
    sl_mult = DecimalParameter(0.5, 2.0, default=1.0, space="sell")

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        symbol = metadata["pair"].split("/")[0] + "USDT"
        daily = dataframe.copy()
        daily["date"] = pd.to_datetime(daily["date"]).dt.tz_localize(None)
        daily = daily.set_index("date")
        funding = load_funding(symbol)
        dataframe[CANDIDATE] = _trigger_for(daily, funding).reindex(daily.index).fillna(False).to_numpy()
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        col = "enter_long" if DIRECTION == "long" else "enter_short"
        dataframe.loc[dataframe[CANDIDATE], col] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return dataframe

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        elapsed = (current_time.replace(tzinfo=None) - trade.open_date_utc.replace(tzinfo=None)).days
        horizons = tuple(sorted(int(h) for h in ANCHORS.keys()))
        horizon = bucket_for_elapsed(elapsed, horizons)
        if horizon is None:
            return "timeout"
        direction = "short" if trade.is_short else "long"
        tp_price, sl_price, _, _ = barrier_prices(
            trade.open_rate, direction, ANCHORS, horizon, self.tp_mult.value, self.sl_mult.value)
        if direction == "long":
            if current_rate >= tp_price:
                return "tp_hit"
            if current_rate <= sl_price:
                return "sl_hit"
        else:
            if current_rate <= tp_price:
                return "tp_hit"
            if current_rate >= sl_price:
                return "sl_hit"
        return None
