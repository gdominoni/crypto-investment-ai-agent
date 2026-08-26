"""Tests a novel condition Haiku/Sonnet flagged and a human approved --
via a small, whitelisted spec (indicator name + comparison + threshold),
never by executing LLM-generated code. Sonnet can only ever choose from
`SUPPORTED_INDICATORS` below; if the condition it wants to test isn't
expressible with what's here, the correct outcome is a human adding a
new indicator to this registry deliberately, not the system running
arbitrary generated logic unattended.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from candidates.data_loading import load_daily, load_funding, zscore
from candidates.definitions import trend_efficiency_ratio
from candidates.macro_calendar import macro_release_days
from candidates.methodology import (
    MethodologyConfig, build_events, classify_status, concentration_check, report, shock_zscore_series, walk_forward,
)

SUPPORTED_INDICATORS: dict[str, Callable[[pd.DataFrame, pd.Series | None], pd.Series]] = {
    "close_return_1d": lambda df, funding: df["close"].pct_change(1),
    "close_return_5d": lambda df, funding: df["close"].pct_change(5),
    "daily_range_pct": lambda df, funding: (df["high"] - df["low"]) / df["close"],
    "volume_zscore_30d": lambda df, funding: zscore(df["volume"], 30),
    "funding_zscore_30d": lambda df, funding: zscore(funding.reindex(df.index).ffill(), 30) if funding is not None else pd.Series(np.nan, index=df.index),
    "efficiency_ratio_20d": lambda df, funding: trend_efficiency_ratio(df["close"], 20),
    "is_macro_day": lambda df, funding: pd.Series(df.index.isin(macro_release_days()), index=df.index).astype(float),
    # the same 'vol-of-vol' shock measure Phase 1's methodology uses to
    # isolate extreme historical events from the static battery -- tested
    # here on ITS OWN excluded population when a live shock fires, giving
    # it a real, walk-forward-validated anchor set rather than an
    # invented one.
    "shock_zscore": lambda df, funding: shock_zscore_series(df),
}

_OPERATORS: dict[str, Callable[[pd.Series, float], pd.Series]] = {
    "<": operator.lt, ">": operator.gt, "<=": operator.le, ">=": operator.ge,
}


@dataclass(frozen=True)
class ConditionSpec:
    label: str
    indicator: str
    op: str
    threshold: float
    direction: str  # "long" or "short"
    horizons: tuple[int, ...] = (1, 3, 7, 14, 21)

    def __post_init__(self):
        if self.indicator not in SUPPORTED_INDICATORS:
            raise ValueError(f"Unsupported indicator '{self.indicator}' -- must be one of {list(SUPPORTED_INDICATORS)}")
        if self.op not in _OPERATORS:
            raise ValueError(f"Unsupported operator '{self.op}' -- must be one of {list(_OPERATORS)}")
        if self.direction not in ("long", "short"):
            raise ValueError("direction must be 'long' or 'short'")


def test_novel_condition(spec: ConditionSpec, coins: list[str]) -> dict:
    cfg = MethodologyConfig(horizons=spec.horizons)
    indicator_fn = SUPPORTED_INDICATORS[spec.indicator]
    op_fn = _OPERATORS[spec.op]

    all_events = []
    ohlc_by_coin = {}
    for coin in coins:
        daily = load_daily(coin)
        ohlc_by_coin[coin] = daily
        funding = load_funding(coin)
        signal = indicator_fn(daily, funding)
        trigger = op_fn(signal, spec.threshold).fillna(False)
        ev = build_events(daily, trigger, spec.direction, spec.horizons)
        if len(ev):
            ev["group"] = coin
            ev["period"] = ev["trigger_time"].dt.year
            all_events.append(ev)

    if not all_events:
        return {"spec": spec, "status": "insufficient_data", "n_raw_triggers": 0}

    events = pd.concat(all_events, ignore_index=True)
    oos, params_log = walk_forward(events, ohlc_by_coin, spec.direction, cfg)
    rep = report(oos)
    coin_conc = concentration_check(oos, "group")
    year_conc = concentration_check(oos, "period")
    status = classify_status(rep, coin_conc, year_conc, cfg)

    # Full-data anchors + the most recent fold's multipliers -- what a
    # caller actually needs to push this condition as a live signal
    # (`execution/signal_store.py::push_manual_signal`) once validated,
    # not just a pass/fail verdict.
    live_anchors, live_tp_mult, live_sl_mult = None, None, None
    if status == "validated":
        from candidates.methodology import compute_anchors
        full_anchors = compute_anchors(events, spec.horizons)
        live_anchors = {str(h): {"mfe": full_anchors[h]["mfe"], "mae": full_anchors[h]["mae"]} for h in spec.horizons}
        if len(params_log):
            live_tp_mult = float(params_log.iloc[-1]["tp_mult"])
            live_sl_mult = float(params_log.iloc[-1]["sl_mult"])

    return {
        "spec": spec, "status": status, "n_raw_triggers": len(events), **rep,
        "coin_concentration": coin_conc, "year_concentration": year_conc,
        "params_log": params_log.to_dict("records"),
        "live_anchors": live_anchors, "live_tp_mult": live_tp_mult, "live_sl_mult": live_sl_mult,
    }
