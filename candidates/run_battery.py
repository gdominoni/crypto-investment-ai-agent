"""Runs every candidate in `definitions.py` through the fixed methodology
(`methodology.py`) across the live coin universe, pooled for anchor
fitting, graded per-coin and per-year for concentration, and writes the
status table this project's weekly refresh cycle re-runs unchanged.

Coin universe: BTC/ETH/BNB/XRP/DOGE/ADA/LTC. SOL/AVAX/LINK are available
in `data/` but excluded here -- their intra-bar volatility relative to
this pool is high enough that a pooled anchor fit needs a volatility-
scaling step this project hasn't built yet; adding them without it risks
diluting the whole pool's fitted barriers, not just misjudging those
three coins on their own.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data_loading import load_daily
from .definitions import CANDIDATE_DIRECTIONS, build_triggers
from .methodology import (
    MethodologyConfig, build_events, classify_status, compute_anchors, concentration_check,
    report, shock_zscore_series, walk_forward,
)

COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LTCUSDT"]
HORIZONS_DAYS = (1, 3, 7, 14, 21)
SHOCK_ZSCORE_THRESHOLD = 3.0
ASSETS_DIR = Path(__file__).resolve().parent.parent / "docs" / "case_study" / "assets"
SIGNAL_STORE_PATH = Path(__file__).resolve().parent.parent / "execution" / "live_battery_state.json"


def run_all() -> tuple[pd.DataFrame, dict]:
    """Returns (status_table, live_state). `live_state` is what the
    execution engine actually reads at trade time -- per (candidate,
    coin) the most recent walk-forward fold's chosen (tp_mult, sl_mult)
    and the anchors fit from ALL available data, keyed only for
    candidates whose pooled status is 'validated'. A 'watch' or
    'rejected' candidate is never allowed to place a trade unattended,
    so it has no entry in `live_state` regardless of what any single
    coin's own numbers look like."""
    cfg = MethodologyConfig(horizons=HORIZONS_DAYS)
    ohlc_by_coin = {c: load_daily(c) for c in COINS}
    triggers_by_coin = {c: build_triggers(c) for c in COINS}
    shock_z_by_coin = {c: shock_zscore_series(ohlc_by_coin[c]) for c in COINS}

    rows = []
    live_state = {"generated_at": datetime.now(timezone.utc).isoformat(), "horizons_days": list(HORIZONS_DAYS), "candidates": {}}
    for variant, direction in CANDIDATE_DIRECTIONS.items():
        all_events, all_shock_events = [], []
        for coin in COINS:
            ev = build_events(ohlc_by_coin[coin], triggers_by_coin[coin][variant], direction, HORIZONS_DAYS,
                               shock_z=shock_z_by_coin[coin], shock_threshold=SHOCK_ZSCORE_THRESHOLD)
            if len(ev):
                ev["group"] = coin
                ev["period"] = ev["trigger_time"].dt.year
                all_shock_events.append(ev[ev["regime"] == "shock"])
                all_events.append(ev[ev["regime"] == "normal"])  # the static battery only ever fits/trades 'normal'-regime events
        n_shock_events = sum(len(e) for e in all_shock_events)
        if not all_events or not any(len(e) for e in all_events):
            rows.append({"candidate": variant, "status": "insufficient_data", "n": 0, "n_shock_excluded": n_shock_events})
            continue
        events = pd.concat(all_events, ignore_index=True)

        oos, params_log = walk_forward(events, ohlc_by_coin, direction, cfg)
        rep = report(oos)
        coin_conc = concentration_check(oos, "group")
        year_conc = concentration_check(oos, "period")
        status = classify_status(rep, coin_conc, year_conc, cfg)

        rows.append({
            "candidate": variant, "direction": direction, "status": status,
            "n": rep["n"], "win_rate": rep["win_rate"], "strict_win_rate": rep["strict_win_rate"],
            "sortino": rep["sortino"], "total_expectancy": rep["total_expectancy"], "timeout_fraction": rep["timeout_fraction"],
            "dominant_coin": coin_conc.get("dominant_group"), "max_coin_share": coin_conc.get("max_group_share"),
            "dominant_year": year_conc.get("dominant_group"), "max_year_share": year_conc.get("max_group_share"),
            "n_shock_excluded": n_shock_events,
        })

        if status == "validated":
            full_anchors = compute_anchors(events, HORIZONS_DAYS)
            last_params = params_log.iloc[-1] if len(params_log) else None
            live_state["candidates"][variant] = {
                "direction": direction,
                "tp_mult": float(last_params["tp_mult"]) if last_params is not None else 1.0,
                "sl_mult": float(last_params["sl_mult"]) if last_params is not None else 1.0,
                "anchors": {str(h): {"mfe": full_anchors[h]["mfe"], "mae": full_anchors[h]["mae"]} for h in HORIZONS_DAYS},
            }
    return pd.DataFrame(rows), live_state


if __name__ == "__main__":
    result, live_state = run_all()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(ASSETS_DIR / "candidate_battery_status.csv", index=False)
    SIGNAL_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SIGNAL_STORE_PATH.write_text(json.dumps(live_state, indent=2))

    pd.set_option("display.width", 200)
    fmt = {
        "win_rate": "{:.1%}".format, "strict_win_rate": "{:.1%}".format, "sortino": "{:.2f}".format,
        "total_expectancy": "{:+.1%}".format, "timeout_fraction": "{:.1%}".format, "max_coin_share": "{:.1%}".format,
        "max_year_share": "{:.1%}".format,
    }
    print(result.to_string(index=False, formatters={k: v for k, v in fmt.items() if k in result.columns}))
    print(f"\n{len(live_state['candidates'])} candidate(s) validated for live trading -> {SIGNAL_STORE_PATH}")
