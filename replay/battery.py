"""Runs the exact same candidate battery methodology as
candidates/run_battery.py, but sourced entirely from as-of data
(replay/time_sandbox.py) and written to the replay's own isolated state
(replay/state.py) -- never candidates/run_battery.py itself, which is
hardwired to production's full-history loaders and production's output
paths. The actual statistical logic (build_events, walk_forward, report,
concentration_check, classify_status) is imported and reused as-is, not
reimplemented -- only the data source and the output destination differ,
which is the whole point of a sandboxed replay. Also records each
candidate's status into replay/status_history.py (simulated-time-aware,
isolated from production's own) so a candidate that never validates for
2 simulated years can be offered the same keep/drop decision production
candidates get.
"""
from __future__ import annotations

import pandas as pd

from candidates.definitions import CANDIDATE_DIRECTIONS, compute_triggers
from candidates.methodology import (
    MethodologyConfig, build_events, classify_status, compute_anchors, concentration_check, pattern_significance,
    report, shock_zscore_series, walk_forward,
)
from candidates.run_battery import COINS, HORIZONS_DAYS, SHOCK_ZSCORE_THRESHOLD
from llm_pipeline.novel_condition_tester import Clause, ConditionSpec, test_novel_condition
from replay import state
from replay import status_history as sh
from replay.time_sandbox import daily_as_of, funding_as_of


def run_replay_battery(as_of: pd.Timestamp) -> dict:
    as_of_str = str(as_of.date())
    cfg = MethodologyConfig(horizons=HORIZONS_DAYS)
    ohlc_by_coin = {c: daily_as_of(c, as_of) for c in COINS}
    funding_by_coin = {c: funding_as_of(c, as_of) for c in COINS}
    triggers_by_coin = {c: compute_triggers(ohlc_by_coin[c], funding_by_coin[c]) for c in COINS}
    shock_z_by_coin = {c: shock_zscore_series(ohlc_by_coin[c]) for c in COINS}

    battery = {"as_of": as_of_str, "candidates": {}}
    status_summary = {}
    for variant, direction in CANDIDATE_DIRECTIONS.items():
        if sh.is_dropped(variant):
            continue
        try:
            all_events = []
            for coin in COINS:
                ev = build_events(ohlc_by_coin[coin], triggers_by_coin[coin][variant], direction, HORIZONS_DAYS,
                                   shock_z=shock_z_by_coin[coin], shock_threshold=SHOCK_ZSCORE_THRESHOLD)
                if len(ev):
                    ev["group"] = coin
                    ev["period"] = ev["trigger_time"].dt.year
                    all_events.append(ev[ev["regime"] == "normal"])
            if not all_events or not any(len(e) for e in all_events):
                status_summary[variant] = {"status": "insufficient_data"}
                sh.record_status(variant, "insufficient_data", as_of_str)
                continue
            events = pd.concat(all_events, ignore_index=True)
            oos, params_log = walk_forward(events, ohlc_by_coin, direction, cfg)
            rep = report(oos)
            coin_conc = concentration_check(oos, "group")
            year_conc = concentration_check(oos, "period")
            # pattern_significance is the actual acceptance gate now --
            # see classify_status's own docstring.
            pattern = pattern_significance(events, ohlc_by_coin, direction, cfg)
            status = classify_status(rep, coin_conc, year_conc, pattern, cfg)
            status_summary[variant] = {"status": status, "n": rep["n"], "win_rate": rep["win_rate"],
                                        "strict_win_rate": rep["strict_win_rate"], "sortino": rep["sortino"],
                                        "pattern_significant": pattern.get("significant"), "pattern_p_value": pattern.get("p_value"),
                                        "pattern_mfe_mae_ratio": pattern.get("mfe_mae_ratio"),
                                        "max_coin_share": coin_conc.get("max_group_share"), "dominant_coin": coin_conc.get("dominant_group"),
                                        "max_year_share": year_conc.get("max_group_share"), "dominant_year": year_conc.get("dominant_group")}
            sh.record_status(variant, status, as_of_str)
            # Horizon is tracked (and its transition from placeholder to
            # empirically-derived reported) independent of accepted/watch/
            # rejected -- pattern_significance can find a best horizon
            # whenever it has enough data, whatever the verdict itself is.
            if pattern.get("status") == "ok":
                horizons = state.load_horizons()
                horizons[variant] = pattern["horizon"]
                state.save_horizons(horizons)
                if sh.record_horizon(variant, pattern["horizon"]):
                    status_summary[variant]["horizon_changed_to"] = pattern["horizon"]
            if status == "accepted":
                full_anchors = compute_anchors(events, HORIZONS_DAYS)
                last_params = params_log.iloc[-1] if len(params_log) else None
                battery["candidates"][variant] = {
                    "direction": direction,
                    # `horizon`: what a live occurrence is actually held for now (see
                    # pattern_significance) -- no TP/SL barrier. tp_mult/sl_mult/anchors are
                    # kept for reporting/reference only, not used to open a live test anymore.
                    "horizon": pattern["horizon"],
                    "tp_mult": float(last_params["tp_mult"]) if last_params is not None else 1.0,
                    "sl_mult": float(last_params["sl_mult"]) if last_params is not None else 1.0,
                    "anchors": {str(h): {"mfe": full_anchors[h]["mfe"], "mae": full_anchors[h]["mae"]} for h in HORIZONS_DAYS},
                }
        except Exception as e:
            print(f"Replay battery: candidate '{variant}' failed as of {as_of.date()}, skipping: {e}")
            status_summary[variant] = {"status": "error"}

    for label, spec_dict in state.load_dynamic_candidates().items():
        if sh.is_dropped(label):
            continue
        try:
            spec = ConditionSpec(label=spec_dict["label"], clauses=tuple(Clause(**c) for c in spec_dict["clauses"]),
                                  direction=spec_dict["direction"], horizons=tuple(spec_dict["horizons"]))
            result = test_novel_condition(spec, COINS, as_of=as_of)
            coin_conc, year_conc = result.get("coin_concentration") or {}, result.get("year_concentration") or {}
            pattern = result.get("pattern_significance") or {}
            status_summary[label] = {"status": result["status"], "n": result.get("n"), "win_rate": result.get("win_rate"),
                                      "strict_win_rate": result.get("strict_win_rate"), "sortino": result.get("sortino"),
                                      "pattern_significant": pattern.get("significant"), "pattern_p_value": pattern.get("p_value"),
                                      "pattern_mfe_mae_ratio": pattern.get("mfe_mae_ratio"),
                                      "max_coin_share": coin_conc.get("max_group_share"), "dominant_coin": coin_conc.get("dominant_group"),
                                      "max_year_share": year_conc.get("max_group_share"), "dominant_year": year_conc.get("dominant_group")}
            sh.record_status(label, result["status"], as_of_str)
            if pattern.get("status") == "ok":
                horizons = state.load_horizons()
                horizons[label] = pattern["horizon"]
                state.save_horizons(horizons)
                if sh.record_horizon(label, pattern["horizon"]):
                    status_summary[label]["horizon_changed_to"] = pattern["horizon"]
            if result["status"] == "accepted" and result.get("live_anchors"):
                battery["candidates"][label] = {
                    "direction": spec.direction, "horizon": result["pattern_significance"]["horizon"],
                    "tp_mult": result["live_tp_mult"], "sl_mult": result["live_sl_mult"],
                    "anchors": result["live_anchors"],
                }
        except Exception as e:
            print(f"Replay battery: dynamic candidate '{label}' failed as of {as_of.date()}, skipping: {e}")
            status_summary[label] = {"status": "error"}

    state.save_battery_status(battery)
    return status_summary
