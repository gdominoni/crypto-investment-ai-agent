"""Runs every candidate in `definitions.py` (the static battery) PLUS
every candidate in the dynamic registry (`llm_pipeline/dynamic_candidates.py`
-- conditions discovered live through "test it", re-tested here every
week with the exact same rigor rather than trusted permanently from
their first validation) through the fixed methodology (`methodology.py`)
across the live coin universe, pooled for anchor fitting, graded per-coin
and per-year for concentration, and writes the status table this
project's weekly refresh cycle re-runs unchanged.

Coin universe: BTC/ETH/BNB/XRP/DOGE/ADA/LTC, matching
`data_ingestion/market_data/binance_fetcher.py::COINS` exactly (the data
directory no longer carries any other symbol -- SOL/AVAX/LINK's data
files were removed rather than left stale and unfetched, so `data/` and
this project's actual coin universe can't silently drift apart again).
SOL/AVAX/LINK were tried once and excluded deliberately, not just never
gotten to: their intra-bar volatility relative to this pool is high
enough that a pooled anchor fit needs a volatility-scaling step this
project hasn't built yet -- adding them back without it risks diluting
the whole pool's fitted barriers, not just misjudging those three coins
on their own.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data_loading import load_daily
from .definitions import CANDIDATE_DIRECTIONS, build_triggers
from .methodology import (
    MethodologyConfig, apply_fdr_demotion, build_events, classify_status, compute_anchors,
    concentration_check, pattern_significance, report, shock_zscore_series, walk_forward,
)
from .status_history import (
    candidates_due_for_prune_decision, is_dropped, is_shut_down, record_horizon, record_status, trigger_shutdown,
)
from execution.live_test_state import load_horizons, save_horizons
from llm_pipeline.dynamic_candidates import record_test_result, registered_specs
from llm_pipeline.novel_condition_tester import test_novel_condition

COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LTCUSDT"]
HORIZONS_DAYS = (1, 3, 7, 14, 21)
SHOCK_ZSCORE_THRESHOLD = 3.0
ASSETS_DIR = Path(__file__).resolve().parent.parent / "docs" / "case_study" / "assets"
SIGNAL_STORE_PATH = Path(__file__).resolve().parent.parent / "execution" / "live_battery_state.json"


def _concentration_for(pattern: dict, oos: pd.DataFrame) -> tuple[dict, dict]:
    """Coin/year concentration measured on `pattern_significance`'s own
    per-event OOS forward returns -- the exact quantity the acceptance
    gate reads -- rather than on `walk_forward`'s TP/SL-conditioned
    `net_return`. Shared by the static and dynamic loops below, and
    mirrored in `replay/battery.py`. See concentration_check's docstring
    for why the two bases genuinely disagree."""
    frame = pattern.get("oos_events") if pattern.get("status") == "ok" else None
    if frame is not None and len(frame):
        return (concentration_check(frame, "group", value_col="forward_return"),
                concentration_check(frame, "period", value_col="forward_return"))
    return concentration_check(oos, "group"), concentration_check(oos, "period")


def run_all() -> tuple[pd.DataFrame, dict, dict]:
    """Returns (status_table, live_state, meta). `live_state` is what the
    execution engine actually reads at trade time -- per (candidate,
    coin) the most recent walk-forward fold's chosen (tp_mult, sl_mult)
    and the anchors fit from ALL available data, keyed only for
    candidates whose pooled status is 'accepted' -- this is the
    historical/backtest bar, not a claim of a live track record; see
    candidates/methodology.py::classify_status's docstring. A 'watch' or
    'rejected' candidate is never allowed to place a trade unattended,
    so it has no entry in `live_state` regardless of what any single
    coin's own numbers look like. `meta` carries `prune_candidates`
    (names tracked 2+ years without ever being accepted, due for a keep-or-drop
    decision), `shutdown_triggered` (True the run that finds zero active
    candidates left -- static and dynamic alike), and `failed_candidates`
    (names that raised an exception this run -- their own bug/bad data
    doesn't cost every OTHER candidate's already-computed result; each
    still shows a status='error' row so the failure is visible, not
    silently swallowed)."""
    if is_shut_down():
        return pd.DataFrame(), {"generated_at": datetime.now(timezone.utc).isoformat(), "candidates": {}}, {
            "prune_candidates": [], "shutdown_triggered": False, "already_shut_down": True, "failed_candidates": [],
        }

    cfg = MethodologyConfig(horizons=HORIZONS_DAYS)
    ohlc_by_coin = {c: load_daily(c) for c in COINS}
    triggers_by_coin = {c: build_triggers(c) for c in COINS}
    shock_z_by_coin = {c: shock_zscore_series(ohlc_by_coin[c]) for c in COINS}

    rows = []
    live_state = {"generated_at": datetime.now(timezone.utc).isoformat(), "horizons_days": list(HORIZONS_DAYS), "candidates": {}}
    failed_candidates = []
    for variant, direction in CANDIDATE_DIRECTIONS.items():
        if is_dropped(variant):
            continue
        try:
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
                record_status(variant, "insufficient_data")
                continue
            events = pd.concat(all_events, ignore_index=True)

            oos, params_log = walk_forward(events, ohlc_by_coin, direction, cfg)
            rep = report(oos)
            # pattern_significance is now the actual acceptance gate --
            # see classify_status's own docstring. `rep` (Sortino/win_rate)
            # is still computed and still reported below, but no longer
            # decides accepted/watch/rejected.
            pattern = pattern_significance(events, ohlc_by_coin, direction, cfg)
            # Concentration is measured on the SAME forward returns the
            # acceptance gate is decided on, not on walk_forward's
            # TP/SL-conditioned net_return -- the two genuinely disagree
            # (see concentration_check's own `value_col` note). Falls back to
            # the TP/SL basis only when the pattern test couldn't run at all,
            # where classify_status returns 'watch' regardless.
            coin_conc, year_conc = _concentration_for(pattern, oos)
            status = classify_status(rep, coin_conc, year_conc, pattern, cfg)
            record_status(variant, status)

            # Horizon is re-derived (and, if it changed, re-synced to the file
            # `_open_live_test` actually reads) every run this candidate has
            # enough data for pattern_significance to compute one -- independent
            # of accepted/watch/rejected. Without this, a live occurrence would
            # keep being held for whatever horizon was empirically best the
            # FIRST time this candidate was ever evaluated, never updated as
            # more history accumulates -- exactly mirrors replay/battery.py,
            # which already does this on every run.
            horizon_changed_to = None
            if pattern.get("status") == "ok":
                horizons = load_horizons()
                horizons[variant] = pattern["horizon"]
                save_horizons(horizons)
                if record_horizon(variant, pattern["horizon"]):
                    horizon_changed_to = pattern["horizon"]

            row = {
                "candidate": variant, "direction": direction, "status": status,
                "n": rep["n"], "win_rate": rep["win_rate"], "strict_win_rate": rep["strict_win_rate"],
                "sortino": rep["sortino"], "total_expectancy": rep["total_expectancy"], "timeout_fraction": rep["timeout_fraction"],
                "dominant_coin": coin_conc.get("dominant_group"), "max_coin_share": coin_conc.get("max_group_share"),
                "dominant_year": year_conc.get("dominant_group"), "max_year_share": year_conc.get("max_group_share"),
                "n_shock_excluded": n_shock_events,
                "pattern_significant": pattern.get("significant"), "pattern_p_value": pattern.get("p_value"),
                "pattern_excess_return": pattern.get("excess_return"), "pattern_mfe_mae_ratio": pattern.get("mfe_mae_ratio"),
            }
            if horizon_changed_to is not None:
                row["horizon_changed_to"] = horizon_changed_to
            rows.append(row)

            if status == "accepted":
                full_anchors = compute_anchors(events, HORIZONS_DAYS)
                last_params = params_log.iloc[-1] if len(params_log) else None
                live_state["candidates"][variant] = {
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
            # One candidate's own bug/bad data must not cost every OTHER
            # candidate's already-computed result this run -- status_history.py
            # deliberately isn't touched here (no real verdict to record),
            # but the failure still shows up in the table and live_state
            # simply keeps whatever this candidate's status was last run.
            print(f"Candidate '{variant}' failed to process, skipping (will retry next run): {e}")
            rows.append({"candidate": variant, "status": "error", "n": 0, "n_shock_excluded": 0})
            failed_candidates.append(variant)

    # Dynamic candidates -- conditions a human approved testing live via
    # "test it" -- are re-tested here with test_novel_condition() every
    # week, exactly like the static ones above: an accepted status is
    # never treated as permanent, and a rejected one is cheaply re-checked
    # in case conditions genuinely changed, not silently dropped.
    for spec in registered_specs():
        if is_dropped(spec.label):
            continue
        try:
            result = test_novel_condition(spec, COINS)
            status = result["status"]
            record_test_result(spec, status, source="weekly_revalidation")
            record_status(spec.label, status)
            if status == "insufficient_data":
                rows.append({"candidate": spec.label, "status": status, "n": 0, "n_shock_excluded": result.get("n_shock_excluded", 0)})
                continue
            coin_conc, year_conc = result["coin_concentration"], result["year_concentration"]
            # Same horizon re-sync as the static loop above -- a dynamic
            # candidate's horizon was previously only ever set once, at the
            # moment a human approved "Test It", and never refreshed by this
            # weekly run afterward.
            pattern = result.get("pattern_significance") or {}
            horizon_changed_to = None
            if pattern.get("status") == "ok":
                horizons = load_horizons()
                horizons[spec.label] = pattern["horizon"]
                save_horizons(horizons)
                if record_horizon(spec.label, pattern["horizon"]):
                    horizon_changed_to = pattern["horizon"]

            row = {
                "candidate": spec.label, "direction": spec.direction, "status": status,
                "n": result["n"], "win_rate": result["win_rate"], "strict_win_rate": result["strict_win_rate"],
                "sortino": result["sortino"], "total_expectancy": result["total_expectancy"], "timeout_fraction": result["timeout_fraction"],
                "dominant_coin": coin_conc.get("dominant_group"), "max_coin_share": coin_conc.get("max_group_share"),
                "dominant_year": year_conc.get("dominant_group"), "max_year_share": year_conc.get("max_group_share"),
                "n_shock_excluded": result.get("n_shock_excluded", 0),  # 0 by construction when the spec's own indicator IS shock_zscore -- see novel_condition_tester.py
            }
            if horizon_changed_to is not None:
                row["horizon_changed_to"] = horizon_changed_to
            rows.append(row)
            if status == "accepted" and result.get("live_anchors"):
                live_state["candidates"][spec.label] = {
                    "direction": spec.direction, "horizon": result["pattern_significance"]["horizon"],
                    "tp_mult": result["live_tp_mult"], "sl_mult": result["live_sl_mult"],
                    "anchors": result["live_anchors"],
                }
        except Exception as e:
            print(f"Dynamic candidate '{spec.label}' failed to process, skipping (will retry next run): {e}")
            rows.append({"candidate": spec.label, "status": "error", "n": 0, "n_shock_excluded": 0})
            failed_candidates.append(spec.label)

    # Family-level multiplicity control, AFTER every candidate in this run has a
    # p-value: with a registry this size, a raw p<0.05 threshold is expected to
    # manufacture several "significant" candidates with no real effect behind
    # them. Demotion-only -- BH can never promote (see methodology.py).
    rows = apply_fdr_demotion(rows, live_state)
    for r in rows:
        if r.get("fdr_demoted"):
            record_status(r["candidate"], r["status"])

    active_static = [v for v in CANDIDATE_DIRECTIONS if not is_dropped(v)]
    active_dynamic = [s.label for s in registered_specs() if not is_dropped(s.label)]
    shutdown_triggered = False
    if not active_static and not active_dynamic:
        trigger_shutdown("No candidates left -- every static and dynamic candidate has been dropped, and none have been proposed to replace them.")
        shutdown_triggered = True

    meta = {
        "prune_candidates": candidates_due_for_prune_decision(), "shutdown_triggered": shutdown_triggered,
        "failed_candidates": failed_candidates,
    }
    return pd.DataFrame(rows), live_state, meta


if __name__ == "__main__":
    result, live_state, meta = run_all()
    if meta.get("already_shut_down"):
        print("System already shut down -- no candidates left. See candidates/SHUTDOWN.")
        raise SystemExit(0)
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
    print(f"\n{len(live_state['candidates'])} candidate(s) accepted for live trading -> {SIGNAL_STORE_PATH}")
    if meta["prune_candidates"]:
        print(f"Due for a keep-or-drop decision (2+ years, never accepted): {meta['prune_candidates']}")
    if meta["shutdown_triggered"]:
        print("SHUTDOWN TRIGGERED: no candidates left to test.")
    if meta["failed_candidates"]:
        print(f"FAILED this run (see 'error' rows above, will retry next run): {meta['failed_candidates']}")
