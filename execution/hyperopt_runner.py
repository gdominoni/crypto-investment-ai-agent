"""Periodic, LOCAL-only Freqtrade hyperopt cross-check. For each tracked
candidate (static or dynamic, any status), runs Freqtrade's own
(independent) Bayesian optimizer over TP/SL multipliers and records the
result as a SEPARATE, purely informational score -- never gating
acceptance, never feeding live execution (this project opens no funded
position at all, see docs/case_study/methodology-decisions.md).

This project's own walk-forward grid search (candidates/methodology.py
::walk_forward) already computes an equivalent "if traded with a barrier
structure" figure -- shown as the "For reference, trading this with a
TP/SL structure..." line in every message. This is a genuinely
independent SECOND opinion: different search machinery (Bayesian
optimization over a continuous space vs. this project's own 25-point
grid), a different, third-party, industry-standard backtesting engine,
run only ever offline and periodically (never in the live/replay hot
path). Deliberately kept OFF the live host entirely (see PROJECT_MAP.md's
"Cost Optimization" Part 3): it's the one genuinely heavy compute cost in
this project (a single candidate at 50 epochs takes several real minutes;
the entire weekly candidate battery, by contrast, is ~4.4s for 38
candidates), it never gates anything so it never needs to be fresh, and
its only output is one small JSON file. Run it locally, by hand or via
your own local cron -- `python3 -m execution.hyperopt_runner` from the
project root (see the __main__ block below) -- then copy or push
`execution/hyperopt_results.json` to wherever the live bot reads from.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from candidates.definitions import CANDIDATE_DIRECTIONS
from candidates.run_battery import COINS
from execution.freqtrade_bridge import FT_DATADIR, FT_USERDIR, build_config, sync_data
from llm_pipeline.dynamic_candidates import registered_specs
from candidates.atomic_json import write_json

STRATEGY_PATH = Path(__file__).resolve().parent / "freqtrade_userdir" / "strategies"
CONFIG_PATH = FT_USERDIR / "hyperopt_config.json"
RESULTS_PATH = Path(__file__).resolve().parent / "hyperopt_results.json"
HYPEROPT_RESULTS_DIR = FT_USERDIR / "hyperopt_results"


def load_results() -> dict:
    """Never raises. This is read from inside the notification loop, where an
    unreadable cross-check file must not take down the message that carries the
    actual result -- a purely informational second opinion is not worth losing a
    live-test notification over."""
    if not RESULTS_PATH.exists():
        return {}
    try:
        data = json.loads(RESULTS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_results(results: dict) -> None:
    write_json(RESULTS_PATH, results)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_one(candidate: str, timerange: str, epochs: int) -> dict:
    """Always returns a dict, never None -- either {"status": "ok", ...}
    or {"status": "failed", "reason": ..., "at": ...}, so a failed or
    data-starved cross-check is PERSISTED and distinguishable from
    "never attempted", instead of being silently indistinguishable (see
    docs/case_study/methodology-decisions.md). A failed cross-check for
    one candidate must never block the others -- this function itself
    never raises; run_all()'s per-candidate isolation is a second layer
    of the same discipline every other batch loop in this project
    follows, in case something outside this function's own try/except
    still goes wrong (e.g. writing the shared config)."""
    import os

    env = {**os.environ, "FT_HYPEROPT_CANDIDATE": candidate}
    cmd = [
        sys.executable, "-m", "freqtrade", "hyperopt",
        "--config", str(CONFIG_PATH),
        "--strategy", "HyperoptCandidateStrategy",
        "--strategy-path", str(STRATEGY_PATH),
        "--datadir", str(FT_DATADIR),
        "--userdir", str(FT_USERDIR),
        "--hyperopt-loss", "SortinoHyperOptLossDaily",
        "--spaces", "sell",
        "--epochs", str(epochs),
        "--timerange", timerange,
        "-j", "1",
    ]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        reason = f"timed out after 1800s (epochs={epochs}, timerange={timerange})"
        print(f"Freqtrade hyperopt for '{candidate}' {reason}.")
        return {"status": "failed", "reason": reason, "at": _now()}
    except Exception as e:
        reason = f"subprocess launch failed: {type(e).__name__}: {e}"
        print(f"Freqtrade hyperopt for '{candidate}' {reason}.")
        return {"status": "failed", "reason": reason, "at": _now()}

    if proc.returncode != 0:
        tail = proc.stderr.strip()[-500:] or "no stderr output"
        reason = f"process exited with code {proc.returncode}: {tail}"
        print(f"Freqtrade hyperopt failed for '{candidate}': {proc.stderr[-2000:]}")
        return {"status": "failed", "reason": reason, "at": _now()}

    last_result_path = HYPEROPT_RESULTS_DIR / ".last_result.json"
    if not last_result_path.exists():
        reason = "process exited cleanly but produced no results file"
        print(f"Freqtrade hyperopt for '{candidate}' produced no results file.")
        return {"status": "failed", "reason": reason, "at": _now()}

    try:
        latest_file = HYPEROPT_RESULTS_DIR / json.loads(last_result_path.read_text())["latest_hyperopt"]

        from freqtrade.optimize.hyperopt_tools import HyperoptTools
        filteroptions = {
            "only_best": False, "only_profitable": False, "filter_min_trades": 0, "filter_max_trades": 0,
            "filter_min_avg_time": None, "filter_max_avg_time": None, "filter_min_avg_profit": None,
            "filter_max_avg_profit": None, "filter_min_total_profit": None, "filter_max_total_profit": None,
            "filter_min_objective": None, "filter_max_objective": None,
        }
        epochs_data, _ = HyperoptTools.load_filtered_results(latest_file, filteroptions)
    except Exception as e:
        reason = f"failed to parse results file: {type(e).__name__}: {e}"
        print(f"Freqtrade hyperopt for '{candidate}' {reason}.")
        return {"status": "failed", "reason": reason, "at": _now()}

    if not epochs_data:
        reason = "results file contained no evaluated epochs (likely insufficient historical data -- no trades generated for this candidate over the timerange)"
        print(f"Freqtrade hyperopt for '{candidate}': {reason}.")
        return {"status": "failed", "reason": reason, "at": _now()}

    best = min(epochs_data, key=lambda e: e.get("loss", float("inf")))
    m = best["results_metrics"]
    return {
        "status": "ok", "at": _now(),
        "tp_mult": best["params_dict"].get("tp_mult"), "sl_mult": best["params_dict"].get("sl_mult"),
        "n": m.get("total_trades"), "winrate": m.get("winrate"), "sortino": m.get("sortino"),
        "profit_total": m.get("profit_total"), "expectancy": m.get("expectancy"),
        "epochs_run": len(epochs_data), "timerange": timerange,
    }


def run_all(candidates: list[str] | None = None, timerange: str = "20180101-", epochs: int = 50) -> dict:
    """Runs the cross-check for every tracked candidate not given
    explicitly (static + dynamic, dropped ones excluded). Syncs data
    once up front, writes one shared config, then isolates each
    candidate's own subprocess run so one failure can't cost the rest --
    _run_one() never raises, so even an unexpected exception there is
    persisted as a "failed" result rather than aborting the remaining
    candidates in this loop."""
    if candidates is None:
        from candidates.status_history import is_dropped
        static = [c for c in CANDIDATE_DIRECTIONS if not is_dropped(c)]
        dynamic = [s.label for s in registered_specs() if not is_dropped(s.label)]
        candidates = static + dynamic

    sync_data(COINS)
    FT_USERDIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(build_config(COINS)))

    results = load_results()
    for candidate in candidates:
        try:
            results[candidate] = _run_one(candidate, timerange, epochs)
        except Exception as e:
            results[candidate] = {"status": "failed", "reason": f"unexpected error: {type(e).__name__}: {e}", "at": _now()}
    _save_results(results)
    return results


def format_result(candidate: str, short: bool = False) -> str:
    """Never raises, for any shape of stored record.

    Every field is fetched with `.get` and type-checked before formatting. The
    previous version indexed `r['tp_mult']` directly on the success path, so a
    record written by an older version of this module -- or a run interrupted
    mid-write -- would have raised KeyError inside the notification loop and
    killed the message it was appended to.

    `short=True` gives the one-line form used per resolved live test; the full
    form is for periodic checkpoints, where the extra detail is worth the space."""
    r = load_results().get(candidate)
    if not isinstance(r, dict):
        return "TP/SL: pending hyperopt cross-check."
    if r.get("status") == "failed":
        return (f"TP/SL: last hyperopt attempt failed ({r.get('at', 'unknown time')}) -- "
                f"{r.get('reason', 'no reason recorded')}.")
    tp, sl = r.get("tp_mult"), r.get("sl_mult")
    if not isinstance(tp, (int, float)) or not isinstance(sl, (int, float)):
        return "TP/SL: pending hyperopt cross-check (stored record incomplete)."
    if short:
        return f"TP/SL from hyperopt: {tp:.2f} / {sl:.2f} (independent cross-check, informational)"

    def _num(key, fmt, default="n/a"):
        v = r.get(key)
        return format(v, fmt) if isinstance(v, (int, float)) else default

    return (f"Freqtrade hyperopt cross-check (independent optimizer, local/periodic, purely informational): "
            f"TP mult={tp:.2f}, SL mult={sl:.2f} -> N={_num('n', 'd')}, win_rate={_num('winrate', '.1%')}, "
            f"Sortino={_num('sortino', '.2f')}, total_profit={_num('profit_total', '+.1%')} "
            f"(searched {_num('epochs_run', 'd')} epochs over {r.get('timerange', 'an unrecorded range')}).")


if __name__ == "__main__":
    # Run from the project root: `python3 -m execution.hyperopt_runner` (plain
    # module execution, not `python3 execution/hyperopt_runner.py` -- the
    # latter never puts the project root on sys.path, so the candidates./
    # llm_pipeline. imports above fail immediately). Meant to be run locally,
    # by hand or your own local cron -- never on the live host, see this
    # module's own docstring and PROJECT_MAP.md's "Cost Optimization" Part 3.
    import argparse

    parser = argparse.ArgumentParser(
        description="Freqtrade hyperopt cross-check -- local-only, periodic, purely informational. "
                     "Writes execution/hyperopt_results.json; copy or push that one file to wherever "
                     "the live bot reads from when you're done.",
    )
    parser.add_argument("candidates", nargs="*", metavar="CANDIDATE",
                         help="Specific candidate label(s) to run (default: every tracked, non-dropped candidate -- "
                              "static and dynamic). Repeatable, e.g. 'c1_long my_dynamic_condition'.")
    parser.add_argument("--epochs", type=int, default=50, help="Bayesian optimizer epochs per candidate (default: 50).")
    parser.add_argument("--timerange", default="20180101-",
                         help="Freqtrade --timerange string (default: '20180101-', meaning from that date to now).")
    args = parser.parse_args()

    print(f"Running hyperopt cross-check ({args.epochs} epochs, timerange {args.timerange}) "
          f"for: {', '.join(args.candidates) if args.candidates else 'every tracked candidate'}")
    results = run_all(candidates=args.candidates or None, timerange=args.timerange, epochs=args.epochs)
    print()
    for candidate in results:
        print(f"{candidate}: {format_result(candidate)}")
