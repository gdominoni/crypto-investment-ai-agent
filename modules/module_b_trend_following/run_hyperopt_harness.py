"""Orchestrates Module B's Phase 9 hyperopt harness: 8 base 3-indicator
combos x 2 exit modes = 16 dedicated Freqtrade hyperopt runs on 1h
In-Sample data, each searching its own coarse-stepped parameter space
(step=5 for integer lengths/thresholds, step=0.5 for stddevs/multipliers,
per instruction -- this keeps hyperopt's search efficient and prevents
it from reporting false precision that isn't really distinguishable from
noise) via ProjectHierarchyLoss (Win Rate -> Sortino -> Net Profit, the
same hierarchy candidate_ranking.py and the capital allocator use
everywhere else in this project -- reused as-is from the earlier
TrendEmaAdx hyperopt work, no changes needed).

For each run: hyperopt searches the combo's own hyperopt spaces on IS
data (see COMBOS below -- Combos 1-4's classic-RSI exit uses a "sell"-
space parameter, rsi_overbought; Combos 5-8's RSI-BB exit is fully
data-driven and has no sell-space parameter at all, so requesting
--spaces buy sell unconditionally for every combo fails on 5-8 with "no
parameter for this space was found" -- caught on the real first full
run, after Combos 1-4 had already completed successfully). The
discovered best parameters are then re-backtested cleanly on IS (for a
parseable BacktestResult) and validated against the untouched OOS
period. Results are appended to a CSV as each run completes (not held in
memory until the end) so a future interruption doesn't lose completed
work the way the first full run did.

Run locally with:
    python -m modules.module_b_trend_following.run_hyperopt_harness
"""

import csv
import json
import subprocess
from datetime import date
from pathlib import Path

from modules.module_b_trend_following.candidate_ranking import BacktestResult, load_backtest_result
from modules.module_b_trend_following.oos_split import split_is_oos

MODULE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = MODULE_DIR / "user_data" / "backtest_results"
STRATEGIES_DIR = MODULE_DIR / "user_data" / "strategies"
OUTPUT_CSV = MODULE_DIR / "hyperopt_harness_results.csv"

DATA_START = date(2023, 8, 20)
DATA_END = date(2026, 8, 17)

EPOCHS = 200
HYPEROPT_LOSS = "ProjectHierarchyLoss"
# Spec said -j -1 (all cores); dropped to 4 after the first real launch
# showed genuine memory pressure (swap at 92% usage, 3.77GB/4GB) with 8
# parallel workers each holding their own copy of the loaded dataset --
# the exact risk flagged when this harness was reviewed, confirmed with
# real data rather than left as a hypothetical. See decisions-log.md.
JOBS = 4

# 8 base combos: (strategy class name, strategy filename, hyperopt spaces).
# Combos 1-4 (classic RSI) have a real "sell"-space parameter
# (rsi_overbought); Combos 5-8 (RSI-BB) exit on a data-driven band cross
# with no separate sell-space scalar to search, so "sell" must be left
# out of --spaces for those or Freqtrade errors out.
COMBOS = [
    ("Combo1ClassicRsiPriceBBVolume", "combo1_classicrsi_pricebb_volume", ["buy", "sell"]),
    ("Combo2ClassicRsiPriceBBCmf", "combo2_classicrsi_pricebb_cmf", ["buy", "sell"]),
    ("Combo3ClassicRsiKeltnerVolume", "combo3_classicrsi_keltner_volume", ["buy", "sell"]),
    ("Combo4ClassicRsiKeltnerCmf", "combo4_classicrsi_keltner_cmf", ["buy", "sell"]),
    ("Combo5RsiBBPriceBBVolume", "combo5_rsibb_pricebb_volume", ["buy"]),
    ("Combo6RsiBBPriceBBCmf", "combo6_rsibb_pricebb_cmf", ["buy"]),
    ("Combo7RsiBBKeltnerVolume", "combo7_rsibb_keltner_volume", ["buy"]),
    ("Combo8RsiBBKeltnerCmf", "combo8_rsibb_keltner_cmf", ["buy"]),
]

EXIT_MODES = {
    "null": {"stoploss": -0.99, "roi": {"0": 100.0}},
    "wide": {"stoploss": -0.15, "roi": {"0": 0.15}},
}

CSV_FIELDS = [
    "strategy", "exit_mode", "period",
    "total_trades", "win_rate", "sortino_ratio", "net_profit_pct", "backtest_days",
]


def _set_exit_mode(strategy_class: str, strategy_filename: str, exit_mode: str) -> None:
    """Pre-writes stoploss/roi into the strategy's auto-loaded parameter
    file. Hyperopt only searches the spaces passed via --spaces -- stoploss/
    roi stay fixed at whatever this file says for the whole run and are
    carried through, unmodified, into the exported "best params" file at
    the end. Confirmed against this project's earlier observed behavior
    (trend_ema_adx.json): a space not included in --spaces is preserved
    from the loaded file, not reset to a class default.
    """
    mode = EXIT_MODES[exit_mode]
    params = {
        "strategy_name": strategy_class,
        "params": {
            "roi": mode["roi"],
            "stoploss": {"stoploss": mode["stoploss"]},
            "trailing": {
                "trailing_stop": False,
                "trailing_stop_positive": 0.01,
                "trailing_stop_positive_offset": 0.02,
                "trailing_only_offset_is_reached": True,
            },
        },
    }
    (STRATEGIES_DIR / f"{strategy_filename}.json").write_text(json.dumps(params, indent=2))


def _run_hyperopt(strategy_class: str, timerange: str, spaces: list[str]) -> None:
    subprocess.run(
        [
            "docker", "compose", "--env-file", "../../.env", "run", "--rm", "freqtrade",
            "hyperopt",
            "--config", "user_data/config.json",
            "--strategy", strategy_class,
            "--hyperopt-loss", HYPEROPT_LOSS,
            "--spaces", *spaces,
            "-i", "1h",
            "--timerange", timerange,
            "-e", str(EPOCHS),
            "-j", str(JOBS),
        ],
        cwd=MODULE_DIR,
        check=True,
    )


def _run_backtest(strategy_class: str, timerange: str) -> Path:
    subprocess.run(
        [
            "docker", "compose", "--env-file", "../../.env", "run", "--rm", "freqtrade",
            "backtesting",
            "--config", "user_data/config.json",
            "--strategy", strategy_class,
            "-i", "1h",
            "--timerange", timerange,
        ],
        cwd=MODULE_DIR,
        check=True,
    )
    return sorted(RESULTS_DIR.glob("backtest-result-*.zip"))[-1]


def _write_row(writer: csv.DictWriter, strategy_class: str, exit_mode: str, period: str, result: BacktestResult) -> None:
    writer.writerow({
        "strategy": strategy_class,
        "exit_mode": exit_mode,
        "period": period,
        "total_trades": result.total_trades,
        "win_rate": round(result.win_rate, 4),
        "sortino_ratio": round(result.sortino_ratio, 3),
        "net_profit_pct": round(result.net_profit_abs / 100, 3),  # dry_run_wallet=10000
        "backtest_days": result.backtest_days,
    })


def main() -> None:
    split = split_is_oos(DATA_START, DATA_END, oos_months=12)

    is_new_file = not OUTPUT_CSV.exists()
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new_file:
            writer.writeheader()

        for strategy_class, strategy_filename, spaces in COMBOS:
            for exit_mode in EXIT_MODES:
                print(f"\n=== {strategy_class} / exit={exit_mode} / spaces={spaces} ===")
                _set_exit_mode(strategy_class, strategy_filename, exit_mode)

                print("  Running hyperopt on IS data...")
                _run_hyperopt(strategy_class, split.is_timerange, spaces)

                print("  Re-backtesting IS with discovered params...")
                is_zip = _run_backtest(strategy_class, split.is_timerange)
                is_result = load_backtest_result(is_zip, strategy_class)
                _write_row(writer, strategy_class, exit_mode, "IS", is_result)
                f.flush()

                print("  Validating against OOS...")
                oos_zip = _run_backtest(strategy_class, split.oos_timerange)
                oos_result = load_backtest_result(oos_zip, strategy_class)
                _write_row(writer, strategy_class, exit_mode, "OOS", oos_result)
                f.flush()

    print(f"\nDone. Results in {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
