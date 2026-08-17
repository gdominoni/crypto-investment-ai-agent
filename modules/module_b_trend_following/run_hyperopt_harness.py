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

For each run: hyperopt searches "buy" and "sell" spaces on IS data: the
discovered best parameters are then re-backtested cleanly on IS (for a
parseable BacktestResult) and validated against the untouched OOS period.
Results are collected into one consolidated IS-vs-OOS matrix.

NOT YET RUN. Per instruction, this harness is for review before any
hyperopt session launches.

Run locally with:
    python -m modules.module_b_trend_following.run_hyperopt_harness
"""

import json
import subprocess
from datetime import date
from pathlib import Path

from modules.module_b_trend_following.candidate_ranking import BacktestResult, load_backtest_result
from modules.module_b_trend_following.oos_split import split_is_oos

MODULE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = MODULE_DIR / "user_data" / "backtest_results"
STRATEGIES_DIR = MODULE_DIR / "user_data" / "strategies"

DATA_START = date(2023, 8, 20)
DATA_END = date(2026, 8, 17)

EPOCHS = 200
HYPEROPT_LOSS = "ProjectHierarchyLoss"

# 8 base combos: (strategy class name, strategy filename)
COMBOS = [
    ("Combo1ClassicRsiPriceBBVolume", "combo1_classicrsi_pricebb_volume"),
    ("Combo2ClassicRsiPriceBBCmf", "combo2_classicrsi_pricebb_cmf"),
    ("Combo3ClassicRsiKeltnerVolume", "combo3_classicrsi_keltner_volume"),
    ("Combo4ClassicRsiKeltnerCmf", "combo4_classicrsi_keltner_cmf"),
    ("Combo5RsiBBPriceBBVolume", "combo5_rsibb_pricebb_volume"),
    ("Combo6RsiBBPriceBBCmf", "combo6_rsibb_pricebb_cmf"),
    ("Combo7RsiBBKeltnerVolume", "combo7_rsibb_keltner_volume"),
    ("Combo8RsiBBKeltnerCmf", "combo8_rsibb_keltner_cmf"),
]

EXIT_MODES = {
    "null": {"stoploss": -0.99, "roi": {"0": 100.0}},
    "wide": {"stoploss": -0.15, "roi": {"0": 0.15}},
}


def _set_exit_mode(strategy_class: str, strategy_filename: str, exit_mode: str) -> None:
    """Pre-writes stoploss/roi into the strategy's auto-loaded parameter
    file. Hyperopt only searches the spaces passed via --spaces (buy,
    sell here) -- stoploss/roi stay fixed at whatever this file says for
    the whole run and are carried through, unmodified, into the exported
    "best params" file at the end. Confirmed against this project's
    earlier observed behavior (trend_ema_adx.json): a space not included
    in --spaces is preserved from the loaded file, not reset to a class
    default.
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


def _run_hyperopt(strategy_class: str, timerange: str) -> None:
    subprocess.run(
        [
            "docker", "compose", "--env-file", "../../.env", "run", "--rm", "freqtrade",
            "hyperopt",
            "--config", "user_data/config.json",
            "--strategy", strategy_class,
            "--hyperopt-loss", HYPEROPT_LOSS,
            "--spaces", "buy", "sell",
            "-i", "1h",
            "--timerange", timerange,
            "-e", str(EPOCHS),
            "-j", "-1",
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


def main() -> None:
    split = split_is_oos(DATA_START, DATA_END, oos_months=12)
    results: list[dict] = []

    for strategy_class, strategy_filename in COMBOS:
        for exit_mode in EXIT_MODES:
            print(f"\n=== {strategy_class} / exit={exit_mode} ===")
            _set_exit_mode(strategy_class, strategy_filename, exit_mode)

            print("  Running hyperopt on IS data...")
            _run_hyperopt(strategy_class, split.is_timerange)

            print("  Re-backtesting IS with discovered params...")
            is_zip = _run_backtest(strategy_class, split.is_timerange)
            is_result = load_backtest_result(is_zip, strategy_class)

            print("  Validating against OOS...")
            oos_zip = _run_backtest(strategy_class, split.oos_timerange)
            oos_result = load_backtest_result(oos_zip, strategy_class)

            results.append(
                {"strategy": strategy_class, "exit_mode": exit_mode, "IS": is_result, "OOS": oos_result}
            )

    _print_matrix(results)


def _print_matrix(results: list[dict]) -> None:
    print("\n\n" + "=" * 100)
    print("CONSOLIDATED IS vs OOS MATRIX")
    print("=" * 100)

    def fmt(r: BacktestResult) -> str:
        profit_pct = r.net_profit_abs / 100  # dry_run_wallet=10000, so /100 == /10000*100
        return f"trades={r.total_trades:<4} win={r.win_rate:.1%} sortino={r.sortino_ratio:>6.2f} profit={profit_pct:>7.2f}%"

    for r in results:
        print(f"{r['strategy']:<32} exit={r['exit_mode']:<6}")
        print(f"  IS:  {fmt(r['IS'])}")
        print(f"  OOS: {fmt(r['OOS'])}")


if __name__ == "__main__":
    main()
