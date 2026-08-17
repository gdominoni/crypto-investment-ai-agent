"""Orchestrates the Phase 4 backtest & candidate-ranking pipeline:

1. Splits the downloaded history into In-Sample (optimization) and
   Out-of-Sample (holdout) periods (oos_split.py).
2. Runs `freqtrade backtesting` in Docker for each candidate strategy,
   once per period.
3. Ranks candidates on their IS results using the project's fixed
   hierarchy (candidate_ranking.py): Win Rate -> Sortino -> Net Profit,
   after clearing the dynamic minimum trade-count filter.
4. Reports each candidate's OOS performance separately, as a validation
   check -- OOS never influences ranking, only whether a selected
   strategy is trusted to move forward.

Requires history already downloaded (see README.md) and Docker running.
Run locally with:
    python -m modules.module_b_trend_following.run_backtest
"""

import subprocess
from datetime import date
from pathlib import Path

from modules.module_b_trend_following.candidate_ranking import filter_and_rank, load_backtest_result
from modules.module_b_trend_following.oos_split import split_is_oos

MODULE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = MODULE_DIR / "user_data" / "backtest_results"
STRATEGIES = ["TrendEmaAdx"]

# Matches the downloaded history range -- see README.md for the download command.
DATA_START = date(2023, 8, 20)
DATA_END = date(2026, 8, 17)


def _run_freqtrade_backtest(strategy: str, timerange: str) -> Path:
    subprocess.run(
        [
            "docker", "compose", "--env-file", "../../.env", "run", "--rm", "freqtrade",
            "backtesting",
            "--config", "user_data/config.json",
            "--strategy", strategy,
            "--timerange", timerange,
        ],
        cwd=MODULE_DIR,
        check=True,
    )
    return sorted(RESULTS_DIR.glob("backtest-result-*.zip"))[-1]


def main() -> None:
    split = split_is_oos(DATA_START, DATA_END, oos_months=12)
    print(f"In-sample:     {split.is_timerange}")
    print(f"Out-of-sample: {split.oos_timerange}\n")

    is_results = []
    oos_results = {}
    for strategy in STRATEGIES:
        is_zip = _run_freqtrade_backtest(strategy, split.is_timerange)
        is_results.append(load_backtest_result(is_zip, strategy))

        oos_zip = _run_freqtrade_backtest(strategy, split.oos_timerange)
        oos_results[strategy] = load_backtest_result(oos_zip, strategy)

    ranked = filter_and_rank(is_results)

    print("=== In-Sample ranking (Win Rate -> Sortino -> Net Profit) ===")
    if not ranked:
        print("No candidate cleared the dynamic minimum trade-count filter.")
    for r in ranked:
        print(
            f"{r.strategy_name}: trades={r.total_trades} win_rate={r.win_rate:.1%} "
            f"sortino={r.sortino_ratio:.2f} net_profit={r.net_profit_abs:.2f} USDT"
        )

    print("\n=== Out-of-sample validation (not used for ranking) ===")
    for strategy, r in oos_results.items():
        print(
            f"{strategy}: trades={r.total_trades} win_rate={r.win_rate:.1%} "
            f"sortino={r.sortino_ratio:.2f} net_profit={r.net_profit_abs:.2f} USDT"
        )


if __name__ == "__main__":
    main()
