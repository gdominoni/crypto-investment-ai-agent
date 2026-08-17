"""Orchestrates Module B's Phase 9 Phase-1 coarse-grid screening: for
every (strategy family, timeframe, entry preset, exit preset)
combination, writes the strategy's auto-loaded parameter JSON, runs a
real Docker backtest for both the In-Sample and Out-of-Sample periods,
and appends results to a consolidated CSV as it goes (so progress
survives an interruption across a run this long).

Hyperopt is NOT run here -- hand-curated grid only, per the human
director's explicit instruction to defer any search/fine-tuning phase
until these raw results are reviewed.

Run locally with:
    python -m modules.module_b_trend_following.run_coarse_grid
"""

import csv
import json
import subprocess
from dataclasses import asdict
from datetime import date
from pathlib import Path

from modules.module_b_trend_following.candidate_ranking import load_backtest_result
from modules.module_b_trend_following.coarse_grid import ENTRY_GRIDS, EXIT_GRIDS, STRATEGIES, TIMEFRAMES
from modules.module_b_trend_following.oos_split import split_is_oos

MODULE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = MODULE_DIR / "user_data" / "backtest_results"
STRATEGIES_DIR = MODULE_DIR / "user_data" / "strategies"
OUTPUT_CSV = MODULE_DIR / "coarse_grid_results.csv"

DATA_START = date(2023, 8, 20)
DATA_END = date(2026, 8, 17)

# Freqtrade resolves a strategy's auto-loaded parameter file from the
# strategy's own .py file path (Path(self.__file__).with_suffix(".json")),
# NOT the class name -- confirmed by reading freqtrade/strategy/hyper.py's
# load_params_from_file() directly, after a first version of this script
# wrote <ClassName>.json and Freqtrade silently logged "Found no parameter
# file", making every "different" exit preset run against the class
# defaults instead (caught because two deliberately different presets
# produced bit-for-bit identical results -- that shouldn't happen, and
# didn't have an innocent explanation).
STRATEGY_FILENAMES = {
    "MeanReversionBBRSI": "mean_reversion_bb_rsi",
    "VolatilityBreakoutKCSqueeze": "volatility_breakout_kc_squeeze",
    "VolumeDrivenVWAPCMF": "volume_driven_vwap_cmf",
}

CSV_FIELDS = [
    "strategy", "timeframe", "entry_preset_idx", "exit_preset", "period",
    "total_trades", "win_rate", "sortino_ratio", "net_profit_pct", "backtest_days",
]


def _write_params_file(strategy: str, entry_preset: dict, exit_preset) -> None:
    roi = {"0": exit_preset.take_profit} if exit_preset.take_profit is not None else {"0": 100.0}
    params = {
        "strategy_name": strategy,
        "params": {
            "roi": roi,
            "stoploss": {"stoploss": exit_preset.stoploss},
            "trailing": {
                "trailing_stop": exit_preset.trailing_stop,
                "trailing_stop_positive": exit_preset.trailing_stop_positive or 0.01,
                "trailing_stop_positive_offset": exit_preset.trailing_stop_positive_offset or 0.02,
                "trailing_only_offset_is_reached": True,
            },
            "buy": entry_preset,
        },
    }
    filename = STRATEGY_FILENAMES[strategy]
    (STRATEGIES_DIR / f"{filename}.json").write_text(json.dumps(params, indent=2))


def _run_backtest(strategy: str, timeframe: str, timerange: str) -> Path:
    subprocess.run(
        [
            "docker", "compose", "--env-file", "../../.env", "run", "--rm", "freqtrade",
            "backtesting",
            "--config", "user_data/config.json",
            "--strategy", strategy,
            "-i", timeframe,
            "--timerange", timerange,
        ],
        cwd=MODULE_DIR,
        check=True,
        capture_output=True,
    )
    return sorted(RESULTS_DIR.glob("backtest-result-*.zip"))[-1]


def main() -> None:
    split = split_is_oos(DATA_START, DATA_END, oos_months=12)
    periods = {"IS": split.is_timerange, "OOS": split.oos_timerange}

    is_new_file = not OUTPUT_CSV.exists()
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new_file:
            writer.writeheader()

        total_combos = sum(
            len(ENTRY_GRIDS[strat][tf]) * len(EXIT_GRIDS[tf]) for strat in STRATEGIES for tf in TIMEFRAMES
        )
        done = 0

        for strategy in STRATEGIES:
            for timeframe in TIMEFRAMES:
                entry_presets = ENTRY_GRIDS[strategy][timeframe]
                exit_presets = EXIT_GRIDS[timeframe]
                for entry_idx, entry_preset in enumerate(entry_presets):
                    for exit_preset in exit_presets:
                        _write_params_file(strategy, entry_preset, exit_preset)
                        done += 1
                        print(
                            f"[{done}/{total_combos}] {strategy} {timeframe} "
                            f"entry#{entry_idx} exit={exit_preset.name}",
                            flush=True,
                        )
                        for period_name, timerange in periods.items():
                            try:
                                zip_path = _run_backtest(strategy, timeframe, timerange)
                                result = load_backtest_result(zip_path, strategy)
                                writer.writerow({
                                    "strategy": strategy,
                                    "timeframe": timeframe,
                                    "entry_preset_idx": entry_idx,
                                    "exit_preset": exit_preset.name,
                                    "period": period_name,
                                    "total_trades": result.total_trades,
                                    "win_rate": round(result.win_rate, 4),
                                    "sortino_ratio": round(result.sortino_ratio, 3),
                                    "net_profit_pct": round(result.net_profit_abs / 10000 * 100, 3),
                                    "backtest_days": result.backtest_days,
                                })
                                f.flush()
                            except Exception as exc:  # noqa: BLE001 -- log and keep the sweep going
                                print(f"  FAILED ({period_name}): {exc}", flush=True)

    print(f"\nDone. Results in {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
