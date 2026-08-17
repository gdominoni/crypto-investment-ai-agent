"""Statistical filter + ranking hierarchy for Module B backtest candidates.

Ranking hierarchy (fixed by the project spec, applied identically here and
during live/dry-run monitoring in Phase 8): Win Rate (desc) -> Sortino
Ratio (desc) -> Net Profit after fees (desc). A candidate must first clear
a *dynamic* minimum trade-count filter before it's ranked at all -- a
strategy with too few trades can look great by chance alone.
"""

import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BacktestResult:
    strategy_name: str
    total_trades: int
    win_rate: float  # 0-1
    sortino_ratio: float
    net_profit_abs: float  # stake currency, after fees (freqtrade nets out trading fees)
    backtest_days: int


def dynamic_min_trade_count(
    backtest_days: int,
    confidence: float = 0.95,
    margin_of_error: float = 0.10,
    min_trades_per_week: float = 1.0,
) -> int:
    """Two independent floors, combined by taking the stricter one:

    1. Statistical floor: the minimum sample size needed to estimate a win
       rate within +/- margin_of_error at the given confidence level
       (worst-case variance, p=0.5) -- the standard margin-of-error formula
       for a proportion.
    2. Activity floor: the strategy must average at least min_trades_per_week
       over the backtest window. Guards against a strategy that clears (1)
       on a handful of lucky trades spread across a very long window.

    Genuinely "dynamic": both floors scale with backtest_days, so a longer
    backtest is held to a higher bar than a short one.
    """
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)
    statistical_floor = math.ceil((z**2) * 0.25 / (margin_of_error**2))
    activity_floor = math.ceil((backtest_days / 7) * min_trades_per_week)
    return max(statistical_floor, activity_floor)


def load_backtest_result(zip_path: Path, strategy_name: str) -> BacktestResult:
    with zipfile.ZipFile(zip_path) as zf:
        json_name = next(n for n in zf.namelist() if n.endswith(".json") and "_config" not in n)
        data = json.loads(zf.read(json_name))
    stats = data["strategy"][strategy_name]
    return BacktestResult(
        strategy_name=strategy_name,
        total_trades=stats["total_trades"],
        win_rate=stats["winrate"],
        sortino_ratio=stats["sortino"],
        net_profit_abs=stats["profit_total_abs"],
        backtest_days=stats["backtest_days"],
    )


def filter_and_rank(results: list[BacktestResult]) -> list[BacktestResult]:
    eligible = [r for r in results if r.total_trades >= dynamic_min_trade_count(r.backtest_days)]
    return sorted(eligible, key=lambda r: (r.win_rate, r.sortino_ratio, r.net_profit_abs), reverse=True)
