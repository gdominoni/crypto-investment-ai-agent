"""Runs the dynamic capital allocator against each module's real, current
results -- funding yield analysis for Module A, and the most recent
Freqtrade backtest result on disk for Modules B and C.

Run locally with:
    python -m orchestrator.run_allocation
"""

from pathlib import Path

from modules.module_a_cash_carry.funding_analysis import analyze_funding_yield
from modules.module_b_trend_following.candidate_ranking import load_backtest_result
from orchestrator.capital_allocator import allocate_capital
from orchestrator.kpi_adapters import from_backtest_result, from_funding_yield_report

REPO_ROOT = Path(__file__).resolve().parent.parent


def _latest_backtest_zip(module_dir: str) -> Path:
    results_dir = REPO_ROOT / "modules" / module_dir / "user_data" / "backtest_results"
    return sorted(results_dir.glob("backtest-result-*.zip"))[-1]


def main() -> None:
    # Module A: use whichever pair currently shows the better net yield --
    # that's the pair Module A would actually deploy capital to first.
    funding_reports = [analyze_funding_yield(s) for s in ["BTCUSDT", "ETHUSDT"]]
    best_funding = max(funding_reports, key=lambda r: r.annualized_yield_net_of_fees_pct)
    module_a_kpi = from_funding_yield_report(best_funding, module_name="module_a_cash_carry")

    module_b_kpi = from_backtest_result(
        load_backtest_result(_latest_backtest_zip("module_b_trend_following"), "TrendEmaAdx"),
        module_name="module_b_trend_following",
    )
    module_c_kpi = from_backtest_result(
        load_backtest_result(_latest_backtest_zip("module_c_volatility_ml"), "VolatilityGateSignal"),
        module_name="module_c_volatility_ml",
    )

    decision = allocate_capital([module_a_kpi, module_b_kpi, module_c_kpi])

    print(f"Module A ({best_funding.symbol}): win_rate={module_a_kpi.win_rate:.1%} "
          f"sortino={module_a_kpi.sortino_ratio:.2f} net_profit={module_a_kpi.net_profit_pct:.2f}% "
          f"(n={module_a_kpi.sample_size})")
    print(f"Module B: win_rate={module_b_kpi.win_rate:.1%} "
          f"sortino={module_b_kpi.sortino_ratio:.2f} net_profit={module_b_kpi.net_profit_pct:.2f}% "
          f"(n={module_b_kpi.sample_size})")
    print(f"Module C: win_rate={module_c_kpi.win_rate:.1%} "
          f"sortino={module_c_kpi.sortino_ratio:.2f} net_profit={module_c_kpi.net_profit_pct:.2f}% "
          f"(n={module_c_kpi.sample_size})")

    print("\n=== Allocation decision ===")
    for module, weight in decision.weights.items():
        print(f"{module}: {weight:.1%}")
    for module, reason in decision.excluded_modules.items():
        print(f"{module}: excluded ({reason})")
    print(f"Cash reserve: {decision.cash_reserve_pct:.1%}")


if __name__ == "__main__":
    main()
