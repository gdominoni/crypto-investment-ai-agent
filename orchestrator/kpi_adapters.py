"""Adapters converting each module's own result type into the shared
ModuleKPI interface capital_allocator.py ranks across modules.

Module A is a fundamentally different kind of strategy (yield-harvesting,
not directional), so its adapter maps funding-rate statistics onto the
same shape rather than reusing a backtest result -- see
modules/module_a_cash_carry/funding_analysis.py's `_annualized_sortino_ratio`
for why that mapping is meaningful, not just a type-coercion.

Modules B and C both produce a Freqtrade-format backtest result
(candidate_ranking.load_backtest_result works for either, regardless of
which module's strategy produced it), so they share one adapter.
"""

from modules.module_a_cash_carry.funding_analysis import FundingYieldReport
from modules.module_b_trend_following.candidate_ranking import BacktestResult, dynamic_min_trade_count
from orchestrator.capital_allocator import ModuleKPI

# Matches dry_run_wallet in both module_b and module_c's Freqtrade configs --
# net_profit_pct needs a common denominator to be comparable across modules.
STARTING_WALLET_USDT = 10_000

# Funding settles every 8h (3x/day); require at least ~10 days of history
# before trusting the yield estimate at all.
MIN_FUNDING_PERIODS = 30


def from_funding_yield_report(report: FundingYieldReport, module_name: str = "module_a_cash_carry") -> ModuleKPI:
    return ModuleKPI(
        module_name=module_name,
        win_rate=report.positive_funding_pct,
        sortino_ratio=report.sortino_ratio,
        net_profit_pct=report.annualized_yield_net_of_fees_pct,
        sample_size=report.periods,
        meets_significance_threshold=report.periods >= MIN_FUNDING_PERIODS,
    )


def from_backtest_result(result: BacktestResult, module_name: str) -> ModuleKPI:
    return ModuleKPI(
        module_name=module_name,
        win_rate=result.win_rate,
        sortino_ratio=result.sortino_ratio,
        net_profit_pct=(result.net_profit_abs / STARTING_WALLET_USDT) * 100,
        sample_size=result.total_trades,
        meets_significance_threshold=result.total_trades >= dynamic_min_trade_count(result.backtest_days),
    )
