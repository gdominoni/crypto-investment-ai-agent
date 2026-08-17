"""Tests for orchestrator/kpi_adapters.py."""

from modules.module_a_cash_carry.funding_analysis import FundingYieldReport
from modules.module_b_trend_following.candidate_ranking import BacktestResult
from orchestrator.kpi_adapters import STARTING_WALLET_USDT, from_backtest_result, from_funding_yield_report


def test_from_funding_yield_report_maps_positive_funding_to_win_rate():
    report = FundingYieldReport(
        symbol="BTCUSDT",
        periods=3285,
        mean_funding_rate=0.00006,
        positive_funding_pct=0.846,
        annualized_yield_gross_pct=7.23,
        annualized_yield_net_of_fees_pct=6.95,
        sortino_ratio=136.0,
        is_currently_attractive=True,
        suggested_min_opening_arbitrage_pct=0.42,
    )
    kpi = from_funding_yield_report(report)
    assert kpi.win_rate == 0.846
    assert kpi.net_profit_pct == 6.95
    assert kpi.sortino_ratio == 136.0
    assert kpi.meets_significance_threshold is True


def test_from_funding_yield_report_flags_insufficient_history():
    report = FundingYieldReport(
        symbol="NEWCOIN",
        periods=5,
        mean_funding_rate=0.0001,
        positive_funding_pct=1.0,
        annualized_yield_gross_pct=10.0,
        annualized_yield_net_of_fees_pct=9.5,
        sortino_ratio=50.0,
        is_currently_attractive=True,
        suggested_min_opening_arbitrage_pct=0.42,
    )
    kpi = from_funding_yield_report(report)
    assert kpi.meets_significance_threshold is False


def test_from_backtest_result_computes_net_profit_pct_of_starting_wallet():
    result = BacktestResult(
        strategy_name="Test",
        total_trades=200,
        win_rate=0.5,
        sortino_ratio=1.2,
        net_profit_abs=500.0,
        backtest_days=365,
    )
    kpi = from_backtest_result(result, module_name="module_b_trend_following")
    assert kpi.net_profit_pct == (500.0 / STARTING_WALLET_USDT) * 100
    assert kpi.module_name == "module_b_trend_following"


def test_from_backtest_result_flags_insufficient_trades():
    result = BacktestResult(
        strategy_name="Test", total_trades=5, win_rate=0.9, sortino_ratio=2.0, net_profit_abs=1000, backtest_days=365
    )
    kpi = from_backtest_result(result, module_name="module_c_volatility_ml")
    assert kpi.meets_significance_threshold is False
