"""Tests for modules/module_b_trend_following/candidate_ranking.py."""

from modules.module_b_trend_following.candidate_ranking import (
    BacktestResult,
    dynamic_min_trade_count,
    filter_and_rank,
)


def test_dynamic_min_trade_count_scales_with_backtest_length():
    short = dynamic_min_trade_count(backtest_days=90)
    long = dynamic_min_trade_count(backtest_days=900)
    assert long > short


def test_dynamic_min_trade_count_respects_statistical_floor():
    n = dynamic_min_trade_count(backtest_days=1, confidence=0.95, margin_of_error=0.10)
    assert n >= 96  # z=1.96, margin=0.10 -> statistical floor ~96


def test_filter_excludes_candidates_below_threshold():
    thin = BacktestResult(
        "Thin", total_trades=5, win_rate=0.9, sortino_ratio=3.0, net_profit_abs=1000, backtest_days=365
    )
    assert filter_and_rank([thin]) == []


def test_filter_and_rank_orders_by_hierarchy():
    a = BacktestResult("A", total_trades=200, win_rate=0.55, sortino_ratio=1.0, net_profit_abs=500, backtest_days=365)
    b = BacktestResult("B", total_trades=200, win_rate=0.60, sortino_ratio=0.5, net_profit_abs=100, backtest_days=365)
    c = BacktestResult("C", total_trades=200, win_rate=0.55, sortino_ratio=2.0, net_profit_abs=100, backtest_days=365)
    ranked = filter_and_rank([a, b, c])
    assert [r.strategy_name for r in ranked] == ["B", "C", "A"]
