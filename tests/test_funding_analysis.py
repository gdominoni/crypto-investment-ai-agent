"""Tests for modules/module_a_cash_carry/funding_analysis.py, using
synthetic funding-rate series so they don't depend on downloaded data."""

import pandas as pd

from modules.module_a_cash_carry.funding_analysis import ROUND_TRIP_COST_PCT, _compute_report


def _synthetic_funding(rate: float, periods: int = 300) -> pd.DataFrame:
    return pd.DataFrame({"fundingRate": [rate] * periods})


def test_consistently_positive_funding_is_attractive():
    # 0.01% per 8h period annualizes well above the round-trip fee cost.
    df = _synthetic_funding(rate=0.0001)
    report = _compute_report(df, "TEST", recent_window_days=30)
    assert report.positive_funding_pct == 1.0
    assert report.annualized_yield_gross_pct > 0
    assert report.is_currently_attractive is True


def test_negative_funding_is_not_attractive():
    df = _synthetic_funding(rate=-0.0001)
    report = _compute_report(df, "TEST", recent_window_days=30)
    assert report.positive_funding_pct == 0.0
    assert report.is_currently_attractive is False


def test_net_yield_is_lower_than_gross_by_round_trip_cost():
    df = _synthetic_funding(rate=0.0001)
    report = _compute_report(df, "TEST", recent_window_days=30)
    expected_net = report.annualized_yield_gross_pct - ROUND_TRIP_COST_PCT * 100
    assert round(report.annualized_yield_net_of_fees_pct, 6) == round(expected_net, 6)


def test_suggested_threshold_scales_with_round_trip_cost():
    df = _synthetic_funding(rate=0.0001)
    report = _compute_report(df, "TEST", recent_window_days=30)
    assert report.suggested_min_opening_arbitrage_pct > ROUND_TRIP_COST_PCT * 100
