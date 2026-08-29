"""Unit tests for candidates/methodology.py's core invariants -- the
claims this project's README makes about causality-safety, duration-
bucketed exits, and honest reporting are only real if something checks
them on every change. Uses small, synthetic OHLC frames (not the real
parquet data) so these stay fast and fully deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from candidates.methodology import (
    MethodologyConfig, barrier_prices, bucket_for_elapsed, build_events, classify_regime,
    classify_status, compute_anchors, concentration_check, report, shock_zscore_series, sortino_ratio,
)


def _flat_ohlc(n=20, price=100.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "open": price, "high": price * 1.01, "low": price * 0.99, "close": price,
        "volume": 1000.0,
    }, index=idx)


def _random_walk_ohlc(n=300, seed=42):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.02, n)
    closes = 100 * np.cumprod(1 + rets)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "open": closes, "high": closes * 1.01, "low": closes * 0.99, "close": closes,
        "volume": 1000.0,
    }, index=idx)


class TestBuildEventsCausality:
    def test_entry_is_the_bar_after_the_trigger_never_the_same_bar(self):
        ohlc = _flat_ohlc(20)
        trigger = pd.Series(False, index=ohlc.index)
        trigger.iloc[5] = True
        events = build_events(ohlc, trigger, "long", (1, 3))
        assert len(events) == 1
        assert events.iloc[0]["entry_loc"] == 6
        assert events.iloc[0]["entry_time"] == ohlc.index[6]

    def test_trigger_on_the_last_bar_produces_no_event(self):
        """No 'next bar' to enter on if the trigger fires on the last bar in the frame."""
        ohlc = _flat_ohlc(10)
        trigger = pd.Series(False, index=ohlc.index)
        trigger.iloc[-1] = True
        events = build_events(ohlc, trigger, "long", (1, 3))
        assert len(events) == 0

    def test_mfe_mae_excludes_the_entry_bars_own_range(self):
        """A spike on the ENTRY bar's own high must not count toward MFE --
        only bars strictly after entry do."""
        ohlc = _flat_ohlc(20)
        ohlc.loc[ohlc.index[6], "high"] = 500.0  # entry bar (trigger at 5 -> entry at 6) -- must be ignored
        trigger = pd.Series(False, index=ohlc.index)
        trigger.iloc[5] = True
        events = build_events(ohlc, trigger, "long", (1,))
        assert events.iloc[0]["mfe_1"] < 0.5  # would be ~4.0 if the entry bar's own spike leaked into MFE


class TestAnchorsAndBarriers:
    def test_compute_anchors_averages_mfe_and_abs_mae_per_horizon(self):
        events = pd.DataFrame({"mfe_1": [0.02, 0.04], "mae_1": [-0.01, -0.03]})
        anchors = compute_anchors(events, (1,))
        assert anchors[1]["mfe"] == pytest.approx(0.03)
        assert anchors[1]["mae"] == pytest.approx(0.02)  # abs().mean() -- sign-stripped, not signed mean

    def test_barrier_prices_long_and_short_are_mirrored(self):
        anchors = {1: {"mfe": 0.05, "mae": 0.03}}
        tp_long, sl_long, _, _ = barrier_prices(100.0, "long", anchors, 1, 1.0, 1.0)
        tp_short, sl_short, _, _ = barrier_prices(100.0, "short", anchors, 1, 1.0, 1.0)
        assert tp_long > 100.0 and sl_long < 100.0
        assert tp_short < 100.0 and sl_short > 100.0


class TestBucketForElapsed:
    def test_returns_the_first_horizon_that_covers_elapsed_time(self):
        assert bucket_for_elapsed(1, (1, 3, 7)) == 1
        assert bucket_for_elapsed(2, (1, 3, 7)) == 3
        assert bucket_for_elapsed(7, (1, 3, 7)) == 7

    def test_returns_none_past_the_last_bucket(self):
        assert bucket_for_elapsed(8, (1, 3, 7)) is None


class TestSortinoRatio:
    def test_all_positive_returns_has_no_downside_deviation_to_divide_by(self):
        assert np.isnan(sortino_ratio(np.array([0.01, 0.02, 0.03])))

    def test_penalizes_a_downside_outlier_more_than_an_equal_upside_one(self):
        upside_outlier = np.array([0.01, -0.01, 0.01, 0.50])
        downside_outlier = np.array([0.01, -0.01, 0.01, -0.50])
        assert sortino_ratio(upside_outlier) > sortino_ratio(downside_outlier)


class TestConcentrationCheck:
    def test_flags_a_result_carried_by_one_group(self):
        oos = pd.DataFrame({"group": ["A"] * 9 + ["B"], "net_return": [0.01] * 9 + [-0.01]})
        result = concentration_check(oos, "group")
        assert result["concentrated"] is True
        assert result["dominant_group"] == "A"

    def test_does_not_flag_a_diversified_result(self):
        oos = pd.DataFrame({"group": ["A", "B", "C", "D"], "net_return": [0.01, 0.01, 0.01, 0.01]})
        result = concentration_check(oos, "group")
        assert result["concentrated"] is False


class TestClassifyStatus:
    """classify_status's acceptance gate is pattern_significance now, not
    Sortino/win_rate (see the function's own docstring for why) -- `rep`
    below is only ever used for the sample-size gate and for the
    Sortino-is-NaN early exit; a `rep` with n > min_report_events and a
    real (non-NaN) sortino is otherwise just a stand-in for "the backtest
    ran fine", since its own P&L numbers no longer decide the verdict."""
    cfg = MethodologyConfig(horizons=(1, 3, 7))
    ok_rep = {"n": 150, "sortino": 5.0, "strict_win_rate": 0.9, "win_rate": 0.9}
    sig_pattern = {"status": "ok", "significant": True, "mfe_mae_ratio": 2.0}

    def test_between_10_and_min_report_events_is_rejected_not_insufficient(self):
        rep = {**self.ok_rep, "n": 15}
        assert classify_status(rep, {"concentrated": False}, {"concentrated": False}, self.sig_pattern, self.cfg) == "rejected"

    def test_under_10_events_is_insufficient_data(self):
        rep = {**self.ok_rep, "n": 5}
        assert classify_status(rep, {"concentrated": False}, {"concentrated": False}, self.sig_pattern, self.cfg) == "insufficient_data"

    def test_n_at_exactly_the_threshold_is_rejected_not_accepted(self):
        rep = {**self.ok_rep, "n": 50}
        assert classify_status(rep, {"concentrated": False}, {"concentrated": False}, self.sig_pattern, self.cfg) == "rejected"

    def test_nan_sortino_in_rep_is_rejected_regardless_of_pattern(self):
        rep = {**self.ok_rep, "sortino": float("nan")}
        assert classify_status(rep, {"concentrated": False}, {"concentrated": False}, self.sig_pattern, self.cfg) == "rejected"

    def test_pattern_not_ok_is_watch(self):
        pattern = {"status": "insufficient_data"}
        assert classify_status(self.ok_rep, {"concentrated": False}, {"concentrated": False}, pattern, self.cfg) == "watch"

    def test_concentrated_result_is_watch_even_with_a_significant_pattern(self):
        assert classify_status(self.ok_rep, {"concentrated": True}, {"concentrated": False}, self.sig_pattern, self.cfg) == "watch"

    def test_not_significant_pattern_is_rejected(self):
        pattern = {"status": "ok", "significant": False, "mfe_mae_ratio": 2.0}
        assert classify_status(self.ok_rep, {"concentrated": False}, {"concentrated": False}, pattern, self.cfg) == "rejected"

    def test_unfavorable_mfe_mae_ratio_is_watch_even_when_significant(self):
        pattern = {"status": "ok", "significant": True, "mfe_mae_ratio": 0.8}
        assert classify_status(self.ok_rep, {"concentrated": False}, {"concentrated": False}, pattern, self.cfg) == "watch"

    def test_significant_and_favorable_risk_and_not_concentrated_is_accepted(self):
        assert classify_status(self.ok_rep, {"concentrated": False}, {"concentrated": False}, self.sig_pattern, self.cfg) == "accepted"


class TestShockRegime:
    def test_early_history_with_no_established_baseline_is_never_classified_shock(self):
        """Classifying early history as a shock by default would be a guess,
        not a measurement -- see the function's own docstring."""
        shock_z = shock_zscore_series(_flat_ohlc(10))
        assert classify_regime(shock_z, 2) == "normal"

    def test_shock_zscore_at_a_given_bar_never_changes_when_future_bars_are_added(self):
        """Every value at position `loc` must use only bars up to and
        including `loc` -- pins down the causal-safety guarantee the
        module's docstring claims, rather than trusting it by convention."""
        full = _random_walk_ohlc(300)
        z_full = shock_zscore_series(full)
        z_truncated = shock_zscore_series(full.iloc[:200])
        pd.testing.assert_series_equal(z_truncated, z_full.iloc[:200], check_names=False)


class TestReport:
    def test_win_rate_excludes_timeouts_strict_win_rate_counts_them_against(self):
        oos = pd.DataFrame({
            "outcome": ["win", "win", "loss", "timeout"],
            "net_return": [0.05, 0.05, -0.03, 0.01],
        })
        rep = report(oos)
        assert rep["win_rate"] == pytest.approx(2 / 3)
        assert rep["strict_win_rate"] == pytest.approx(2 / 4)

    def test_total_expectancy_sums_returns_rather_than_compounding_them(self):
        oos = pd.DataFrame({"outcome": ["win", "win"], "net_return": [0.10, 0.10]})
        rep = report(oos)
        assert rep["total_expectancy"] == pytest.approx(0.20)  # sum, not (1.1*1.1 - 1) = 0.21
