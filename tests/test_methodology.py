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
    classify_status, compute_anchors, concentration_check, path_outcome, report,
    shock_zscore_series, sortino_ratio,
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
    def test_a_candidate_that_never_loses_scores_infinite_not_nan(self):
        """`downside_dev == 0` has two genuinely different causes that must
        not return the same thing: an empty sample (nothing to say -> nan)
        versus a sample with no losing trade at all (unboundedly good ->
        +inf). Collapsing both into nan made classify_status REJECT a
        flawless candidate, since it treats a nan Sortino as unusable."""
        assert sortino_ratio(np.array([0.01, 0.02, 0.03])) == float("inf")

    def test_an_empty_sample_is_nan_not_infinite(self):
        assert np.isnan(sortino_ratio(np.array([])))

    def test_no_downside_and_no_upside_is_nan(self):
        assert np.isnan(sortino_ratio(np.array([0.0, 0.0])))

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

    def test_no_positive_return_anywhere_is_unassessable_not_a_pass(self):
        """A candidate losing money on every single group has nothing for
        one group to concentrate -- that is "cannot assess", not "passed".
        Returning False here let an all-negative candidate clear both
        concentration checks."""
        oos = pd.DataFrame({"group": ["A", "B", "C"], "net_return": [-0.1, -0.5, -0.5]})
        assert concentration_check(oos, "group")["concentrated"] is None

    def test_measures_whichever_return_column_the_caller_names(self):
        """Acceptance is decided on pattern_significance's forward returns,
        so concentration must be measurable on the same quantity -- the two
        bases genuinely disagree on identical events."""
        events = pd.DataFrame({
            "group": ["A"] * 3 + ["B"] * 3,
            "net_return": [0.30, 0.30, 0.30, 0.01, 0.01, 0.01],   # TP/SL view: A dominates
            "forward_return": [0.05] * 6,                          # forward view: even
        })
        assert concentration_check(events, "group")["concentrated"] is True
        assert concentration_check(events, "group", value_col="forward_return")["concentrated"] is False


class TestClassifyStatus:
    """classify_status's acceptance gate is pattern_significance now, not
    Sortino/win_rate (see the function's own docstring for why) -- `rep`
    below is only ever used for the sample-size gate and for the
    Sortino-is-NaN early exit; a `rep` with n > min_report_events and a
    real (non-NaN) sortino is otherwise just a stand-in for "the backtest
    ran fine", since its own P&L numbers no longer decide the verdict."""
    cfg = MethodologyConfig(horizons=(1, 3, 7))
    ok_rep = {"n": 150, "sortino": 5.0, "strict_win_rate": 0.9, "win_rate": 0.9}
    # `excess_return` is part of the contract now, not decoration: the effect
    # must point the SAME way as the direction the candidate trades.
    sig_pattern = {"status": "ok", "significant": True, "mfe_mae_ratio": 2.0, "excess_return": 0.03}

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
        pattern = {**self.sig_pattern, "significant": False}
        assert classify_status(self.ok_rep, {"concentrated": False}, {"concentrated": False}, pattern, self.cfg) == "rejected"

    def test_unfavorable_mfe_mae_ratio_is_watch_even_when_significant(self):
        pattern = {**self.sig_pattern, "mfe_mae_ratio": 0.8}
        assert classify_status(self.ok_rep, {"concentrated": False}, {"concentrated": False}, pattern, self.cfg) == "watch"

    def test_significant_and_favorable_risk_and_not_concentrated_is_accepted(self):
        assert classify_status(self.ok_rep, {"concentrated": False}, {"concentrated": False}, self.sig_pattern, self.cfg) == "accepted"

    def test_a_pattern_running_opposite_to_its_own_direction_is_never_accepted(self):
        """The headline audit finding. `_forward_return` signs its output by
        direction, so a NEGATIVE excess return means the effect runs the wrong
        way for what this candidate trades. Before the fix, `excess_return` was
        computed, shown to humans, and never read by the gate -- a pattern with
        p=0.001, a favorable risk path and a -5% excess return was `accepted`.
        Four of six real static candidates were 'significant' this way."""
        wrong_way = {**self.sig_pattern, "p_value": 0.001, "excess_return": -0.05}
        assert classify_status(self.ok_rep, {"concentrated": False}, {"concentrated": False}, wrong_way, self.cfg) == "rejected"

    def test_a_missing_excess_return_is_rejected_not_silently_accepted(self):
        no_excess = {k: v for k, v in self.sig_pattern.items() if k != "excess_return"}
        assert classify_status(self.ok_rep, {"concentrated": False}, {"concentrated": False}, no_excess, self.cfg) == "rejected"

    def test_unassessable_concentration_holds_at_watch_rather_than_passing(self):
        """`concentrated=None` means "cannot assess" (no positive return
        anywhere for a single group to concentrate), NOT "passed" -- an
        all-negative candidate used to sail through both concentration
        checks because they returned False."""
        assert classify_status(self.ok_rep, {"concentrated": None}, {"concentrated": False}, self.sig_pattern, self.cfg) == "watch"

    def test_an_infinite_sortino_is_not_treated_as_unusable(self):
        """A candidate with no losing trade at all now scores +inf rather
        than nan, and must not be rejected by the nan-Sortino early exit."""
        rep = {**self.ok_rep, "sortino": float("inf")}
        assert classify_status(rep, {"concentrated": False}, {"concentrated": False}, self.sig_pattern, self.cfg) == "accepted"


class TestPathOutcome:
    def test_an_incomplete_horizon_is_nan_not_a_silently_clamped_short_hold(self):
        """path_outcome used to clamp exit_loc to the last available bar, so
        a live test resolved at the very edge of the data returned a partial
        hold dressed up as a full-horizon result -- with a real-looking
        forward_return. A caller must be able to tell "not ready yet" from
        "resolved", so it can leave the test open instead of recording it."""
        ohlc = _flat_ohlc(10)
        out = path_outcome(100.0, 8, ohlc, "long", 21)
        assert np.isnan(out["forward_return"])

    def test_a_complete_horizon_still_resolves_normally(self):
        ohlc = _flat_ohlc(30)
        out = path_outcome(100.0, 2, ohlc, "long", 7)
        assert not np.isnan(out["forward_return"])


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


class TestBenjaminiHochberg:
    """Testing ~98 candidates at p<0.05 is EXPECTED to yield ~5
    'significant' results with no real effect behind any of them. That
    is the failure this project's whole methodology exists to prevent."""

    def test_a_family_of_pure_nulls_yields_no_discoveries(self):
        from candidates.methodology import benjamini_hochberg
        rng = np.random.default_rng(0)
        # uniform p-values are exactly what a family of true nulls produces
        nulls = list(rng.uniform(0, 1, 98))
        raw_hits = sum(p < 0.05 for p in nulls)
        bh_hits = sum(benjamini_hochberg(nulls))
        assert raw_hits >= 1, "fixture should show the raw threshold firing on noise"
        assert bh_hits == 0, "BH must not report discoveries in a family of pure nulls"

    def test_a_genuinely_strong_result_still_survives(self):
        from candidates.methodology import benjamini_hochberg
        rng = np.random.default_rng(1)
        pvals = [1e-6] + list(rng.uniform(0.2, 1.0, 40))
        assert benjamini_hochberg(pvals)[0] is True

    def test_it_is_never_more_permissive_than_the_raw_threshold(self):
        """The demotion-only property the callers rely on: BH may remove
        an `accepted` candidate but must never create one."""
        from candidates.methodology import benjamini_hochberg
        rng = np.random.default_rng(2)
        pvals = list(rng.uniform(0, 0.2, 60))
        for p, survived in zip(pvals, benjamini_hochberg(pvals)):
            if survived:
                assert p < 0.05

    def test_missing_p_values_never_survive(self):
        from candidates.methodology import benjamini_hochberg
        assert benjamini_hochberg([None, float("nan"), 1e-9]) == [False, False, True]

    def test_matches_the_canonical_1995_worked_example(self):
        """Benjamini & Hochberg 1995, Table 1: 15 hypotheses at alpha=0.05
        yield exactly 4 discoveries. Pins the implementation to the
        published answer rather than to my own reading of the algorithm."""
        from candidates.methodology import benjamini_hochberg
        p = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
             0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000]
        assert sum(benjamini_hochberg(p, 0.05)) == 4

    def test_demotion_pass_downgrades_a_lone_marginal_result(self):
        """The realistic case: ONE marginal p=0.04 among 39 nulls. Alone
        it clears p<0.05; against the family it is exactly the false
        positive a raw threshold is expected to manufacture."""
        from candidates.methodology import apply_fdr_demotion
        rng = np.random.default_rng(7)
        rows = [{"candidate": "marginal", "status": "accepted", "pattern_p_value": 0.04}]
        rows += [{"candidate": f"n{i}", "status": "rejected", "pattern_p_value": float(p)}
                 for i, p in enumerate(rng.uniform(0.2, 1.0, 39))]
        live = {"candidates": {"marginal": {}}}
        apply_fdr_demotion(rows, live)
        assert rows[0]["status"] == "rejected" and rows[0]["fdr_demoted"] is True
        assert live["candidates"] == {}, "a demoted candidate must not stay able to open live tests"

    def test_a_pile_of_identical_marginal_p_values_is_NOT_demoted(self):
        """The complement, and it is correct: 40 p-values all at 0.04 is
        collectively very unlikely under a global null, so BH keeps them.
        Recorded so nobody later 'fixes' this into over-correction."""
        from candidates.methodology import apply_fdr_demotion
        rows = [{"candidate": f"c{i}", "status": "accepted", "pattern_p_value": 0.04} for i in range(40)]
        apply_fdr_demotion(rows, None)
        assert all(r["status"] == "accepted" for r in rows)


class TestTelegramChunkingIsHtmlSafe:
    """The SECOND time this silent-drop failure has been found. A chunk
    boundary landing inside a `<b>...</b>` pair produces malformed HTML,
    Telegram rejects it with a 400, and since almost no caller checks
    `_send`'s return value the message simply never arrives."""

    def _line(self, n):
        from candidates.methodology import _insufficient_data_block
        return _insufficient_data_block([(f"sonnet_proposed_condition_name_{i}", {"n": 0}) for i in range(n)])[-1]

    def test_the_unbounded_name_list_line_never_splits_a_tag(self):
        """`_insufficient_data_block` emits ONE line listing every zero-N
        candidate, each bolded. At 90 tracked candidates it is ~4,900
        characters -- comfortably past the limit, and the old raw
        `line[:limit]` cut produced 65 `<b>` against 64 `</b>`."""
        from telegram.bot import _chunk_message
        for n in (60, 90, 150, 300):
            for chunk in _chunk_message(self._line(n)):
                assert chunk.count("<b>") == chunk.count("</b>"), f"unbalanced markup at n={n}"

    def test_chunking_never_loses_content(self):
        from telegram.bot import _chunk_message
        line = self._line(150)
        assert "".join(_chunk_message(line)).replace(" ", "") == line.replace(" ", "")

    def test_no_chunk_exceeds_telegrams_hard_limit(self):
        from telegram.bot import _chunk_message, TELEGRAM_MAX_MESSAGE_LENGTH
        for chunk in _chunk_message(self._line(300)):
            assert len(chunk) <= TELEGRAM_MAX_MESSAGE_LENGTH

    def test_a_single_unbreakable_token_still_terminates(self):
        """No safe boundary exists inside one enormous token -- the cut
        must fall back to the hard limit rather than loop forever."""
        from telegram.bot import _chunk_message, _SAFE_CHUNK_LENGTH
        chunks = _chunk_message("x" * (_SAFE_CHUNK_LENGTH * 3))
        assert len(chunks) == 3 and all(len(c) <= _SAFE_CHUNK_LENGTH for c in chunks)
