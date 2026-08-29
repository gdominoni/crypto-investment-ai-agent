"""Unit tests for llm_pipeline/novel_condition_tester.py's Clause/
ConditionSpec validation -- this whitelist is the entire boundary
between an LLM's proposal and code that actually runs; it must reject
anything outside it at construction time, not fail confusingly later.
"""
from __future__ import annotations

import pytest

from llm_pipeline.novel_condition_tester import SUPPORTED_INDICATORS, Clause, ConditionSpec, condition_desc


def test_every_supported_indicator_constructs_a_valid_clause():
    for indicator in SUPPORTED_INDICATORS:
        clause = Clause(indicator=indicator, op=">", threshold=1.0)
        assert clause.indicator == indicator


def test_unsupported_indicator_is_rejected():
    with pytest.raises(ValueError, match="Unsupported indicator"):
        Clause(indicator="totally_made_up", op=">", threshold=1.0)


def test_unsupported_operator_is_rejected():
    with pytest.raises(ValueError, match="Unsupported operator"):
        Clause(indicator="close_return_1d", op="==", threshold=1.0)


def test_invalid_direction_is_rejected():
    with pytest.raises(ValueError, match="direction must be"):
        ConditionSpec(label="x", clauses=(Clause(indicator="close_return_1d", op=">", threshold=1.0),), direction="sideways")


def test_empty_clauses_is_rejected():
    with pytest.raises(ValueError, match="at least one clause"):
        ConditionSpec(label="x", clauses=(), direction="long")


def test_multi_clause_spec_is_anded_in_its_description():
    spec = ConditionSpec(
        label="x",
        clauses=(Clause(indicator="rsi_14d", op="<", threshold=30.0), Clause(indicator="shock_zscore", op=">=", threshold=3.0)),
        direction="long",
    )
    desc = condition_desc(spec)
    assert " AND " in desc
    assert desc.count(" AND ") == 1  # two clauses, joined once -- not silently collapsed to one
    assert "below 30.0" in desc and "at least 3.0" in desc
    # indicator names are translated to plain language, not shown as raw variable names
    assert "rsi_14d" not in desc and "shock_zscore" not in desc
    # comparisons are translated to plain English too -- a raw "<"/">" HTML-escaped
    # to "&lt;"/"&gt;" was observed live to sometimes render as literal text in
    # Telegram instead of decoding back to the symbol; never generating the raw
    # symbol at all sidesteps that failure mode entirely
    assert "<" not in desc and ">" not in desc


class TestLaggedClauses:
    """`within_days` is what makes an ORDERED hypothesis expressible at
    all -- "crash, THEN news" versus "crash AND news on the same day".
    Before it, the grammar could only ever say the second."""

    def test_default_is_same_bar_so_existing_specs_are_unchanged(self):
        assert Clause(indicator="rsi_14d", op="<", threshold=30.0).within_days == 0

    def test_negative_or_oversized_lookback_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            Clause(indicator="rsi_14d", op="<", threshold=30.0, within_days=-1)
        with pytest.raises(ValueError, match="at most"):
            Clause(indicator="rsi_14d", op="<", threshold=30.0, within_days=999)

    def test_a_lagged_clause_stays_true_for_its_window_and_never_before(self):
        """The window looks strictly BACKWARD: it must extend the signal
        forward in time from the event, never make it true beforehand."""
        import numpy as np, pandas as pd
        from llm_pipeline.novel_condition_tester import clause_signal
        idx = pd.date_range("2024-01-01", periods=12, freq="D")
        close = np.array([100, 100, 100, 100, 100, 80, 100, 100, 100, 100, 100, 100], dtype=float)
        df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                           "close": close, "volume": 1000.0}, index=idx)
        same_bar = clause_signal(Clause("close_return_1d", "<", -0.10), df, None)
        lagged = clause_signal(Clause("close_return_1d", "<", -0.10, within_days=3), df, None)
        assert list(same_bar.astype(int)) == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        assert list(lagged.astype(int)) == [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
        assert not lagged.iloc[:5].any(), "a lagged clause must never be true before its own event"

    def test_the_lag_is_rendered_to_humans_not_silently_applied(self):
        """Without this, "crash then news" and "crash and news same day"
        read identically while testing different hypotheses."""
        spec = ConditionSpec(
            label="x",
            clauses=(Clause("shock_zscore", ">=", 3.0, within_days=3), Clause("rsi_14d", "<", 30.0)),
            direction="long",
        )
        desc = condition_desc(spec)
        assert "last 3 days" in desc
        assert desc.count("last 3 days") == 1, "only the lagged clause carries the window"


class TestDailyNativeIndicators:
    """A real, measured train/serve skew: four indicators are not
    distributionally comparable when recomputed on an hourly frame at
    scale=24, so a condition accepted on `rsi_14d < 30` (289 real daily
    occurrences) could never once fire in the live hourly scan."""

    def test_the_skewed_indicators_are_declared_daily_native(self):
        from llm_pipeline.novel_condition_tester import DAILY_NATIVE_INDICATORS
        for ind in ("rsi_14d", "atr_pct_14d", "daily_range_pct", "efficiency_ratio_20d", "shock_zscore"):
            assert ind in DAILY_NATIVE_INDICATORS

    def test_hourly_evaluation_of_a_daily_native_clause_matches_the_daily_one(self):
        import numpy as np, pandas as pd
        from llm_pipeline.novel_condition_tester import clause_signal, clause_signal_hourly
        rng = np.random.default_rng(0)
        d_idx = pd.date_range("2024-01-01", periods=120, freq="D")
        close = 100 * np.cumprod(1 + rng.normal(0, 0.03, 120))
        daily = pd.DataFrame({"open": close, "high": close * 1.02, "low": close * 0.98,
                              "close": close, "volume": 1000.0}, index=d_idx)
        h_idx = pd.date_range("2024-01-01", periods=120 * 24, freq="h")
        hc = np.repeat(close, 24)
        hourly = pd.DataFrame({"open": hc, "high": hc * 1.02, "low": hc * 0.98,
                               "close": hc, "volume": 1000.0}, index=h_idx)
        clause = Clause("rsi_14d", "<", 45.0)
        daily_days = set(clause_signal(clause, daily, None).pipe(lambda s: s[s]).index.normalize())
        hourly_days = set(clause_signal_hourly(clause, hourly, daily, None).pipe(lambda s: s[s]).index.normalize())
        assert daily_days, "fixture should produce some triggers"
        assert not (daily_days - hourly_days), "the live scan must never miss a day the backtest accepted on"


class TestIncrementalBaseline:
    """Testing `event AND market_state` against an ORDINARY day is close
    to meaningless -- the market state alone already differs from an
    ordinary day, so the event clause could be pure decoration and the
    test would still pass it. The control must be the same condition
    with the event clause removed."""

    def test_reduced_spec_drops_only_the_event_clauses(self):
        from llm_pipeline.novel_condition_tester import reduced_spec
        spec = ConditionSpec(
            label="x",
            clauses=(Clause("shock_zscore", ">=", 3.0), Clause("rsi_14d", "<", 30.0),
                     Clause("volume_zscore_30d", ">", 1.0)),
            direction="long")
        red = reduced_spec(spec)
        assert {c.indicator for c in red.clauses} == {"rsi_14d", "volume_zscore_30d"}

    def test_no_contrast_exists_when_there_is_no_event_clause(self):
        from llm_pipeline.novel_condition_tester import reduced_spec
        spec = ConditionSpec(label="x", clauses=(Clause("rsi_14d", "<", 30.0),), direction="long")
        assert reduced_spec(spec) is None

    def test_no_contrast_exists_when_the_spec_is_only_event_clauses(self):
        """Removing them would leave no condition at all -- and for an
        event-only spec, 'vs. an ordinary day' is the right test anyway."""
        from llm_pipeline.novel_condition_tester import reduced_spec
        spec = ConditionSpec(label="x", clauses=(Clause("shock_zscore", ">=", 3.0),), direction="long")
        assert reduced_spec(spec) is None

    def test_the_two_baselines_answer_different_questions_and_say_which(self):
        import numpy as np, pandas as pd
        from candidates.methodology import MethodologyConfig, pattern_significance
        rng = np.random.default_rng(3)
        n = 2200   # >3 yearly folds, else pattern_significance returns insufficient_data
        idx = pd.date_range("2016-01-01", periods=n, freq="D")
        close = 100 * np.cumprod(1 + rng.normal(0, 0.02, n))
        ohlc = {"A": pd.DataFrame({"open": close, "high": close * 1.02, "low": close * 0.98,
                                   "close": close, "volume": 1000.0}, index=idx)}
        cfg = MethodologyConfig(horizons=(1, 3, 7))
        locs = np.arange(60, n - 40, 7)
        ev = pd.DataFrame({"entry_loc": locs, "group": "A",
                           "period": idx[locs].year, "trigger_time": idx[locs]})
        for h in cfg.horizons:
            ev[f"mfe_{h}"], ev[f"mae_{h}"] = 0.05, -0.03
        ctrl_locs = np.arange(63, n - 40, 7)
        ctrl = pd.DataFrame({"entry_loc": ctrl_locs, "group": "A", "period": idx[ctrl_locs].year})
        uncond = pattern_significance(ev, ohlc, "long", cfg)
        incr = pattern_significance(ev, ohlc, "long", cfg, baseline_events=ctrl)
        assert uncond["baseline_kind"] == "unconditional"
        assert incr["baseline_kind"] == "incremental"
        assert incr["baseline_n"] > 0
        # the two are answering different questions, so they must not be
        # silently identical numbers reported under the same label
        assert uncond["baseline_mean_return"] != incr["baseline_mean_return"]
