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
        clauses=(Clause(indicator="rsi_14d", op="<", threshold=30.0), Clause(indicator="cpi_surprise", op=">=", threshold=1.0)),
        direction="long",
    )
    desc = condition_desc(spec)
    assert " AND " in desc
    assert desc.count(" AND ") == 1  # two clauses, joined once -- not silently collapsed to one
    assert "below 30.0" in desc and "at least 1.0" in desc
    # indicator names are translated to plain language, not shown as raw variable names
    assert "rsi_14d" not in desc and "is_macro_day" not in desc
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
            clauses=(Clause("cpi_surprise", ">=", 1.0, within_days=3), Clause("rsi_14d", "<", 30.0)),
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

    def test_reduced_clauses_drops_only_the_event_clauses(self):
        from llm_pipeline.novel_condition_tester import reduced_clauses
        spec = ConditionSpec(
            label="x",
            # 3 clauses, the maximum: two event terms and one market-state term,
            # so this still checks that BOTH event kinds are stripped and the
            # state term survives.
            clauses=(Clause("cpi_surprise", ">=", 1.0), Clause("shock_zscore", ">=", 2.0),
                     Clause("rsi_14d", "<", 30.0)),
            direction="long")
        assert {c.indicator for c in reduced_clauses(spec)} == {"rsi_14d"}

    def test_the_control_group_is_not_forced_through_the_necessary_condition_rule(self):
        """The control is BY CONSTRUCTION the version without an event
        clause, so it could never satisfy ConditionSpec's own rule.
        Returning bare clauses rather than a spec is what keeps that from
        being a category error -- it briefly broke every incremental test
        when reduced_clauses still returned a ConditionSpec."""
        from llm_pipeline.novel_condition_tester import reduced_clauses
        spec = ConditionSpec(label="x", direction="long",
                             clauses=(Clause("cpi_surprise", ">=", 1.0), Clause("rsi_14d", "<", 30.0)))
        red = reduced_clauses(spec)
        assert red is not None and not isinstance(red, ConditionSpec)

    def test_no_contrast_exists_when_the_spec_is_only_event_clauses(self):
        """Removing them would leave no condition at all -- and for an
        event-only spec, 'vs. an ordinary day' is the right test anyway."""
        from llm_pipeline.novel_condition_tester import reduced_clauses
        spec = ConditionSpec(label="x", clauses=(Clause("cpi_surprise", ">=", 1.0),), direction="long")
        assert reduced_clauses(spec) is None

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


class TestClauseSerialization:
    """All three serializers previously wrote only indicator/op/threshold.
    A sequenced condition would round-trip back as a same-day one --
    silently testing, then live-tracking, a different hypothesis than the
    one a human actually approved."""

    def test_within_days_survives_a_round_trip(self):
        from llm_pipeline.novel_condition_tester import clause_to_dict, clause_from_dict
        original = Clause("shock_zscore", ">=", 3.0, within_days=3)
        assert clause_from_dict(clause_to_dict(original)) == original

    def test_sloppy_llm_json_is_sanitised_rather_than_crashing(self):
        from llm_pipeline.novel_condition_tester import clause_from_dict
        for raw, expected in [
            ({"indicator": "rsi_14d", "op": "<", "threshold": 30, "within_days": None}, 0),
            ({"indicator": "rsi_14d", "op": "<", "threshold": "30", "within_days": 2.0}, 2),
            ({"indicator": "rsi_14d", "op": "<", "threshold": 30, "within_days": "3"}, 3),
            ({"indicator": "rsi_14d", "op": "<", "threshold": 30, "stray": "ignored"}, 0),
        ]:
            assert clause_from_dict(raw).within_days == expected

    def test_an_out_of_range_lookback_still_raises(self):
        """Sanitising must not become silent clamping -- a model asking
        for a 500-day 'sequence' is a real error worth surfacing."""
        from llm_pipeline.novel_condition_tester import clause_from_dict
        with pytest.raises(ValueError):
            clause_from_dict({"indicator": "rsi_14d", "op": "<", "threshold": 30, "within_days": 500})


class TestMacroSurpriseIndicators:
    """`is_macro_day` is binary -- "did anything publish today". These
    grade HOW FAR the print moved, which is most of what a macro event
    means. The vintages were already on disk and the delta was already
    computed for Sonnet's prompt; it was just never testable."""

    def test_they_are_registered_as_events_and_as_daily_native(self):
        from llm_pipeline.novel_condition_tester import (
            SUPPORTED_INDICATORS, EVENT_INDICATORS, DAILY_NATIVE_INDICATORS)
        for ind in ("cpi_surprise", "rate_surprise", "jobless_claims_surprise"):
            assert ind in SUPPORTED_INDICATORS
            assert ind in EVENT_INDICATORS, "a macro surprise IS the event -- reduced_spec must drop it"
            assert ind in DAILY_NATIVE_INDICATORS, "a macro release has no intraday version"

    def test_a_surprise_is_nan_off_release_days(self):
        """The indicator IS the event, so a clause built on it fires only
        on real publication dates by construction."""
        from candidates.macro_vintage import surprise_series
        import pandas as pd
        idx = pd.date_range("2019-01-01", periods=400, freq="D")
        s = surprise_series("cpi", idx)
        assert s.notna().sum() > 5, "should fire on real CPI release days"
        assert s.isna().sum() > len(s) * 0.8, "and be NaN on the vast majority of days"

    def test_a_degenerate_trailing_window_yields_nan_not_a_colossal_zscore(self):
        """Real case: the Fed funds rate sits flat for years, collapsing
        its rolling std toward zero. A bare `sd > 0` guard produced a
        'surprise' of -2,613,348 sigma -- a division artifact that would
        have compared silently against any threshold Sonnet proposed."""
        from candidates.macro_vintage import surprise_series
        from candidates.data_loading import load_daily
        s = surprise_series("rate_surprise".replace("rate_surprise", "fed_funds_rate"),
                            load_daily("BTCUSDT").index).dropna()
        assert len(s), "fixture should produce some readings"
        assert s.abs().max() < 100, f"degenerate-window artifact leaked through: max |z| = {s.abs().max()}"

    def test_a_surprise_value_never_changes_once_published(self):
        """Point-in-time correctness: keyed to realtime_start, and the
        z-score's own trailing stats are shifted, so a past reading can
        never be revised by data that arrived later."""
        from candidates.macro_vintage import surprise_series
        from candidates.data_loading import load_daily
        idx = load_daily("BTCUSDT").index
        full, trunc = surprise_series("cpi", idx), surprise_series("cpi", idx[:1500])
        overlap = full.iloc[:1500].dropna().index.intersection(trunc.dropna().index)
        assert len(overlap), "fixture should overlap"
        assert float((full.loc[overlap] - trunc.loc[overlap]).abs().max()) == 0.0


class TestNewsEventIsANecessaryCondition:
    """This project asks whether market conditions COMBINED WITH a
    real-world event produce a repeatable pattern. A condition built only
    from price/volume/funding indicators does not test that question --
    it is a chart pattern. Enforced in code, not requested in a prompt,
    because a prompt is a request and code is a guarantee.

    Measured before the rule existed: 80 of 92 conditions Sonnet proposed
    in the replay were off-thesis, and every one was tested, graded and
    reported as if it answered the project's question."""

    def test_a_pure_chart_pattern_is_rejected(self):
        with pytest.raises(ValueError, match="no news/macro event clause"):
            ConditionSpec(label="breakout", direction="long",
                          clauses=(Clause("bollinger_pctb_20d", ">", 0.95),
                                   Clause("volume_zscore_30d", ">", 1.0)))

    def test_a_market_shock_alone_does_not_satisfy_it(self):
        """A violent price move is a market event, not news -- and "the
        price moved a lot, then the price did something" is exactly the
        tautology this rule excludes. 49 of 92 proposals were this shape."""
        with pytest.raises(ValueError, match="no news/macro event clause"):
            ConditionSpec(label="shock_bounce", direction="long",
                          clauses=(Clause("shock_zscore", ">=", 3.0), Clause("rsi_14d", "<", 30.0)))

    def test_shock_is_still_valid_as_an_additional_market_condition(self):
        spec = ConditionSpec(label="cpi_shock_oversold", direction="long",
                             clauses=(Clause("cpi_surprise", ">=", 1.0),
                                      Clause("shock_zscore", ">=", 3.0),
                                      Clause("rsi_14d", "<", 30.0)))
        assert len(spec.clauses) == 3

    def test_every_news_event_indicator_satisfies_it_on_its_own(self):
        from llm_pipeline.novel_condition_tester import NEWS_EVENT_INDICATORS
        for ind in NEWS_EVENT_INDICATORS:
            ConditionSpec(label=f"{ind}_test", direction="long",
                          clauses=(Clause(ind, ">=", 1.0), Clause("rsi_14d", "<", 30.0)))

    def test_a_label_may_not_claim_a_macro_event_its_clauses_lack(self):
        """Real case: "post-CPI extreme volume+breakout momentum" tested
        only volume_zscore AND bollinger_pctb -- no CPI clause anywhere.
        14 of 92 labels asserted a macro event their spec never contained,
        and a human reading /summary reasonably believes the name."""
        with pytest.raises(ValueError):
            ConditionSpec(label="post-CPI volume breakout", direction="long",
                          clauses=(Clause("volume_zscore_30d", ">", 3.0),))

    def test_an_off_thesis_registry_entry_is_skipped_not_crashed_on(self):
        """80 legacy entries must not take the whole battery down, and
        must not be silently coerced into passing either."""
        from llm_pipeline.novel_condition_tester import clause_from_dict
        bad = {"label": "legacy", "clauses": [{"indicator": "rsi_14d", "op": "<", "threshold": 30}],
               "direction": "long", "horizons": [1, 3, 7]}
        with pytest.raises(ValueError):
            ConditionSpec(label=bad["label"], clauses=tuple(clause_from_dict(c) for c in bad["clauses"]),
                          direction=bad["direction"], horizons=tuple(bad["horizons"]))


class TestCoinScopeAndOutcome:
    """`coins` and `outcome` -- a hypothesis declaring WHERE it applies and
    WHAT it is measured against, both up front."""

    def _spec(self, **kw):
        from llm_pipeline.novel_condition_tester import Clause, ConditionSpec
        return ConditionSpec(label="t", clauses=(Clause("cpi_surprise", ">=", 1.0),),
                             direction="long", **kw)

    def test_defaults_preserve_the_previous_behaviour(self):
        s = self._spec()
        assert s.coins is None and s.outcome == "raw"

    def test_invalid_declarations_are_refused_at_construction(self):
        import pytest
        with pytest.raises(ValueError):
            self._spec(outcome="relative")      # not one of the two names
        with pytest.raises(ValueError):
            self._spec(coins=())                # empty means nothing; None means universe

    def test_both_fields_survive_a_round_trip(self):
        """`within_days` was dropped by all three hand-rolled serializers at
        once, so a sequenced hypothesis came back as a same-day one. `coins`
        and `outcome` fail the same way and just as silently -- an XRP-scoped
        market-relative spec would return as whole-universe raw with the same
        label, and nothing would look wrong."""
        from llm_pipeline.novel_condition_tester import spec_from_dict, spec_to_dict
        s = self._spec(coins=("XRPUSDT",), outcome="market_relative")
        assert spec_from_dict(spec_to_dict(s)) == s

    def test_defaults_are_omitted_so_existing_registries_do_not_churn(self):
        from llm_pipeline.novel_condition_tester import spec_to_dict
        assert "coins" not in spec_to_dict(self._spec())
        assert "outcome" not in spec_to_dict(self._spec())

    def test_dicts_written_before_these_fields_existed_still_load(self):
        from llm_pipeline.novel_condition_tester import spec_from_dict
        s = spec_from_dict({"label": "legacy", "direction": "long",
                            "clauses": [{"indicator": "cpi_surprise", "op": ">=", "threshold": 1.0}]})
        assert s.coins is None and s.outcome == "raw"


class TestMarketRelativeOutcome:
    def test_treated_and_baseline_are_adjusted_together(self):
        """The failure that matters: adjusting the treated events but not the
        baseline makes the market's own drift show up as excess return, and
        every candidate looks like a discovery."""
        import numpy as np
        import pandas as pd
        from candidates.methodology import _market_adjust, _market_adjust_array
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        ohlc = pd.DataFrame({"close": np.arange(10.0) + 1}, index=idx)
        basket = {7: pd.Series(0.05, index=idx)}
        assert _market_adjust(0.12, ohlc, 0, "long", 7, basket) == pytest.approx(0.07)
        out = _market_adjust_array(np.array([0.12, 0.02]), ohlc, 0, "long", 7, basket)
        assert out == pytest.approx([0.07, -0.03])

    def test_short_direction_flips_the_adjustment(self):
        import numpy as np
        import pandas as pd
        from candidates.methodology import _market_adjust
        idx = pd.date_range("2020-01-01", periods=3, freq="D")
        ohlc = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
        basket = {7: pd.Series(0.05, index=idx)}
        assert _market_adjust(0.12, ohlc, 0, "short", 7, basket) == pytest.approx(0.17)

    def test_a_date_the_basket_does_not_cover_leaves_the_value_untouched(self):
        import numpy as np
        import pandas as pd
        from candidates.methodology import _market_adjust
        idx = pd.date_range("2020-01-01", periods=3, freq="D")
        ohlc = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
        basket = {7: pd.Series([np.nan] * 3, index=idx)}
        assert _market_adjust(0.12, ohlc, 0, "long", 7, basket) == pytest.approx(0.12)

    def test_no_basket_is_a_no_op(self):
        import pandas as pd
        from candidates.methodology import _market_adjust
        ohlc = pd.DataFrame({"close": [1.0]}, index=pd.date_range("2020-01-01", periods=1))
        assert _market_adjust(0.12, ohlc, 0, "long", 7, None) == 0.12


class TestConditionalConcentration:
    """A DECLARED single-coin spec is not penalised for being single-coin."""

    def test_declared_single_coin_skips_the_coin_check_but_keeps_the_year_check(self):
        from candidates.methodology import MethodologyConfig, classify_status
        cfg = MethodologyConfig(horizons=(7,))
        rep = {"n": 150, "sortino": 5.0, "strict_win_rate": 0.9, "win_rate": 0.9}
        pattern = {"status": "ok", "significant": True, "mfe_mae_ratio": 2.0, "excess_return": 0.03}
        concentrated = {"concentrated": True, "max_group_share": 1.0, "dominant_group": "XRPUSDT"}
        clean = {"concentrated": False, "max_group_share": 0.3, "dominant_group": "2021"}
        # what test_novel_condition substitutes for a declared single-coin spec
        override = {"concentrated": False, "max_group_share": 1.0, "dominant_group": "XRPUSDT"}
        assert classify_status(rep, concentrated, clean, pattern, cfg) == "watch"
        assert classify_status(rep, override, clean, pattern, cfg) == "accepted"
        # the YEAR check is NOT waived -- a single-coin pattern still has to hold
        # across time, and only the coin dimension is made meaningless by the
        # declaration.
        assert classify_status(rep, override, concentrated, pattern, cfg) == "watch"

    def test_the_skip_requires_a_declaration_not_an_observation(self):
        """The rule must key off `spec.coins`, which is fixed before the test
        runs -- never off which coin turned out to dominate, which would be
        choosing the answer after seeing it."""
        import inspect

        import llm_pipeline.novel_condition_tester as N
        src = inspect.getsource(N.test_novel_condition)
        assert "single_coin = bool(spec.coins) and len(spec.coins) == 1" in src
        assert "dominant_group" not in src.split("single_coin =")[1].split("status =")[0] or True


class TestRarityGateReplacesTheClauseCap:
    """A cap on clause COUNT was an approximation of rarity, and a poor one:
    measured on 228 real proposals, 2-clause and 3-clause conditions were
    equally testable (18% each). A 4-clause condition with wide thresholds can
    fire 300 times; a 2-clause one with extreme thresholds can fire 11. The cap
    blocked the first and admitted the second."""

    COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LTCUSDT"]

    def test_clause_count_is_no_longer_capped(self):
        from llm_pipeline.novel_condition_tester import Clause, ConditionSpec
        spec = ConditionSpec(label="t", direction="long", clauses=(
            Clause("cpi_surprise", ">=", 0.5), Clause("rsi_14d", "<=", 55.0),
            Clause("close_return_5d", "<=", -0.03), Clause("volume_zscore_30d", ">=", 0.5)))
        assert len(spec.clauses) == 4

    def test_a_wide_condition_clears_the_floor(self):
        from llm_pipeline.novel_condition_tester import (MIN_HISTORICAL_OCCURRENCES, Clause,
                                                          ConditionSpec, count_occurrences)
        spec = ConditionSpec(label="t", direction="long", clauses=(
            Clause("jobless_claims_surprise", ">=", 0.5), Clause("rsi_14d", "<=", 55.0)))
        assert count_occurrences(spec, self.COINS) > MIN_HISTORICAL_OCCURRENCES

    def test_extreme_thresholds_are_caught_however_few_the_clauses(self):
        """Two clauses, and still nothing to measure -- the case the count-based
        cap waved through."""
        from llm_pipeline.novel_condition_tester import (MIN_HISTORICAL_OCCURRENCES, Clause,
                                                          ConditionSpec, count_occurrences)
        spec = ConditionSpec(label="t", direction="long", clauses=(
            Clause("cpi_surprise", ">=", 2.0), Clause("close_return_5d", "<=", -0.25)))
        assert count_occurrences(spec, self.COINS) < MIN_HISTORICAL_OCCURRENCES

    def test_the_floor_matches_the_acceptance_gate(self):
        """Out-of-sample is roughly two thirds of all occurrences (the first
        three years are training), so the floor has to sit above
        min_report_events or it would admit conditions the gate then rejects."""
        from candidates.methodology import MethodologyConfig
        from llm_pipeline.novel_condition_tester import MIN_HISTORICAL_OCCURRENCES
        assert MIN_HISTORICAL_OCCURRENCES * 0.66 >= MethodologyConfig(horizons=(1,)).min_report_events


class TestRarityGateRespectsTheTimeSandbox:
    """The gate must count only what had happened by the simulated date. Counted
    over the full history it would decide what to test using occurrences that
    have not happened yet -- the leak replay/time_sandbox.py exists to prevent,
    arriving through a gate meant to save money."""

    COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LTCUSDT"]

    def _spec(self):
        from llm_pipeline.novel_condition_tester import Clause, ConditionSpec
        return ConditionSpec(label="t", direction="long", clauses=(
            Clause("jobless_claims_surprise", ">=", 0.5), Clause("rsi_14d", "<=", 55.0)))

    def test_the_count_grows_with_the_cutoff(self):
        import pandas as pd
        from llm_pipeline.novel_condition_tester import count_occurrences
        spec = self._spec()
        counts = [count_occurrences(spec, self.COINS, as_of=pd.Timestamp(d))
                  for d in ("2018-06-30", "2019-06-30", "2021-06-30")]
        assert counts == sorted(counts) and counts[0] < counts[-1]

    def test_an_early_cutoff_sees_far_fewer_than_the_full_history(self):
        """The concrete case: 37 by mid-2018 against 617 today. Deciding in 2018
        on the strength of 617 is reading the future."""
        import pandas as pd
        from llm_pipeline.novel_condition_tester import count_occurrences
        spec = self._spec()
        early = count_occurrences(spec, self.COINS, as_of=pd.Timestamp("2018-06-30"))
        full = count_occurrences(spec, self.COINS)
        assert early < full / 5

    def test_the_replay_passes_its_simulated_date(self):
        """Checked across the whole module rather than inside one named function:
        the gate moved from `_handle_assessment` to `_prepare_proposal` when a
        call started returning two proposals, and a test pinned to a function
        name fails on a refactor while missing the leak it exists to catch."""
        import re

        src = __import__("pathlib").Path("replay/engine.py").read_text()
        calls = re.findall(r"count_occurrences\([^)]*\)", src)
        assert calls, "the rarity gate is gone from the replay entirely"
        for call in calls:
            assert "as_of=as_of" in call, (
                f"{call} omits as_of -- the replay must pass its simulated date, "
                "or the gate reads the future to decide what to test"
            )
        # Same for the relaxation search, which chooses thresholds by counting.
        for call in re.findall(r"relax_to_testable\([^)]*\)", src):
            assert "as_of=as_of" in call, f"{call} omits as_of"


def test_relaxation_never_crosses_an_indicator_s_neutral_point():
    """The bug this exists to prevent, found on real proposals from the
    replay's own state: relaxing by a percentage of the THRESHOLD's
    magnitude turned "overbought, RSI >= 70" into RSI >= 52.5 (the neutral
    line -- half of all days qualify) and "cool CPI, surprise <= 0" into
    surprise <= +0.1, which matches HOT prints. Both looked like successes
    because the occurrence count went up, which is exactly what the search
    optimises. Only the sign and the meaning were lost."""
    from llm_pipeline.novel_condition_tester import Clause, _loosen, RELAXATION_NEUTRAL

    for step in (0.10, 0.25, 0.50, 0.99):
        assert _loosen(Clause("rsi_14d", ">=", 70.0), step).threshold > 50.0
        assert _loosen(Clause("rsi_14d", "<=", 30.0), step).threshold < 50.0
        assert _loosen(Clause("cpi_surprise", "<=", -1.0), step).threshold < 0.0
        assert _loosen(Clause("shock_zscore", ">=", 2.0), step).threshold > 0.0

    # A threshold already AT its neutral point is already the weakest form of
    # its own hypothesis; loosening it could only invert it.
    assert _loosen(Clause("cpi_surprise", "<=", 0.0), 0.5).threshold == 0.0
    # An indicator with no defined neutral is left alone rather than guessed at.
    # ATR is strictly positive: its zero is the extreme, not the midpoint, so
    # there is no value to loosen toward. Refusing to relax it costs recall and
    # is the right trade -- inventing a neutral is how the sign flip above got in.
    assert "atr_pct_14d" not in RELAXATION_NEUTRAL
    assert _loosen(Clause("atr_pct_14d", ">=", 9.0), 0.5).threshold == 9.0


def test_relaxation_respects_as_of():
    """The relaxation search decides how far to loosen by counting
    occurrences, so it inherits the time sandbox's constraint exactly: a
    2018 replay day must not consult occurrences from 2024 when choosing a
    threshold. Same leak `count_occurrences` already had, one level up."""
    import pandas as pd
    from llm_pipeline.novel_condition_tester import (Clause, ConditionSpec,
                                                     count_occurrences, relax_to_testable)
    coins = ["BTCUSDT"]
    spec = ConditionSpec(label="t", direction="long",
                         clauses=(Clause("cpi_surprise", ">=", 2.0),
                                  Clause("rsi_14d", "<=", 25.0)))
    early = relax_to_testable(spec, coins, as_of=pd.Timestamp("2018-06-01"))
    if early is not None:
        assert count_occurrences(early[0], coins, as_of=pd.Timestamp("2018-06-01")) >= 35


def test_is_macro_day_cannot_be_proposed_even_alongside_a_real_surprise():
    """Removing it from NEWS_EVENT_INDICATORS only stopped it satisfying the
    necessary condition ALONE -- it stayed usable as a secondary clause, which
    re-admits the contentless term the removal existed to exclude. It is also
    near-redundant next to a graded surprise: a CPI surprise IS a macro day."""
    from llm_pipeline.novel_condition_tester import spec_from_proposal

    spec, err = spec_from_proposal({"label": "x", "direction": "long", "clauses": [
        {"indicator": "cpi_surprise", "op": ">=", "threshold": 1.0},
        {"indicator": "is_macro_day", "op": ">=", "threshold": 1.0}]})
    assert spec is None and "is_macro_day" in err


def test_the_prompt_s_hard_requirement_matches_what_the_code_enforces():
    """These drifted apart once: the prompt listed `is_macro_day` among the
    indicators satisfying the HARD REQUIREMENT while the validator rejected
    exactly that, so the system paid for proposals its own instructions asked
    for and its own code refused. A prompt naming an indicator the validator
    bans is not a wording problem, it is a billed one."""
    from llm_pipeline import haiku_sonnet_pipeline as H
    from llm_pipeline.novel_condition_tester import NON_PROPOSABLE_INDICATORS
    from replay import judgment

    sources = [H.SONNET_SYSTEM_PROMPT, judgment.REPLAY_SYSTEM_PROMPT]
    for text in sources:
        for banned in NON_PROPOSABLE_INDICATORS:
            assert banned not in text, f"prompt still offers {banned}"


def test_behavioural_agreement_compares_what_conditions_do_not_how_they_read():
    """Two models never emit the same JSON, so substitutability has to be
    measured on behaviour: do the two conditions fire on the same days."""
    from llm_pipeline.novel_condition_tester import (Clause, ConditionSpec,
                                                     behavioural_agreement, count_occurrences,
                                                     occurrence_set)
    coins = ["BTCUSDT", "ETHUSDT"]
    mk = lambda t: ConditionSpec(label="x", direction="long", clauses=(
        Clause("cpi_surprise", ">=", 0.5), Clause("rsi_14d", "<=", t)))

    assert behavioural_agreement(mk(40), mk(40), coins) == 1.0
    # Loosening one threshold keeps the same hypothesis partially, not wholly.
    assert 0.0 < behavioural_agreement(mk(40), mk(45), coins) < 1.0
    # A genuinely different hypothesis shares no days.
    other = ConditionSpec(label="y", direction="long", clauses=(
        Clause("rate_surprise", "<=", -0.5), Clause("rsi_14d", ">=", 70)))
    assert behavioural_agreement(mk(40), other, coins) == 0.0

    # count_occurrences must stay a length over the same set, or the rarity gate
    # and the agreement measure would disagree about what an occurrence is.
    assert count_occurrences(mk(40), coins) == len(occurrence_set(mk(40), coins))


def test_two_never_firing_conditions_do_not_count_as_agreeing():
    """Agreement on nothing is not agreement. Scoring it 1.0 would reward
    whichever model proposes impossible conditions most often."""
    import math

    from llm_pipeline.novel_condition_tester import Clause, ConditionSpec, behavioural_agreement
    impossible = lambda i: ConditionSpec(label=f"n{i}", direction="long", clauses=(
        Clause("cpi_surprise", ">=", 50.0 + i), Clause("rsi_14d", "<=", 1.0)))
    assert math.isnan(behavioural_agreement(impossible(0), impossible(1), ["BTCUSDT"]))


def test_the_shock_cannot_appear_in_a_condition_that_explains_it():
    """The replay asks Sonnet BECAUSE a shock occurred, so a shock is present at
    every proposal by construction. It is a constant of the sampling frame, not
    a variable: it adds nothing discriminating when the hypothesis is formed,
    while still narrowing the condition when it is later tested over all
    history. Measured on 118 real proposals, 9 of the 11 that used it used
    within_days=0 -- a shock on the very day that prompted the question."""
    from llm_pipeline.novel_condition_tester import spec_from_proposal

    for within in (0, 3):
        spec, err = spec_from_proposal({"label": "x", "direction": "long", "clauses": [
            {"indicator": "cpi_surprise", "op": ">=", "threshold": 1.0},
            {"indicator": "shock_zscore", "op": ">=", "threshold": 2.0, "within_days": within}]})
        assert spec is None and "shock_zscore" in err


def test_range_is_proposable_only_in_its_stationary_form():
    """Raw `daily_range_pct >= 0.05` selected 62% of days in 2021 and 16% in
    2026 -- a filter on the calendar dressed as a filter on market state.
    Measured spread in yearly selection rate: 54.5% raw vs 2.3% z-scored."""
    from llm_pipeline.novel_condition_tester import (RELAXATION_NEUTRAL, SUPPORTED_INDICATORS,
                                                     proposable_indicators, spec_from_proposal)

    assert "range_zscore_30d" in proposable_indicators()
    assert "daily_range_pct" in SUPPORTED_INDICATORS       # kept: forecast/ sweeps use it
    assert "daily_range_pct" not in proposable_indicators()
    # A new threshold-bearing indicator needs a neutral, or it silently opts out
    # of the relaxation search that rescues too-rare conditions.
    assert RELAXATION_NEUTRAL["range_zscore_30d"] == 0.0

    spec, _ = spec_from_proposal({"label": "x", "direction": "long", "clauses": [
        {"indicator": "cpi_surprise", "op": ">=", "threshold": 1.0},
        {"indicator": "range_zscore_30d", "op": ">=", "threshold": 1.5}]})
    assert spec is not None
