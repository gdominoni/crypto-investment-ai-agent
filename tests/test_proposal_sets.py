"""Two proposals per call, capped at two clauses each, checked for redundancy."""
import pytest

from llm_pipeline.novel_condition_tester import (MAX_PROPOSABLE_CLAUSES, MAX_PROPOSAL_OVERLAP,
                                                  MIN_HISTORICAL_OCCURRENCES, Clause, ConditionSpec,
                                                  filter_redundant_proposals,
                                                  proposals_from_assessment, spec_from_proposal)

COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LTCUSDT"]
RATE = ("rate_surprise", "<=", -1.0, 7)


def _mk(label, *clauses):
    return ConditionSpec(label=label, direction="long",
                         clauses=tuple(Clause(*c) for c in clauses))


class TestClauseCap:
    def test_three_clauses_are_refused_with_the_reason(self):
        """Each added clause divides historical occurrences by roughly eight; a
        three-clause condition's median is 12 against a floor of 120."""
        c = lambda i, o, t: {"indicator": i, "op": o, "threshold": t}
        spec, err = spec_from_proposal({"label": "x", "direction": "long", "clauses": [
            c("rate_surprise", "<=", -1.0), c("funding_zscore_30d", "<=", -1.5),
            c("rsi_14d", "<=", 30)]})
        assert spec is None and "at most 2" in err and "two separate" in err

    def test_two_clauses_pass(self):
        c = lambda i, o, t: {"indicator": i, "op": o, "threshold": t}
        spec, _ = spec_from_proposal({"label": "x", "direction": "long", "clauses": [
            c("rate_surprise", "<=", -1.0), c("funding_zscore_30d", "<=", -1.5)]})
        assert spec is not None

    def test_the_cap_is_enforced_at_proposal_time_not_in_ConditionSpec(self):
        """The relaxation search and the co-occurrence appendix both construct
        specs internally, and the committed sweeps in forecast/ contain wider
        ones. The cap governs what may be PROPOSED, not what can be represented."""
        assert len(_mk("internal", RATE, ("rsi_14d", "<=", 30),
                        ("funding_zscore_30d", "<=", -1.5)).clauses) == 3


class TestRedundancyCheck:
    def test_the_intended_split_survives_intact(self):
        """The whole point of the design: two proposals SHARING their news term
        and differing in the market term. A text- or clause-based check would
        reject exactly this. Measured, their behavioural overlap is 0.000."""
        a = _mk("funding", RATE, ("funding_zscore_30d", "<=", -1.5))
        b = _mk("oversold", RATE, ("rsi_14d", "<=", 30))
        kept, notes = filter_redundant_proposals([a, b], COINS)
        assert [s.label for s in kept] == ["funding", "oversold"]
        assert notes == []

    def test_a_near_duplicate_is_dropped_with_its_reason(self):
        a = _mk("oversold", RATE, ("rsi_14d", "<=", 30))
        b = _mk("oversold_nudged", RATE, ("rsi_14d", "<=", 31.5))
        kept, notes = filter_redundant_proposals([a, b], COINS)
        assert [s.label for s in kept] == ["oversold"]
        assert len(notes) == 1 and "same days" in notes[0]

    def test_the_first_survives_not_the_better_looking_one(self):
        """Choosing between redundant proposals on any measured quality would be
        selecting a hypothesis on its outcome at proposal time."""
        import inspect
        src = inspect.getsource(filter_redundant_proposals)
        for forbidden in ("p_value", "excess_return", "significant", "sortino"):
            assert forbidden not in src

    def test_the_threshold_sits_in_a_measured_gap(self):
        """0.6 is not a round number picked by hand: over 120 random pairs of
        plausible proposals the maximum overlap was 0.390, while a near-duplicate
        scores 0.837."""
        assert 0.39 < MAX_PROPOSAL_OVERLAP < 0.83


class TestSchema:
    def test_both_shapes_are_accepted(self):
        """Stored replay state and pending-test queues carry the singular form."""
        assert len(proposals_from_assessment({
            "recommended_action": "propose_novel_test",
            "novel_condition_specs": [{"a": 1}, {"b": 2}]})) == 2
        assert len(proposals_from_assessment({
            "recommended_action": "propose_novel_test",
            "novel_condition_spec": {"a": 1}})) == 1
        assert proposals_from_assessment({"recommended_action": "no_action"}) == []

    def test_no_more_than_two_are_taken_however_many_are_returned(self):
        assert len(proposals_from_assessment({
            "recommended_action": "propose_novel_test",
            "novel_condition_specs": [{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}]})) == 2


class TestPromptsMatchTheCode:
    def test_every_prompt_states_the_cap_and_asks_for_a_set(self):
        from llm_pipeline.haiku_sonnet_pipeline import (COMPRESSION_SYSTEM_PROMPT,
                                                         SONNET_SYSTEM_PROMPT)
        from replay.judgment import REPLAY_SYSTEM_PROMPT

        for prompt in (REPLAY_SYSTEM_PROMPT, COMPRESSION_SYSTEM_PROMPT, SONNET_SYSTEM_PROMPT):
            assert "novel_condition_specs" in prompt
            assert "AT MOST TWO CLAUSES" in prompt
            assert "LOOKBACK ON THE NEWS TERM" in prompt
            # The floor is interpolated, never restated by hand -- it changed
            # from 35 to 120 and a hardcoded number would now be a lie.
            assert str(MIN_HISTORICAL_OCCURRENCES) in prompt
