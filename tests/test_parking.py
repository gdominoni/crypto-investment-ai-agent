"""Parked proposals, and the prospective split that makes the wait worth having."""
import pandas as pd
import pytest

COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LTCUSDT"]


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    from replay import state
    monkeypatch.setattr(state, "STATE_DIR", tmp_path)


class TestParking:
    def _entry(self, label, when="2019-03-01"):
        return {"spec": {"label": label, "direction": "long", "clauses": [
                    {"indicator": "cpi_surprise", "op": ">=", "threshold": 1.5, "within_days": 3},
                    {"indicator": "rsi_14d", "op": "<=", "threshold": 25}]},
                "proposed_at": when, "reason": "too rare"}

    def test_a_proposal_too_rare_today_is_kept_not_discarded(self):
        """Only 8% of the grammar is testable as of January 2019. Discarding
        these threw away most of what a replay's first four years discovers, and
        threw it away permanently -- nothing stored a rejected proposal."""
        from replay import state
        state.park_proposal(self._entry("a"))
        assert [p["spec"]["label"] for p in state.load_parked_proposals()] == ["a"]

    def test_parking_is_idempotent_on_label(self):
        from replay import state
        state.park_proposal(self._entry("a"))
        state.park_proposal(self._entry("a", when="2020-06-01"))
        assert len(state.load_parked_proposals()) == 1

    def test_promotion_takes_the_oldest_never_the_best(self):
        """Choosing which parked hypothesis to promote by any measured quality
        would be selecting on the outcome at proposal time."""
        import inspect

        import replay.engine as E
        src = inspect.getsource(E._check_parked_proposals)
        assert 'sorted(parked, key=lambda e: e.get("proposed_at", "")' in src
        for forbidden in ("p_value", "excess_return", "significant", "accepted"):
            assert forbidden not in src

    def test_a_proposal_the_grammar_no_longer_accepts_is_dropped_not_stuck(self):
        """Otherwise one stale entry blocks the queue for the rest of the run."""
        from replay import state
        import replay.engine as E
        bad = self._entry("stale")
        bad["spec"]["clauses"] = [{"indicator": "is_macro_day", "op": ">=", "threshold": 1.0}]
        state.park_proposal(bad)
        assert E._check_parked_proposals(pd.Timestamp("2024-01-01")) is None
        assert state.load_parked_proposals() == []


class TestProspectiveSplit:
    def test_it_separates_evidence_that_postdates_the_hypothesis(self):
        """The distinction the codebase's own vocabulary blurs: walk-forward
        holds out a FOLD, which is out-of-sample with respect to the parameters
        but not to the idea -- every one of those rows existed when the
        hypothesis was written. Occurrences after the proposal date did not."""
        from candidates.methodology import prospective_split
        from llm_pipeline.novel_condition_tester import Clause, ConditionSpec

        spec = ConditionSpec(label="x", direction="long", clauses=(
            Clause("cpi_surprise", ">=", 0.5, within_days=7), Clause("rsi_14d", "<=", 40)))
        early = prospective_split(spec, COINS, "2020-01-01")
        late = prospective_split(spec, COINS, "2024-01-01")
        assert early["n_after"] > late["n_after"], "a later hypothesis has less future left"
        assert early["n_before"] < late["n_before"]
        assert early["n_before"] + early["n_after"] == late["n_before"] + late["n_after"]

    def test_the_baseline_is_the_same_span_not_zero(self):
        """Comparing a post-formulation mean against zero would credit a bull
        market to the condition."""
        import inspect

        from candidates.methodology import prospective_split
        src = inspect.getsource(prospective_split)
        assert "span = fwd.loc[proposed_at:]" in src
        assert "excess_after" in src

    def test_it_reports_and_never_gates(self):
        """At these counts a significance test would usually be underpowered,
        and a test that cannot detect anything must not read as a negative."""
        import inspect

        from candidates.methodology import prospective_split
        src = inspect.getsource(prospective_split)
        assert "p_value" not in src and "significant" not in src


class TestConfirmationCounting:
    def test_an_occurrence_counts_when_it_postdates_the_hypothesis(self):
        """The rule that replaced the indiscriminate backtest top-up. A condition
        with 120 historical occurrences used to reach its checkpoint on day one
        with zero prospective evidence -- but an occurrence from 2019 cannot
        confirm a hypothesis written in 2023, however uncontaminated the model."""
        from replay.engine import _effective_milestone_count

        # Parked three years: those occurrences postdate the hypothesis, so they count.
        assert _effective_milestone_count("dynamic_parked", prior_confirmations=34, live_n=6) == 40
        # Proposed and testable at once: nothing postdates it yet.
        assert _effective_milestone_count("dynamic_fresh", prior_confirmations=0, live_n=6) == 6
        # Static candidates were mined from this history, so none of it postdates them.
        assert _effective_milestone_count("c1_long", prior_confirmations=999, live_n=6) == 6

    def test_the_full_backtest_count_is_no_longer_a_shortcut(self):
        import inspect

        from replay.engine import _effective_milestone_count
        src = inspect.getsource(_effective_milestone_count)
        assert "backtest_n" not in src
        assert "prior_confirmations" in src


class TestConfirmationNotValidation:
    def test_the_word_validated_is_gone_from_the_live_path(self):
        """At these horizons a conclusive test needs occurrences in the hundreds
        (307 at 7 days for a 5% effect). No reachable count demonstrates that, so
        the checkpoint asserts persistence and must not say otherwise."""
        src = __import__("pathlib").Path("replay/engine.py").read_text()
        assert "VALIDATED" not in src
        assert "CONFIRMED" in src

    def test_every_resolved_test_states_what_would_be_required(self):
        """Showing the achieved count AS A FRACTION of the required one is what
        stops an accumulating counter from implying a proof: "23 of 307" cannot
        be misread the way a bare "23" can."""
        import inspect

        from replay.engine import _confirmation_block
        src = inspect.getsource(_confirmation_block)
        assert "required_n_for_power" in src
        assert "{n} of {need:.0f}" in src
        assert "hyperopt_runner.format_result" in src

    def test_the_market_adjusted_rate_lives_on_the_checkpoint(self):
        """Kept out of the per-occurrence message, which stays short, but not
        dropped: a long-only rule is right most of the time in a rising market
        for reasons unrelated to any macro release."""
        import inspect

        from replay.engine import _check_n50_milestones
        src = inspect.getsource(_check_n50_milestones)
        assert "baseline_return" in src and "holding the whole coin universe" in src

    def test_the_confirmation_block_never_raises(self):
        """It runs inside the notification loop: a missing statistic must degrade
        to a stated gap, never take down the message carrying the actual result."""
        from replay.engine import _confirmation_block
        out = _confirmation_block("candidate_that_does_not_exist")
        assert "Confirmation record" in out
        assert "pending hyperopt" in out


class TestTelegramRateLimit:
    """Found by actually running the replay, which no unit test would have."""

    def test_a_429_is_retried_not_swallowed(self):
        """The first unattended run lost NINE messages in eight chunks: `_send`
        printed the 429, set all_ok = False and continued. That loss is invisible
        -- 44 of 47 call sites ignore the return value -- and the Telegram record
        is this project's evidence."""
        import inspect

        import telegram.bot as B
        src = inspect.getsource(B._send)
        assert "retry_after" in src, "Telegram states how long to wait; honour it"
        assert "_MAX_429_RETRIES" in src
        assert "_throttle()" in src

    def test_sends_are_spaced_before_the_limit_is_hit(self):
        """Cheaper than any retry: a 429 costs `retry_after` seconds, observed at
        174, against the ~1s wait that avoids it."""
        import time

        import telegram.bot as B
        B._last_send_at = 0.0
        start = time.monotonic()
        B._throttle()
        B._throttle()
        elapsed = time.monotonic() - start
        assert elapsed >= B._MIN_SEND_INTERVAL * 0.9

    def test_a_network_failure_is_retried_not_abandoned(self):
        """Eight transient DNS/timeout failures appeared in twenty minutes of a
        dry run. `_send` used to `return False` on the first one, losing the
        message -- and over a multi-hour unattended run these are a certainty."""
        import inspect

        import telegram.bot as B
        src = inspect.getsource(B._send)
        assert "retrying" in src
        assert src.count("return False") <= 2, "a network blip must not abandon on first failure"

    def test_the_bot_token_never_reaches_a_log(self):
        """requests embeds the full request URL in its exception text, and the
        URL carries the token -- so a routine network error was writing a live
        credential into a log file in plaintext."""
        import inspect

        import telegram.bot as B
        assert B._redact("hit /bot99:SECRET/sendMessage", "99:SECRET") == "hit /bot<BOT_TOKEN>/sendMessage"
        assert B._redact("no token here", "") == "no token here"
        src = inspect.getsource(B._send)
        assert src.count("_redact(") >= 2, "both the network and the HTTP-error paths must redact"
