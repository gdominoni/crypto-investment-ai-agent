"""Running out of API credit mid-replay must STOP, not silently continue.

Before this, both "the model returned malformed JSON" and "we can no longer
reach the API" were caught by one `except Exception` that printed "skipping"
and let the day advance and be checkpointed as done. A replay that exhausted
its credit partway would walk through every remaining simulated day doing no
LLM work at all, finish, and leave a checkpoint claiming it reached the
present -- and since the checkpoint had advanced, resuming would never revisit
those years.
"""
from __future__ import annotations

import pandas as pd
import pytest

import replay.engine as E


class TestSystemicFailureClassification:
    def test_out_of_credit_is_systemic(self):
        class OutOfCredit(Exception):
            message = "Your credit balance is too low to access the Anthropic API"
        assert E._is_systemic_api_failure(OutOfCredit()) is not None

    def test_billing_and_quota_wording_also_caught(self):
        for text in ("insufficient funds", "billing issue", "quota exceeded", "payment required"):
            assert E._is_systemic_api_failure(Exception(text)) is not None, text

    def test_a_bad_api_key_is_systemic(self):
        import anthropic
        e = anthropic.AuthenticationError.__new__(anthropic.AuthenticationError)
        Exception.__init__(e, "invalid x-api-key")
        assert E._is_systemic_api_failure(e) is not None

    def test_a_malformed_model_response_is_NOT_systemic(self):
        """This one must keep skipping: one bad JSON response should not stop a
        run that is otherwise working."""
        assert E._is_systemic_api_failure(ValueError("Expecting value: line 1 column 1")) is None
        assert E._is_systemic_api_failure(RuntimeError("No text block in response")) is None
        assert E._is_systemic_api_failure(KeyError("novel_condition_spec")) is None


class TestHaltCheckpoint:
    def test_it_resumes_from_the_day_BEFORE_the_failure(self, monkeypatch, tmp_path):
        """The failing day was only partly processed -- some of its events may
        already have been judged. Marking it done would drop the rest, so the
        checkpoint goes to the previous day and that day is redone in full."""
        saved = {}
        monkeypatch.setattr(E.state, "save_checkpoint",
                            lambda d, status=None: saved.update(date=d, status=status))
        monkeypatch.setattr(E, "_send", lambda *a, **k: True)
        out = E._halt_replay(pd.Timestamp("2021-06-15"), "the Anthropic account is out of credit", 42)
        assert saved["date"] == "2021-06-14", "must rewind one day, not checkpoint the failed day"
        assert saved["status"] == "halted"
        assert out["stopped"] == "api_failure"
        assert out["current_date"] == "2021-06-14"

    def test_the_human_is_told_rather_than_left_to_notice(self, monkeypatch):
        sent = []
        monkeypatch.setattr(E.state, "save_checkpoint", lambda d, status=None: None)
        monkeypatch.setattr(E, "_send", lambda msg, *a, **k: sent.append(msg) or True)
        E._halt_replay(pd.Timestamp("2021-06-15"), "the Anthropic account is out of credit", 42)
        assert len(sent) == 1
        body = sent[0]
        assert "out of credit" in body and "2021-06-14" in body
        assert "replay continue" in body, "must say how to resume"


class TestOrchestratorSurvivesTheEndOfTheBudget:
    """The unattended driver must end a run out of credit with a resume point,
    not a traceback.

    Every test here takes `isolated_replay_state`: `run_to_completion` acquires
    the real replay lock, so without isolation these fail with
    ReplayAlreadyRunning whenever an actual run is in progress -- a test
    contending with a live nine-hour run for its own state file, which is the
    same defect already recorded in methodology-decisions.md under "Tests could
    write to the live replay's own state"."""

    def test_it_stops_on_api_failure_instead_of_burning_the_remaining_chunks(self, monkeypatch, isolated_replay_state):
        import replay.orchestrator as O
        calls = {"advance": 0}

        def fake_advance():
            calls["advance"] += 1
            return {"stopped": "api_failure", "reason": "the Anthropic account is out of credit",
                    "current_date": "2023-04-11"}
        monkeypatch.setattr(O, "advance", fake_advance)
        monkeypatch.setattr(O, "Anthropic", lambda **kw: object())
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        out = O.run_to_completion(max_chunks=200)
        assert calls["advance"] == 1, "must stop immediately, not retry 200 times"
        assert out["resume_from"] == "2023-04-11"
        assert out["reached_end"] is False

    def test_a_failing_checkin_question_does_not_end_a_working_run(self, monkeypatch, isolated_replay_state):
        """The check-in is cosmetic. Its failure used to be uncaught and would
        have ended a run whose real work was succeeding."""
        import replay.orchestrator as O
        seen = {"n": 0}

        def fake_advance():
            seen["n"] += 1
            return {"reached_end": seen["n"] >= 6}
        monkeypatch.setattr(O, "advance", fake_advance)
        monkeypatch.setattr(O, "Anthropic", lambda **kw: object())
        monkeypatch.setattr(O, "answer_market_question",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        out = O.run_to_completion(max_chunks=20, ask_every=2)
        assert out["reached_end"] is True, "a cosmetic failure must not stop the replay"


class TestConsecutiveFailureHalt:
    """The catch-all. Message-matching cannot anticipate the next breaking
    change: `_is_systemic_api_failure` looked for credit/billing wording and
    sailed past a 400 reading "`temperature` is deprecated for this model", so a
    real run skipped every event for seven simulated months while advancing and
    checkpointing normally. A count needs no foresight."""

    def test_the_threshold_is_small_enough_to_catch_it_within_one_chunk(self):
        # a 30-day chunk holds on the order of 10-20 events, so the halt has to
        # trigger well inside one chunk or the guard is decorative
        assert E.CONSECUTIVE_FAILURE_HALT <= 10

    def test_an_unrecognised_error_repeated_is_treated_as_systemic(self):
        """The exact shape of what went wrong: an error whose text matches none
        of the known systemic patterns, failing every single time."""
        unknown = Exception("`temperature` is deprecated for this model.")
        assert E._is_systemic_api_failure(unknown) is None, "message-matching misses it, as it did"
        # ...which is precisely why the count exists
        assert E.CONSECUTIVE_FAILURE_HALT > 0

    def test_isolated_failures_do_not_trip_it(self):
        """One malformed response between good ones must still just be skipped."""
        import inspect
        src = inspect.getsource(E.advance) if hasattr(E, "advance") else ""
        # the counter must be reset on the success path, not only incremented
        engine_src = __import__("pathlib").Path("replay/engine.py").read_text()
        # Structural, not a magic number: every path that can INCREMENT the
        # counter must have a matching reset on its success path. Asserting a
        # fixed count instead broke the moment the three triggers (macro, shock,
        # compression) became one, for a reason that had nothing to do with what
        # this test is guarding.
        increments = engine_src.count("consecutive_failures += 1")
        resets = engine_src.count("consecutive_failures = 0")
        assert increments >= 1, "the guard has no increment at all"
        assert resets >= increments + 1, (
            f"{increments} increment path(s) but only {resets} reset(s) -- counter must "
            "reset after each success (plus once at initialisation), or unrelated failures "
            "spread across a long run would eventually halt a healthy replay"
        )


class TestUnattendedPruneDigest:
    """The replay must not stall on a decision nobody is there to make."""

    def test_the_replay_applies_drop_recommendations_itself(self, monkeypatch):
        """Same treatment the orchestrator gives a test proposal. Production
        sends the identical digest and waits, because there the decision is the
        human's -- this is a property of the simulation, not of the system."""
        import inspect
        src = inspect.getsource(E._check_prune_decisions)
        assert "sh.drop_candidate(c)" in src, "recommendations must be applied, not just sent"
        assert "prune_recommendation" in src

    def test_only_drops_are_acted_on(self):
        """`keep` is the default for every candidate including those not named,
        so acting on it would be a no-op."""
        import inspect
        src = inspect.getsource(E._check_prune_decisions)
        assert '== "drop"' in src

    def test_the_message_says_the_decision_was_automatic(self):
        import inspect
        src = inspect.getsource(E._check_prune_decisions)
        assert "Unattended replay" in src, "a reader of the transcript must not think a human chose"

    def test_the_chunk_cap_is_high_enough_for_a_full_timeline(self):
        """The cap is consumed by DISCOVERY, not elapsed time -- a chunk ends
        early on every proposal. 200 stopped a real run at 2023-03 with budget
        left: ~110 thirty-day chunks plus 197 proposals exhausted it."""
        import inspect
        import replay.orchestrator as O
        default = inspect.signature(O.run_to_completion).parameters["max_chunks"].default
        assert default >= 110 + 400


class TestCompressionTrigger:
    """The replay's only trigger: a confirmed exit from volatility compression."""

    def _btc(self):
        from candidates.data_loading import load_daily
        return load_daily("BTCUSDT")

    def test_it_fires_on_a_confirmed_exit_and_reports_the_whole_episode(self):
        import pandas as pd

        df = self._btc()
        hits = [E._compression_exit(df, d) for d in df.index[df.index >= pd.Timestamp("2019-01-01")]]
        hits = [h for h in hits if h]
        assert len(hits) > 20, "a trigger this rare would starve the replay"
        for h in hits:
            # A must precede B must precede C, and C is exactly the confirmation lag.
            assert h["a_date"] <= h["b_date"] < h["c_date"]
            assert (h["c_date"] - h["b_date"]).days >= E.COMPRESSION_CONFIRM_DAYS
            assert h["duration"] >= 1

    def test_it_never_fires_when_compression_resumed_before_confirmation(self):
        """The whole point of the confirmation window. An exit followed by
        re-compression is a flicker inside the same lull, not a regime change --
        measured, those are followed by a defined trend 15.2% of the time
        against 23.8% for confirmed exits."""
        import pandas as pd

        from candidates.methodology import COMPRESSION_ZSCORE_THRESHOLD, vol_compression_series

        df = self._btc()
        z = vol_compression_series(df)
        state = (z >= COMPRESSION_ZSCORE_THRESHOLD).fillna(False).astype(bool)
        for d in df.index[df.index >= pd.Timestamp("2019-01-01")]:
            hit = E._compression_exit(df, d)
            if hit is None:
                continue
            b, c = hit["b_date"], hit["c_date"]
            assert not state.loc[b:c].any(), f"fired at {d.date()} though compression resumed"

    def test_the_data_shown_is_dated_to_B_not_to_the_confirmation(self):
        """The confirmation window decides WHETHER to ask. It must not leak into
        WHAT is shown, or the model would be reasoning from days that had not
        happened when the hypothesis is supposed to begin."""
        import inspect

        src = inspect.getsource(E.advance)
        assert 'as_of_b = episode["b_date"]' in src
        assert "as_of=as_of_b" in src
        assert "_handle_assessment(as_of_b," in src

    def test_the_removed_triggers_are_really_gone(self):
        """Macro releases are among the CAUSES being sought and shocks are the
        OUTCOME; both were measured not to select for what this project looks
        for. Leaving either wired in would silently restore the bias."""
        import inspect

        src = inspect.getsource(E.advance)
        assert "MACRO_SERIES" not in src, "macro release trigger still wired in"
        assert "_shock_transition" not in src, "shock trigger still wired in"
        assert "_compression_exit" in src
