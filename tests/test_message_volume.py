"""The replay's Telegram traffic, after the per-live-test stream was removed.

That stream was 95% of all messages (15,500 of 16,363 over a full run) and it
had two costs. Telegram answered the volume with retry_after = 63,364s, and a
real run stalled at 2021-09-23 -- but the rate limit is only the mechanical
objection. The substantive one is that 7,800 notifications are not a history
anybody reads. These tests pin the removal, the bounded digest that replaced
it, and the short ids that make a 45-character trigger name typeable.
"""
from __future__ import annotations

import inspect

import pandas as pd
import pytest


class TestPerLiveTestMessagesAreGone:
    def test_opening_a_live_test_sends_nothing(self):
        import replay.engine as E
        src = inspect.getsource(E._scan_mechanical_triggers)
        assert "_send(" not in src, "the per-open message is back -- 7,800 of them per run"

    def test_resolving_a_live_test_sends_only_the_failure_alert(self):
        """`_check_consecutive_failures` stays: it is a FAST warning for a
        confirmed candidate, and burying it in a monthly digest is exactly what
        would destroy its purpose. Nothing else may notify per resolution."""
        import replay.engine as E
        src = inspect.getsource(E._check_live_tests)
        assert "Live test resolved" not in src
        assert "_check_consecutive_failures" in src


class TestMonthlyDigestIsBounded:
    def test_it_caps_its_rows_rather_than_listing_every_candidate(self):
        """Its predecessor listed every tracked candidate with a description,
        which at the ~105 a real run reaches would exceed Telegram's 4,096-char
        limit -- the failure that once made a /replay_summary section vanish."""
        import replay.engine as E
        src = inspect.getsource(E._send_monthly_digest)
        assert "MAX_DIGEST_ROWS" in src
        assert E.MAX_DIGEST_ROWS <= 12

    def test_rows_are_ordered_by_progress_never_by_outcome(self):
        """Ranking by success rate puts the luckiest small sample on top --
        measured on a real run, that was candidates at n=6 with a 100% hit rate
        whose own backtest status was `rejected`."""
        import replay.engine as E
        src = inspect.getsource(E._send_monthly_digest)
        assert "_effective_milestone_count" in src
        for forbidden in ("sort(key=lambda.*win", "sorted(.*wins", "by_return"):
            assert forbidden not in src

    def test_it_fits_in_one_telegram_message_on_real_state(self, monkeypatch):
        import replay.engine as E
        sent = []
        monkeypatch.setattr(E, "_send", lambda t, **k: sent.append(t) or True)
        E._send_monthly_digest(pd.Timestamp("2021-08-24"), pd.Timestamp("2021-09-23"), 3, False)
        assert sent, "the digest sent nothing"
        assert len(sent[0]) < 4096, f"digest is {len(sent[0])} chars, over Telegram's limit"


class TestShortIds:
    def test_an_id_is_stable_and_short(self):
        from telegram.bot import SHORT_ID_LEN, short_id
        name = "jobless_claims_beat_low_efficiency_fade_short"
        assert len(short_id(name)) == SHORT_ID_LEN
        assert short_id(name) == short_id(name)

    def test_a_trigger_resolves_by_id_or_by_name(self):
        from telegram.bot import resolve_candidate, short_id
        names = ["jobless_claims_beat_volume_surge_long", "c2_short"]
        assert resolve_candidate("c2_short", names)[0] == "c2_short"
        assert resolve_candidate(short_id(names[0]), names)[0] == names[0]
        assert resolve_candidate(short_id(names[0]).upper(), names)[0] == names[0]

    def test_an_unknown_query_resolves_to_nothing_rather_than_a_guess(self):
        from telegram.bot import resolve_candidate
        name, msg = resolve_candidate("zzzz", ["c2_short"])
        assert name is None and msg is None

    def test_a_colliding_id_reports_the_ambiguity_instead_of_picking_one(self, monkeypatch):
        """Answering about the wrong trigger is worse than asking again."""
        import telegram.bot as B
        monkeypatch.setattr(B, "short_id", lambda c: "dead")
        name, msg = B.resolve_candidate("dead", ["alpha_trigger", "beta_trigger"])
        assert name is None
        assert "more than one" in msg and "alpha_trigger" in msg


class TestFinalQuestions:
    def test_they_are_silent_until_the_run_is_nearly_done(self):
        from replay.orchestrator import _final_questions_due
        assert _final_questions_due("2021-09-23", []) is None

    def test_each_is_asked_the_configured_number_of_rounds_alternating(self):
        from replay.orchestrator import (FINAL_QUESTION_ROUNDS, FINAL_QUESTIONS,
                                          _final_questions_due)
        near = str((pd.Timestamp.today().normalize() - pd.Timedelta(days=30)).date())
        asked = []
        while (q := _final_questions_due(near, asked)) is not None:
            asked.append(q)
            assert len(asked) <= 20, "never terminates"
        assert len(asked) == len(FINAL_QUESTIONS) * FINAL_QUESTION_ROUNDS
        for q in FINAL_QUESTIONS:
            assert asked.count(q) == FINAL_QUESTION_ROUNDS
        assert asked[0] != asked[1], "the two must interleave, not fire back to back"

    def test_the_detail_question_asks_for_furthest_along_not_best(self):
        """Asking for the best-performing trigger would select on the outcome in
        the one message the case study puts on display."""
        from replay.orchestrator import FINAL_QUESTIONS
        detail = FINAL_QUESTIONS[1].lower()
        assert "furthest along" in detail
        for forbidden in ("best performing", "most profitable", "highest return"):
            assert forbidden not in detail


class TestAdverseExcursionSign:
    def test_the_worst_point_is_shown_as_a_loss(self):
        """`path_outcome` stores MAE as an absolute magnitude, so printing it
        with a leading '+' made a 14.7% loss read as a gain."""
        from candidates.methodology import format_candidate_details
        out = format_candidate_details(
            "x", {"status": "accepted", "n": 100},
            recent_occurrences=[{"close_date": "2024-01-01", "coin": "BTCUSDT",
                                  "forward_return": -0.1394, "mfe": -0.097, "mae": 0.147}])
        assert "worst -14.7%" in out, out
        assert "worst +14.7%" not in out
