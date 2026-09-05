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


class TestOutboxIsDrainedNotJustFilled:
    """The outbox exists so a Telegram ban cannot stall the replay. It is only
    a safety net if something empties it: `drain_outbox` runs from inside
    `_send`, so a run that ENDS with messages queued has no send left to push
    them out, and the queue becomes a silent hole in the evidence rather than a
    delay. Found by checking a real 12.5-hour ban before restarting a run."""

    def test_the_orchestrator_flushes_on_every_exit_path(self):
        import inspect

        import replay.orchestrator as O
        src = inspect.getsource(O._run_to_completion_locked)
        reached_end = src[src.index('result.get("reached_end")'):]
        assert "_finish(" in reached_end, "a completed run leaves the outbox unflushed"
        assert "_finish(" in src[src.index("safety cap"):], "the cap path leaves it unflushed"
        assert "drain_outbox()" in src, "nothing drains opportunistically during the run"

    def test_flush_reports_what_it_could_not_deliver(self, monkeypatch, tmp_path):
        """An undelivered message must be stated, not swallowed -- otherwise an
        incomplete record is indistinguishable from a complete one."""
        import telegram.bot as B
        monkeypatch.setattr(B, "_OUTBOX_PATH", tmp_path / "outbox.json")
        B._outbox_save([{"payload": {"text": "x"}, "queued_at": 0}])
        monkeypatch.setattr(B, "drain_outbox", lambda limit=3: 0)
        assert B.flush_outbox(max_wait_s=0.0) == 1

    def test_a_long_ban_queues_instead_of_sleeping(self):
        """A retry_after over the threshold must not be waited out inline: that
        made every later message cost ~20 minutes and be lost anyway."""
        import inspect

        import telegram.bot as B
        src = inspect.getsource(B._send)
        assert "_MAX_INLINE_WAIT" in src
        assert B._MAX_INLINE_WAIT <= 120


class TestCompressionReportShowsWhatWasProposed:
    """The Post-Squeeze State line reports the indicators SONNET reasoned from,
    not a fixed pair. A hardcoded RSI/Bollinger line would print two numbers
    unrelated to the hypothesis printed underneath it whenever Sonnet reached
    for funding, volume or efficiency instead -- which is most of the time."""

    def _episode(self):
        import pandas as pd
        from candidates.data_loading import load_daily
        from replay.engine import _compression_exit
        ep = _compression_exit(load_daily("DOGEUSDT"), pd.Timestamp("2019-11-26"))
        assert ep is not None, "the fixture episode no longer exists in the data"
        return ep

    def _spec(self, *indicators):
        return {"label": "x", "direction": "long",
                "clauses": [{"indicator": i, "op": "<=", "threshold": 1.0} for i in indicators]}

    def _line(self, specs):
        from replay.judgment import format_compression_report
        out = format_compression_report("DOGEUSDT", self._episode(), specs)
        return next((l for l in out.splitlines() if "Post-Squeeze" in l), "")

    def test_it_reports_the_indicator_the_proposal_uses(self):
        assert "RSI" in self._line([self._spec("rsi_14d")])
        assert "volume" in self._line([self._spec("volume_zscore_30d")]).lower()

    def test_it_does_not_report_an_indicator_nobody_proposed(self):
        line = self._line([self._spec("volume_zscore_30d")])
        assert "RSI" not in line and "Bollinger" not in line

    def test_a_macro_only_proposal_gets_no_state_line(self):
        """Event indicators are already the MACRO section; repeating them here
        would state the same number twice as though it were two facts."""
        assert self._line([self._spec("jobless_claims_surprise")]) == ""

    def test_oversold_is_claimed_only_from_an_rsi_reading(self):
        """A label is a claim. "Oversold" asserted from a volume z-score would be
        one this project has not earned."""
        assert "Oversold" in self._line([self._spec("rsi_14d")])
        assert "Oversold" not in self._line([self._spec("volume_zscore_30d")])


class TestTheRetiredWordStaysRetired:
    """"Validated" was retired for a documented reason: `required_n_for_power`
    puts a conclusive test in the hundreds of occurrences, so no reachable count
    demonstrates an effect and a checkpoint can only assert persistence.

    It came back twice in one session, in two different places, and both times
    in a message headed for the README. Once in the state summary handed to
    Sonnet -- which then repeated the vocabulary it was given -- and once in
    `/replay_details`, the single most detail-oriented command in the project.
    Neither was caught by the prompt-level ban, because neither went through the
    prompt."""

    def test_the_details_view_says_confirmed(self):
        from candidates.methodology import format_candidate_details
        out = format_candidate_details(
            "x", {"status": "accepted", "n": 900},
            milestone={"milestone_reported": True, "milestone_cleared": True, "last_checkpoint_n": 180},
            milestone_step=20)
        assert "CONFIRMED" in out
        assert "validated" not in out.lower(), out

    def test_the_next_checkpoint_follows_milestone_n_not_a_hardcoded_50(self):
        """It read `n_reached + 50` while MILESTONE_N was 20, announcing the next
        checkpoint 30 occurrences late. The same stale constant this project was
        already bitten by once, in the same place."""
        from candidates.methodology import format_candidate_details
        out = format_candidate_details(
            "x", {"status": "accepted", "n": 900},
            milestone={"milestone_reported": True, "milestone_cleared": True, "last_checkpoint_n": 180},
            milestone_step=20)
        assert "200" in out, out
        assert "230" not in out

    def test_sonnets_state_summary_never_hands_it_the_word(self):
        """The prompt forbids the word, but a model repeats the vocabulary in its
        context -- forbidding it in one place and supplying it in the other just
        puts the two in contradiction."""
        from replay import judgment

        # Checked on the OUTPUT, not the source. An earlier version of this
        # assertion scanned the function text and matched the comment explaining
        # why the word is banned -- a test failing on its own documentation.
        summary = judgment._all_candidates_status_summary()
        assert "validated" not in summary.lower(), summary[:400]
