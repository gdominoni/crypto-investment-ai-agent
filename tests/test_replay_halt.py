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
