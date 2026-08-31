"""The live compression trigger, and the deduplication its predecessor lacked."""
import json

import pandas as pd
import pytest

from llm_pipeline import compression_detector as C


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "ESCALATED_PATH", tmp_path / "escalated.json")


def test_the_same_episode_is_escalated_once_not_once_per_hour():
    """The defect this replaces: `scan_for_shocks` tested the volatility STATE,
    so an hourly daemon re-escalated the same multi-day shock every hour. A
    confirmed compression exit stays true for the whole day it is detected, so
    without a ledger the same episode would fire ~24 times."""
    assert not C.already_escalated("BTCUSDT", "2024-03-15")
    C.mark_escalated("BTCUSDT", "2024-03-15")
    assert C.already_escalated("BTCUSDT", "2024-03-15")
    # Keyed on the EXIT date, not the detection date -- the same episode seen
    # again tomorrow must still collapse to one entry.
    assert C.already_escalated("BTCUSDT", pd.Timestamp("2024-03-15 11:00"))
    # A different coin, and a different episode on the same coin, are separate.
    assert not C.already_escalated("ETHUSDT", "2024-03-15")
    assert not C.already_escalated("BTCUSDT", "2024-05-01")


def test_the_scan_skips_what_it_has_already_escalated(monkeypatch):
    episode = {"symbol": "BTCUSDT", "b_date": pd.Timestamp("2024-03-15"),
               "a_date": pd.Timestamp("2024-03-01"), "duration": 14}
    monkeypatch.setattr(C, "current_compression_exit", lambda c: episode if c == "BTCUSDT" else None)
    assert [e["symbol"] for e in C.scan_for_compression_exits(["BTCUSDT", "ETHUSDT"])] == ["BTCUSDT"]
    C.mark_escalated("BTCUSDT", episode["b_date"])
    assert C.scan_for_compression_exits(["BTCUSDT", "ETHUSDT"]) == []


def test_one_coin_s_bad_data_does_not_cost_the_others_their_scan(monkeypatch):
    def flaky(coin):
        if coin == "BTCUSDT":
            raise ValueError("no data on disk")
        return {"symbol": coin, "b_date": pd.Timestamp("2024-03-15")}
    monkeypatch.setattr(C, "current_compression_exit", flaky)
    assert [e["symbol"] for e in C.scan_for_compression_exits(["BTCUSDT", "ETHUSDT"])] == ["ETHUSDT"]


def test_replay_and_production_share_one_episode_definition():
    """A second, drifted copy of "what counts as a compression exit" would mean
    the replay validates one thing and production tracks another -- the same
    reason shock_detector imported shock_zscore_series rather than restating it."""
    from candidates.methodology import compression_exit
    from replay.engine import _compression_exit

    assert _compression_exit is compression_exit
    import inspect
    assert "compression_exit" in inspect.getsource(C.current_compression_exit)


def test_production_prompt_offers_no_banned_indicator():
    from llm_pipeline.haiku_sonnet_pipeline import COMPRESSION_SYSTEM_PROMPT
    from llm_pipeline.novel_condition_tester import (NON_PROPOSABLE_INDICATORS,
                                                     proposable_indicators)

    for banned in NON_PROPOSABLE_INDICATORS:
        assert banned not in COMPRESSION_SYSTEM_PROMPT
    assert all(i in COMPRESSION_SYSTEM_PROMPT for i in proposable_indicators())


def test_the_shock_escalation_path_is_gone_from_production():
    from llm_pipeline import haiku_sonnet_pipeline as H

    assert not hasattr(H, "run_shock_scan")
    assert not hasattr(H, "sonnet_shock_response")
    assert hasattr(H, "run_compression_scan")
    import scheduler.live_daemon as D
    assert "run_compression_scan" in __import__("inspect").getsource(D.run_forever)
