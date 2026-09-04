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


def test_the_haiku_headline_path_is_gone_from_production():
    """Removed 2026-09-02. No headline or sentiment term exists in
    `proposable_indicators()`, and every proposal must carry a FRED macro
    surprise, so nothing Haiku surfaced could enter a testable hypothesis --
    measured: of 771 live tests opened for Sonnet-discovered candidates, zero
    were news-linked. The backfill that would have fixed it was measured too
    (forecast/sentiment_power.py): at the feed quality real news sentiment
    achieves, accepted conditions are indistinguishable from the pure-noise
    floor. This guards the deletion the way the shock removal above is
    guarded -- the path is cheap to reintroduce by reflex and the reason it
    is gone is not obvious from the code that remains."""
    import inspect

    from llm_pipeline import haiku_sonnet_pipeline as H
    from llm_pipeline.novel_condition_tester import proposable_indicators

    for gone in ("haiku_scout", "HAIKU_SYSTEM_PROMPT", "HAIKU_MODEL",
                 "sonnet_strategist", "SONNET_SYSTEM_PROMPT", "run_once",
                 "format_sonnet_message", "_asset_to_coin",
                 "_recent_headlines_summary"):
        assert not hasattr(H, gone), f"the removed headline path is back: {gone}"

    # The daemon must not schedule it either.
    import scheduler.live_daemon as D
    assert "run_headline_scan" not in inspect.getsource(D)

    # The premise of the removal: no proposable indicator is news-derived, so a
    # headline could never appear in a clause. If this ever stops being true
    # (a real sentiment backfill), revisit the decision rather than this test.
    assert not {i for i in proposable_indicators()
                if "news" in i or "sentiment" in i or "headline" in i}

    # Sonnet must not be shown headlines it cannot encode -- production used to
    # include them here while the replay never did, a live train/serve gap.
    # Checked on the function BODY, not its source text: the docstring
    # legitimately explains the removal and would match a naive search.
    import ast, textwrap
    fn = ast.parse(textwrap.dedent(inspect.getsource(H.sonnet_compression_response))).body[0]
    body = fn.body[1:] if isinstance(fn.body[0], ast.Expr) and isinstance(
        getattr(fn.body[0], "value", None), ast.Constant) else fn.body
    body_src = "\n".join(ast.unparse(node) for node in body).upper()
    assert "HEADLINE" not in body_src, "the compression prompt is showing headlines again"


def test_run_compression_scan_reaches_telegram_end_to_end(monkeypatch, tmp_path):
    """`run_compression_scan()`'s own `send_telegram` call was undefined for a
    year -- deleted by commit 20b134f, which was rewriting the neighbouring
    shock->compression code and took three functions with it. Every real
    escalation raised NameError on its very last line, AFTER the proposal had
    been queued and `mark_escalated` called, so the episode was permanently
    suppressed and the pending test expired behind buttons nobody saw. The
    per-episode try/except printed it and moved on. No test called this
    function end-to-end, which is exactly how that went unnoticed."""
    import llm_pipeline.haiku_sonnet_pipeline as H
    import llm_pipeline.pending_tests as pending_tests
    import data_ingestion.market_data.binance_fetcher as fetcher

    # `run_compression_scan` builds an Anthropic client before it reaches the
    # stubbed judgment call, so the key must exist as a VALUE even though no
    # request is ever made. Set here rather than relied upon from a local .env:
    # CI has no .env, and a test that passes only on the author's machine is the
    # same blind spot as the 3.11 syntax and the ccxt import before it.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-never-used")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "test-chat")
    monkeypatch.setattr(H, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(H, "Anthropic", lambda **kw: object())

    monkeypatch.setattr(pending_tests, "PENDING_TESTS_PATH", tmp_path / "pending_test.json")
    monkeypatch.setattr(fetcher, "update_all", lambda coins: None)

    episode = {"symbol": "BTCUSDT", "b_date": pd.Timestamp("2024-03-15"),
               "a_date": pd.Timestamp("2024-03-01"), "duration": 14,
               "z_at_a": -1.5, "b_return": 0.03}
    monkeypatch.setattr(C, "current_compression_exit", lambda c: episode if c == "BTCUSDT" else None)
    monkeypatch.setattr(H, "sonnet_compression_response", lambda ep, client: {
        "assessment": "worth testing",
        "recommended_action": "propose_novel_test",
        "novel_condition_specs": [{
            "label": "hot_cpi_into_oversold_btc", "direction": "long",
            "clauses": [{"indicator": "cpi_surprise", "op": ">=", "threshold": 1.0, "within_days": 2},
                        {"indicator": "rsi_14d", "op": "<=", "threshold": 30, "within_days": 0}],
        }],
    })
    sent = {}
    monkeypatch.setattr(H, "send_telegram",
                         lambda message, reply_markup=None: sent.update(message=message, reply_markup=reply_markup) or True)

    H.run_compression_scan(["BTCUSDT"])

    assert "hot_cpi_into_oversold_btc" in sent.get("message", "")
    assert sent.get("reply_markup") is not None
    queue = pending_tests._load_queue()
    assert len(queue) == 1
    assert queue[0]["specs"][0]["label"] == "hot_cpi_into_oversold_btc"
    assert C.already_escalated("BTCUSDT", episode["b_date"])
