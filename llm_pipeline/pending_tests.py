"""Holds the single most recent novel-condition-test proposal Sonnet has
sent to Telegram, so that when the human later replies "test it" the bot
knows which spec/coin to actually run -- a single-user, single-chat
system only ever has one pending proposal outstanding at a time, so a
one-slot store (not a queue) is the right amount of complexity here, not
a simplification that will need revisiting.
"""
from __future__ import annotations

import json
from pathlib import Path

from llm_pipeline.novel_condition_tester import ConditionSpec

PENDING_TEST_PATH = Path(__file__).resolve().parent / "pending_test.json"


def push_pending_test(spec: ConditionSpec, coins: list[str], live_coin: str | None, signal_class: str) -> None:
    PENDING_TEST_PATH.write_text(json.dumps({
        "spec": {"label": spec.label, "indicator": spec.indicator, "op": spec.op,
                  "threshold": spec.threshold, "direction": spec.direction, "horizons": list(spec.horizons)},
        "coins": coins, "live_coin": live_coin, "signal_class": signal_class,
    }))


def pop_pending_test() -> tuple[ConditionSpec, list[str], str | None, str] | None:
    if not PENDING_TEST_PATH.exists():
        return None
    data = json.loads(PENDING_TEST_PATH.read_text())
    PENDING_TEST_PATH.unlink()
    s = data["spec"]
    spec = ConditionSpec(label=s["label"], indicator=s["indicator"], op=s["op"],
                          threshold=s["threshold"], direction=s["direction"], horizons=tuple(s["horizons"]))
    return spec, data["coins"], data.get("live_coin"), data["signal_class"]
