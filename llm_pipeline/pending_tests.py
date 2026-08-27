"""Holds novel-condition-test proposals Sonnet has sent to Telegram, so
that when the human later replies "test it" the bot knows which
spec/coin to actually run.

A FIFO queue, not a single slot: a single run of haiku_sonnet_pipeline.py
can flag more than one proposal in one go (`run_once()` loops over every
escalated headline, and `run_shock_scan()` runs right after it in the
same process) -- a one-slot store would let a later proposal silently
overwrite an earlier one the human hasn't replied to yet, so "test it"
would resolve to the wrong spec with no indication anything was lost.
"test it" always resolves the OLDEST pending proposal -- the one that's
been sitting unanswered in the chat the longest, matching what a human
replying "test it" to the first thing they see would actually expect.

Entries expire (default 48h, longer than a routine trade signal's 24h in
execution/signal_store.py -- testing a condition isn't as time-sensitive
as firing a live trade on it) so a proposal nobody ever answers doesn't
sit in the queue forever and get resolved to a "test it" reply weeks
later, testing a condition against market context that's long gone.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from llm_pipeline.novel_condition_tester import ConditionSpec

PENDING_TESTS_PATH = Path(__file__).resolve().parent / "pending_test.json"


def _load_queue() -> list[dict]:
    if not PENDING_TESTS_PATH.exists():
        return []
    return json.loads(PENDING_TESTS_PATH.read_text())


def _save_queue(queue: list[dict]) -> None:
    if queue:
        PENDING_TESTS_PATH.write_text(json.dumps(queue))
    elif PENDING_TESTS_PATH.exists():
        PENDING_TESTS_PATH.unlink()


def _drop_expired(queue: list[dict]) -> list[dict]:
    now = datetime.now(timezone.utc)
    return [q for q in queue if datetime.fromisoformat(q["expires_at"]) > now]


def push_pending_test(spec: ConditionSpec, coins: list[str], live_coin: str | None, signal_class: str,
                       expires_hours: float = 48.0) -> None:
    queue = _drop_expired(_load_queue())
    queue.append({
        "spec": {"label": spec.label, "indicator": spec.indicator, "op": spec.op,
                  "threshold": spec.threshold, "direction": spec.direction, "horizons": list(spec.horizons)},
        "coins": coins, "live_coin": live_coin, "signal_class": signal_class,
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=expires_hours)).isoformat(),
    })
    _save_queue(queue)


def pop_pending_test() -> tuple[ConditionSpec, list[str], str | None, str] | None:
    queue = _drop_expired(_load_queue())
    if not queue:
        _save_queue(queue)
        return None
    data = queue.pop(0)
    _save_queue(queue)
    s = data["spec"]
    spec = ConditionSpec(label=s["label"], indicator=s["indicator"], op=s["op"],
                          threshold=s["threshold"], direction=s["direction"], horizons=tuple(s["horizons"]))
    return spec, data["coins"], data.get("live_coin"), data["signal_class"]
