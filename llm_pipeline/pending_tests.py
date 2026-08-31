"""Holds novel-condition-test proposals Sonnet has sent to Telegram, so
that when the human presses one of the two buttons attached to the
proposal ("Test It" / "Don't Test It") the bot knows which spec/coin to
actually act on.

A FIFO queue, not a single slot: a single run of haiku_sonnet_pipeline.py
can flag more than one proposal in one go (`run_once()` loops over every
escalated headline, and `run_compression_scan()` runs right after it in the
same process) -- a one-slot store would let a later proposal silently
overwrite an earlier one the human hasn't answered yet.

Each entry carries its own short `id`, embedded in that message's own
button `callback_data` (see llm_pipeline.haiku_sonnet_pipeline's
PROPOSAL_KEYBOARD_TEMPLATE) -- pressing a button always resolves that
EXACT proposal, never "whichever one happens to be oldest right now".
This replaced an earlier free-text "reply 'test it'" design that matched
only a few exact phrases and silently did nothing for anything else
(e.g. "no need, just add it") -- a real, observed disconnect between
what a human would naturally type and what the bot actually understood.
Every question this project's bot poses that has a fixed, small set of
valid mechanical answers (test/don't test, keep/drop a candidate, which
KPI breakdown) is answered with buttons for exactly this reason: the
human literally cannot give an answer outside the valid set. Free text
stays reserved for genuinely open-ended questions, routed to Sonnet.

Entries expire (default 48h, longer than a routine trade signal's 24h in
execution/signal_store.py -- testing a condition isn't as time-sensitive
as firing a live trade on it) so a proposal nobody ever answers doesn't
sit in the queue forever.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from llm_pipeline.novel_condition_tester import ConditionSpec, clause_from_dict, clause_to_dict, spec_from_dict

from candidates.atomic_json import write_json

PENDING_TESTS_PATH = Path(__file__).resolve().parent / "pending_test.json"


def _load_queue() -> list[dict]:
    if not PENDING_TESTS_PATH.exists():
        return []
    return json.loads(PENDING_TESTS_PATH.read_text())


def _save_queue(queue: list[dict]) -> None:
    if queue:
        write_json(PENDING_TESTS_PATH, queue, indent=None)
    elif PENDING_TESTS_PATH.exists():
        PENDING_TESTS_PATH.unlink()


def _drop_expired(queue: list[dict]) -> list[dict]:
    now = datetime.now(timezone.utc)
    return [q for q in queue if datetime.fromisoformat(q["expires_at"]) > now]


def push_pending_test(spec: ConditionSpec, coins: list[str], live_coin: str | None, signal_class: str,
                       expires_hours: float = 48.0) -> str:
    """Returns the new entry's id, so the caller can embed it in that
    proposal message's own Test It / Don't Test It buttons."""
    queue = _drop_expired(_load_queue())
    pending_id = uuid.uuid4().hex[:10]
    queue.append({
        "id": pending_id,
        "spec": {"label": spec.label,
                 "clauses": [clause_to_dict(c) for c in spec.clauses],
                 "direction": spec.direction, "horizons": list(spec.horizons)},
        "coins": coins, "live_coin": live_coin, "signal_class": signal_class,
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=expires_hours)).isoformat(),
    })
    _save_queue(queue)
    return pending_id


def _spec_from_entry(data: dict) -> ConditionSpec:
    s = data["spec"]
    return spec_from_dict(s)


def pop_pending_test_by_id(pending_id: str) -> tuple[ConditionSpec, list[str], str | None, str] | None:
    """Removes and returns the one entry matching `pending_id` -- called
    when the human presses "Test It". None if it already expired or was
    already answered (e.g. a double-tap on the same button)."""
    queue = _drop_expired(_load_queue())
    for i, data in enumerate(queue):
        if data["id"] == pending_id:
            data = queue.pop(i)
            _save_queue(queue)
            return _spec_from_entry(data), data["coins"], data.get("live_coin"), data["signal_class"]
    _save_queue(queue)
    return None


def discard_pending_test_by_id(pending_id: str) -> bool:
    """Removes the entry matching `pending_id` WITHOUT testing it --
    called when the human presses "Don't Test It". Returns whether an
    entry was actually found and removed (False means it already expired
    or was already answered)."""
    queue = _drop_expired(_load_queue())
    remaining = [q for q in queue if q["id"] != pending_id]
    found = len(remaining) != len(queue)
    _save_queue(remaining)
    return found
