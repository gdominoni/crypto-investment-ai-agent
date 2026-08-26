"""Tracks how long every candidate -- static (definitions.py) or dynamic
(llm_pipeline/dynamic_candidates.py) -- has been tracked and its weekly
status history, so run_battery.py can recognize "this has never
validated in N years" and ask the human whether to keep testing it or
drop it, instead of re-testing forever with no off-ramp. Committed to
git, like the other state files: a durable record, not ephemeral state.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HISTORY_PATH = Path(__file__).resolve().parent / "status_history.json"
PRUNE_YEARS_THRESHOLD = 2.0
RE_ASK_MONTHS = 6  # don't re-propose "keep or drop?" every single week once asked


def load_history() -> dict:
    if not HISTORY_PATH.exists():
        return {}
    return json.loads(HISTORY_PATH.read_text())


def _save(history: dict) -> None:
    HISTORY_PATH.write_text(json.dumps(history, indent=2))


def record_status(candidate: str, status: str) -> None:
    history = load_history()
    now = datetime.now(timezone.utc).isoformat()
    entry = history.setdefault(candidate, {"first_tracked_at": now, "dropped": False, "last_asked_at": None, "status_log": []})
    entry["status_log"].append({"status": status, "at": now})
    entry["status_log"] = entry["status_log"][-104:]  # ~2 years of weekly entries is plenty of history to keep
    _save(history)


def is_dropped(candidate: str) -> bool:
    return load_history().get(candidate, {}).get("dropped", False)


def drop_candidate(candidate: str) -> None:
    history = load_history()
    if candidate in history:
        history[candidate]["dropped"] = True
        _save(history)


def _years_tracked(entry: dict) -> float:
    first = datetime.fromisoformat(entry["first_tracked_at"])
    return (datetime.now(timezone.utc) - first).days / 365.25


def years_tracked(candidate: str) -> float | None:
    """Public accessor -- how long THIS system has been weekly-testing
    the candidate (operational metadata), not the span of market data
    its backtest evidence is pooled from (those are different numbers;
    conflating them is what let Sonnet's prune advice invent a "this
    window spanned choppy conditions" story it had no actual evidence
    for)."""
    entry = load_history().get(candidate)
    return _years_tracked(entry) if entry else None


def candidates_due_for_prune_decision(years_threshold: float = PRUNE_YEARS_THRESHOLD) -> list[str]:
    """Candidates tracked for at least `years_threshold` years that have
    NEVER reached 'validated' in their whole recorded history, and
    haven't already been asked about recently."""
    now = datetime.now(timezone.utc)
    due = []
    for name, entry in load_history().items():
        if entry.get("dropped"):
            continue
        if _years_tracked(entry) < years_threshold:
            continue
        if any(e["status"] == "validated" for e in entry["status_log"]):
            continue
        last_asked = entry.get("last_asked_at")
        if last_asked and (now - datetime.fromisoformat(last_asked)).days < RE_ASK_MONTHS * 30:
            continue
        due.append(name)
    return due


def mark_asked(candidate: str) -> None:
    history = load_history()
    if candidate in history:
        history[candidate]["last_asked_at"] = datetime.now(timezone.utc).isoformat()
        _save(history)


SHUTDOWN_FLAG_PATH = Path(__file__).resolve().parent / "SHUTDOWN"


def is_shut_down() -> bool:
    return SHUTDOWN_FLAG_PATH.exists()


def trigger_shutdown(reason: str) -> None:
    SHUTDOWN_FLAG_PATH.write_text(f"{datetime.now(timezone.utc).isoformat()}: {reason}\n")
