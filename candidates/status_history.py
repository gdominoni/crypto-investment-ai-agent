"""Tracks how long every candidate -- static (definitions.py) or dynamic
(llm_pipeline/dynamic_candidates.py) -- has been tracked and its weekly
status history, so run_battery.py can recognize "this has never been
accepted in N years" and ask the human whether to keep testing it or
drop it, instead of re-testing forever with no off-ramp. This project
reserves the word "validated" for a candidate that has lived through 50
real, resolved live tests while still 'accepted' at that point -- see
MILESTONE_N/candidates_due_for_milestone below (mirrors
replay/status_history.py's own version, same N=50 threshold, live data
instead of simulated). Committed to git, like the other state files: a
durable record, not ephemeral state.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from candidates.atomic_json import write_json

HISTORY_PATH = Path(__file__).resolve().parent / "status_history.json"
PRUNE_YEARS_THRESHOLD = 2.0
RE_ASK_MONTHS = 6  # don't re-propose "keep or drop?" every single week once asked


def load_history() -> dict:
    if not HISTORY_PATH.exists():
        return {}
    return json.loads(HISTORY_PATH.read_text())


def _save(history: dict) -> None:
    write_json(HISTORY_PATH, history)


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
    NEVER reached 'accepted' in their whole recorded history, and
    haven't already been asked about recently."""
    now = datetime.now(timezone.utc)
    due = []
    for name, entry in load_history().items():
        if entry.get("dropped"):
            continue
        if _years_tracked(entry) < years_threshold:
            continue
        if any(e["status"] == "accepted" for e in entry["status_log"]):
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


def record_horizon(candidate: str, horizon: int) -> bool:
    """Records the empirically-derived horizon just computed for
    `candidate`. Returns True the FIRST time a real horizon is recorded
    (the transition away from the placeholder used before enough data
    existed) or any time it later changes -- callers use this to decide
    whether to notify. Mirrors replay/status_history.py's own version."""
    history = load_history()
    now = datetime.now(timezone.utc).isoformat()
    entry = history.setdefault(candidate, {"first_tracked_at": now, "dropped": False, "last_asked_at": None, "status_log": []})
    prev = entry.get("last_horizon")
    changed = prev != horizon
    entry["last_horizon"] = horizon
    _save(history)
    return changed


MILESTONE_N = 50  # same threshold classify_status itself requires -- see docs/case_study/methodology-decisions.md


def candidates_due_for_milestone(resolved_live_test_counts: dict[str, int], n_step: int = MILESTONE_N) -> list[str]:
    """NOT one-time -- fires again every time a candidate crosses a NEW
    multiple of `n_step` resolved LIVE tests (50, 100, 150, ...),
    regardless of current status. This is the ONE moment this project
    calls a candidate "validated" in production, re-earned (or lost)
    fresh at each checkpoint -- mirrors replay/status_history.py's own
    version exactly, driven by real resolved live tests instead of
    simulated ones."""
    due = []
    for name, entry in load_history().items():
        count = resolved_live_test_counts.get(name, 0)
        current_multiple = (count // n_step) * n_step
        if current_multiple < n_step:
            continue
        if entry.get("last_checkpoint_n", 0) >= current_multiple:
            continue
        due.append(name)
    return due


def mark_milestone_reported(candidate: str, n_reached: int, cleared: bool) -> None:
    history = load_history()
    if candidate in history:
        history[candidate]["last_checkpoint_n"] = n_reached
        history[candidate]["milestone_reported"] = True
        history[candidate]["milestone_cleared"] = cleared
        _save(history)


def all_latest_statuses() -> dict[str, dict]:
    """Every candidate ever tracked and its most recent status, plus
    milestone info -- mirrors replay/status_history.py's own version,
    used the same way (e.g. by a checkpoint/status report)."""
    result = {}
    for name, entry in load_history().items():
        if not entry["status_log"]:
            continue
        result[name] = {
            "status": entry["status_log"][-1]["status"], "dropped": entry.get("dropped", False),
            "milestone_reported": entry.get("milestone_reported", False),
            "milestone_cleared": entry.get("milestone_cleared"),
            "last_checkpoint_n": entry.get("last_checkpoint_n"),
        }
    return result


SHUTDOWN_FLAG_PATH = Path(__file__).resolve().parent / "SHUTDOWN"


def is_shut_down() -> bool:
    return SHUTDOWN_FLAG_PATH.exists()


def trigger_shutdown(reason: str) -> None:
    SHUTDOWN_FLAG_PATH.write_text(f"{datetime.now(timezone.utc).isoformat()}: {reason}\n")
