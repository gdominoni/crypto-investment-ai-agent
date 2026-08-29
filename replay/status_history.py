"""Same mechanism as candidates/status_history.py -- how long a candidate
has been tracked without ever being accepted, and the keep/drop decision
that follows after 2 years -- but keyed off the SIMULATED date passed
in, not real wall-clock time (this replay can walk through years of
history in a few real hours, so `datetime.now()` would never show a
candidate as "2 years old"). This module's OTHER milestone (has a
candidate actually validated) is no longer time-based at all -- see
MILESTONE_N below. Entirely isolated from production's own
status_history.json, same as every other file in replay/state/.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HISTORY_PATH = Path(__file__).resolve().parent / "state" / "status_history.json"
PRUNE_YEARS_THRESHOLD = 2.0
RE_ASK_MONTHS = 6


def load_history() -> dict:
    if not HISTORY_PATH.exists():
        return {}
    return json.loads(HISTORY_PATH.read_text())


def _save(history: dict) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, indent=2))


def record_status(candidate: str, status: str, as_of: str) -> None:
    history = load_history()
    entry = history.setdefault(candidate, {"first_tracked_at": as_of, "dropped": False, "last_asked_at": None, "status_log": []})
    entry["status_log"].append({"status": status, "at": as_of})
    entry["status_log"] = entry["status_log"][-320:]  # ~3 years of every-few-days entries, comfortably
    _save(history)


def record_horizon(candidate: str, horizon: int) -> bool:
    """Records the empirically-derived horizon just computed for
    `candidate`. Returns True the FIRST time a real horizon is recorded
    (the transition away from the placeholder default used before enough
    data existed) or any time it later changes -- callers use this to
    decide whether to notify, so the placeholder-to-real transition is
    never silent (see docs/case_study/methodology-decisions.md)."""
    history = load_history()
    entry = history.setdefault(candidate, {"first_tracked_at": None, "dropped": False, "last_asked_at": None, "status_log": []})
    prev = entry.get("last_horizon")
    changed = prev != horizon
    entry["last_horizon"] = horizon
    _save(history)
    return changed


def all_latest_statuses() -> dict[str, dict]:
    """Every candidate ever tracked and its most recent status -- unlike
    replay/state.py's battery_status.json (which only ever holds the
    currently-ACCEPTED subset, since that's all live trading needs),
    this is the full picture: rejected and watch candidates too, so a
    question like "what's been tested, anything accepted or dropped?"
    can actually be answered completely. `status` here is the raw,
    ongoing technical classification (accepted/watch/rejected/
    insufficient_data) -- it is NOT the same claim as "validated":
    `milestone_reported`/`milestone_cleared` (set by
    mark_milestone_reported below) is the one place that records whether
    a candidate actually crossed its first 2 (simulated) years while
    still accepted, which is what this project means by "validated"."""
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


def is_dropped(candidate: str) -> bool:
    return load_history().get(candidate, {}).get("dropped", False)


def drop_candidate(candidate: str) -> None:
    history = load_history()
    if candidate in history:
        history[candidate]["dropped"] = True
        _save(history)


def _years_tracked(entry: dict, as_of: str) -> float:
    first = pd.Timestamp(entry["first_tracked_at"])
    return (pd.Timestamp(as_of) - first).days / 365.25


def years_tracked(candidate: str, as_of: str) -> float | None:
    entry = load_history().get(candidate)
    return _years_tracked(entry, as_of) if entry else None


def candidates_due_for_prune_decision(as_of: str, years_threshold: float = PRUNE_YEARS_THRESHOLD) -> list[str]:
    """Candidates tracked at least `years_threshold` simulated years,
    never accepted in their recorded history, not already asked
    recently -- identical logic to production's own version, just driven
    by the simulated `as_of` instead of the real calendar."""
    due = []
    for name, entry in load_history().items():
        if entry.get("dropped"):
            continue
        if _years_tracked(entry, as_of) < years_threshold:
            continue
        if any(e["status"] == "accepted" for e in entry["status_log"]):
            continue
        last_asked = entry.get("last_asked_at")
        if last_asked and (pd.Timestamp(as_of) - pd.Timestamp(last_asked)).days < RE_ASK_MONTHS * 30:
            continue
        due.append(name)
    return due


def mark_asked(candidate: str, as_of: str) -> None:
    history = load_history()
    if candidate in history:
        history[candidate]["last_asked_at"] = as_of
        _save(history)


MILESTONE_N = 50  # replaces an earlier, purely calendar-based 2-year milestone -- see docs/case_study/methodology-decisions.md


def candidates_due_for_milestone(resolved_live_test_counts: dict[str, int], n_step: int = MILESTONE_N) -> list[str]:
    """NOT a one-time event -- fires again every time a candidate crosses
    a NEW multiple of `n_step` resolved LIVE tests (50, 100, 150, ...),
    regardless of current status, each time re-asking whether the
    candidate is worth continuing to test (see mark_milestone_reported,
    which now also carries the same keep/drop decision
    candidates_due_for_prune_decision uses, reusing the same Telegram
    buttons). Unlike candidates_due_for_prune_decision (which only fires
    for candidates that have NEVER been accepted), this fires for every
    candidate, accepted or not -- even a currently-accepted one is worth
    periodically re-confirming, not assumed permanent. Gated on the same
    sample size the statistics themselves require (MILESTONE_N ==
    classify_status's own min_report_events). "Validated" is earned (or
    re-confirmed, or lost) fresh at EACH checkpoint, based on that
    checkpoint's own accepted/not status -- not a permanent one-time
    badge; see mark_milestone_reported. candidates_due_for_prune_decision
    (unaffected by this change, still purely calendar-based) remains the
    only safety net for a trigger too rare to ever reach even the first
    N-step on its own."""
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
    """`n_reached`: the multiple of MILESTONE_N just crossed (50, 100,
    150, ...) -- recorded so this checkpoint doesn't re-fire until the
    NEXT multiple. `cleared` records whether the candidate was 'accepted'
    at THIS checkpoint specifically -- re-evaluated fresh every time, not
    a permanent one-time badge, since a candidate's status can genuinely
    drift between checkpoints in either direction. This is the one bit
    of state that lets all_latest_statuses() (and anything downstream,
    like the checkpoint digest) say "validated" truthfully as of the
    most recent checkpoint."""
    history = load_history()
    if candidate in history:
        history[candidate]["last_checkpoint_n"] = n_reached
        history[candidate]["milestone_reported"] = True  # at least one checkpoint has fired, ever
        history[candidate]["milestone_cleared"] = cleared  # as of the MOST RECENT checkpoint only
        _save(history)


SHUTDOWN_FLAG_PATH = Path(__file__).resolve().parent / "state" / "SHUTDOWN"


def is_shut_down() -> bool:
    return SHUTDOWN_FLAG_PATH.exists()


def trigger_shutdown(reason: str, as_of: str) -> None:
    SHUTDOWN_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SHUTDOWN_FLAG_PATH.write_text(f"{as_of}: {reason}\n")
