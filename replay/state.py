"""Persistent state for the historical replay -- entirely separate paths
from the real production state (candidates/status_history.json,
execution/live_battery_state.json, execution/tradesv3.sqlite, etc.), so
running the replay can never corrupt or get confused with the real
system's own state. Everything here lives under replay/state/.
"""
from __future__ import annotations

import json
from pathlib import Path

from candidates.atomic_json import write_json

STATE_DIR = Path(__file__).resolve().parent / "state"
CHECKPOINT_PATH = STATE_DIR / "checkpoint.json"
BATTERY_STATUS_PATH = STATE_DIR / "battery_status.json"
TRADE_LOG_PATH = STATE_DIR / "trade_log.json"
PENDING_TEST_PATH = STATE_DIR / "pending_test.json"
DYNAMIC_CANDIDATES_PATH = STATE_DIR / "dynamic_candidates.json"
PENDING_REVEALS_PATH = STATE_DIR / "pending_reveals.json"
HORIZONS_PATH = STATE_DIR / "horizons.json"


def _read(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _write(path: Path, data) -> None:
    """Atomic -- the replay checkpoint is rewritten after EVERY simulated
    day precisely so a crash can resume; a truncated one would invert that
    guarantee into "resume is now impossible". See candidates/atomic_json."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(path, data, default=str)


def load_checkpoint() -> dict:
    """`current_date`: the simulated 'today', the replay's whole reason
    for existing -- everything the engine does is gated on never looking
    past it. `status`: 'running' or 'waiting_for_human' (a novel-condition
    proposal is pending a real Telegram reply)."""
    return _read(CHECKPOINT_PATH, {"current_date": None, "status": "running"})


def save_checkpoint(current_date: str, status: str = "running") -> None:
    _write(CHECKPOINT_PATH, {"current_date": current_date, "status": status})


def load_battery_status() -> dict:
    return _read(BATTERY_STATUS_PATH, {"candidates": {}})


def save_battery_status(status: dict) -> None:
    _write(BATTERY_STATUS_PATH, status)


def load_trade_log() -> list[dict]:
    return _read(TRADE_LOG_PATH, [])


def append_trade(entry: dict) -> int:
    """Records a trade at OPEN time only -- `status: "open"`, no outcome
    yet, matching how this would actually unfold live: the outcome isn't
    known until real (simulated) time passes and a barrier is actually
    hit. Returns the trade's id, used by `update_trade` to record its
    resolution later, on whatever day that actually happens."""
    log = load_trade_log()
    trade_id = len(log)
    log.append({"id": trade_id, "status": "open", **entry})
    _write(TRADE_LOG_PATH, log)
    return trade_id


def update_trade(trade_id: int, updates: dict) -> None:
    log = load_trade_log()
    for entry in log:
        if entry["id"] == trade_id:
            entry.update(updates)
            break
    _write(TRADE_LOG_PATH, log)


def load_open_trades() -> list[dict]:
    return [t for t in load_trade_log() if t["status"] == "open"]


def load_dynamic_candidates() -> dict:
    return _read(DYNAMIC_CANDIDATES_PATH, {})


def save_dynamic_candidates(registry: dict) -> None:
    _write(DYNAMIC_CANDIDATES_PATH, registry)


def load_pending_test() -> dict | None:
    return _read(PENDING_TEST_PATH, None)


def save_pending_test(entry: dict | None) -> None:
    if entry is None and PENDING_TEST_PATH.exists():
        PENDING_TEST_PATH.unlink()
    elif entry is not None:
        _write(PENDING_TEST_PATH, entry)


def queue_reveal(message: str, reveal_date: str) -> None:
    """A fully-computed result (already known, computed instantly against
    existing history) held back from delivery until `reveal_date` --
    checked day-by-day during the walk (`due_reveals`), same as an open
    trade's own resolution, so it surfaces INTERLEAVED with whatever else
    happens around that day rather than dumped all at once the moment
    simulated time next moves at all."""
    queue = _read(PENDING_REVEALS_PATH, [])
    queue.append({"message": message, "reveal_date": reveal_date})
    _write(PENDING_REVEALS_PATH, queue)


def load_horizons() -> dict:
    """Every candidate's most recent EMPIRICALLY-derived horizon (from
    pattern_significance, whenever it had enough data to compute one --
    independent of accepted/watch/rejected status). A candidate absent
    here has never had enough data yet; `_open_live_test` falls back to
    the documented placeholder in that case -- see
    docs/case_study/methodology-decisions.md."""
    return _read(HORIZONS_PATH, {})


def save_horizons(horizons: dict) -> None:
    _write(HORIZONS_PATH, horizons)


def due_reveals(as_of) -> list[str]:
    """Pops and returns every queued reveal whose `reveal_date` has
    arrived (<= as_of) -- called once per simulated day in the main walk."""
    queue = _read(PENDING_REVEALS_PATH, [])
    due = [q for q in queue if q["reveal_date"] <= str(as_of.date())]
    remaining = [q for q in queue if q["reveal_date"] > str(as_of.date())]
    if remaining:
        _write(PENDING_REVEALS_PATH, remaining)
    elif PENDING_REVEALS_PATH.exists():
        PENDING_REVEALS_PATH.unlink()
    return [q["message"] for q in due]


def load_parked_proposals() -> list[dict]:
    """Proposals that were on-thesis and well-formed but did not yet have enough
    history behind them to be tested, waiting for the data to catch up.

    They exist because discarding them was throwing away most of the early
    replay. Measured against the current grammar, only 8% of conditions have
    enough occurrences to be testable as of January 2019, rising through 50% by
    2022 to 78% by 2025 -- so a replay walking from 2018 would reject the large
    majority of everything it discovered in its first four years, permanently,
    since a rejected proposal was never stored anywhere.

    Parking them costs no API calls: they are re-checked by the weekly battery
    refresh that already runs.

    The delay also makes the eventual test STRONGER, not weaker. A condition
    formulated in 2019 and tested in 2022 is tested partly on data that did not
    exist when the hypothesis was written down -- which is the cleanest form of
    out-of-sample evidence this system can produce, and the reason
    `prospective_split` reports the two halves separately."""
    return _read(STATE_DIR / "parked_proposals.json", [])


def park_proposal(entry: dict) -> None:
    """Idempotent on label: a condition re-proposed later must not queue twice."""
    parked = load_parked_proposals()
    if any(p["spec"].get("label") == entry["spec"].get("label") for p in parked):
        return
    parked.append(entry)
    _write(STATE_DIR / "parked_proposals.json", parked)


def unpark_proposal(label: str) -> None:
    parked = [p for p in load_parked_proposals() if p["spec"].get("label") != label]
    _write(STATE_DIR / "parked_proposals.json", parked)


def load_confirmation_priors() -> dict:
    """Per candidate: how many occurrences already postdated its hypothesis at the
    moment it was registered. Nonzero only for a proposal that sat parked -- see
    `_effective_milestone_count`."""
    return _read(STATE_DIR / "confirmation_priors.json", {})


def save_confirmation_prior(candidate: str, n: int) -> None:
    priors = load_confirmation_priors()
    priors[candidate] = int(n)
    _write(STATE_DIR / "confirmation_priors.json", priors)


class ReplayAlreadyRunning(RuntimeError):
    """Raised when a second `replay.orchestrator` tries to start while one is
    already advancing this same state."""


LOCK_PATH = STATE_DIR / "replay.lock"


def acquire_replay_lock() -> int:
    """Claims exclusive ownership of this replay's state, or raises
    `ReplayAlreadyRunning`. Returns the PID written to the lock, for logging.

    WHY THIS EXISTS -- a real incident, not a hypothetical. Two orchestrator
    processes ran against the same state overnight with no lock between them.
    Neither checkpoint's writer knew about the other, so `advance()`'s own
    invariant (the checkpoint date only ever moves forward) held true FOR EACH
    PROCESS INDIVIDUALLY but not for the file both were writing: whichever
    process finished a chunk last simply overwrote the other's more recent
    progress. The result was a checkpoint that visibly jumped backward between
    log lines, ~300 near-duplicate proposals for the same handful of days (each
    process discovering its own independent, non-deterministic Sonnet
    proposals for the same episodes), and a trade log containing entries dated
    AFTER the final checkpoint -- 218 of them, up to 8 days ahead, found only
    by checking that invariant directly. Cost was inflated by roughly the same
    factor as the duplication.

    STALE LOCKS ARE RECLAIMED, not treated as permanent. A lock file naming a
    PID that is no longer running (checked via `os.kill(pid, 0)`, which raises
    without signalling anything if the process is gone) means the previous run
    crashed or was killed without cleaning up -- reclaiming it is exactly right,
    since honouring a dead process's lock forever would need a human to notice
    and delete a file by hand every time.
    """
    import os

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        try:
            holder = int(LOCK_PATH.read_text().strip())
        except (ValueError, OSError):
            holder = None
        if holder is not None:
            try:
                os.kill(holder, 0)
                alive = True
            except ProcessLookupError:
                alive = False
            except PermissionError:
                # Exists and is owned by someone else -- treat as alive rather
                # than guess.
                alive = True
            if alive:
                raise ReplayAlreadyRunning(
                    f"Another replay orchestrator is already running (PID {holder}). "
                    f"Two processes writing the same state at once is what corrupted "
                    f"a run before -- see this function's docstring. If that PID is "
                    f"genuinely gone, delete {LOCK_PATH} and retry.")
    pid = os.getpid()
    LOCK_PATH.write_text(str(pid))
    return pid


def release_replay_lock() -> None:
    """Best-effort: a failure to remove the lock file must not mask whatever
    exception the caller is already unwinding from."""
    try:
        LOCK_PATH.unlink(missing_ok=True)
    except OSError:
        pass
