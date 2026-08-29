"""Persistent state for production's own live tests -- this project
never opens a funded position (see docs/case_study/methodology-decisions.md
and the case-study README: this is a pattern-discovery investigation,
not an investment strategy), so "live" here means the same thing it
means in replay/state.py: an observational record of a real, dated
occurrence and its real, dated outcome, never a Freqtrade order. Kept
entirely separate from execution/signal_store.py and
execution/live_battery_state.json (the now-superseded TP/SL-execution
state), and from replay/state.py's own files (simulated, isolated).
"""
from __future__ import annotations

import json
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent
TRADE_LOG_PATH = STATE_DIR / "live_tests.json"
HORIZONS_PATH = STATE_DIR / "horizons.json"


def _read(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _write(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))


def load_trade_log() -> list[dict]:
    return _read(TRADE_LOG_PATH, [])


def append_trade(entry: dict) -> int:
    log = load_trade_log()
    trade_id = len(log)
    log.append({"id": trade_id, "status": "open", **entry})
    _write(TRADE_LOG_PATH, log)
    return trade_id


def update_trade(trade_id: int, updates: dict) -> None:
    log = load_trade_log()
    for e in log:
        if e["id"] == trade_id:
            e.update(updates)
            break
    _write(TRADE_LOG_PATH, log)


def load_open_trades() -> list[dict]:
    return [t for t in load_trade_log() if t["status"] == "open"]


def load_horizons() -> dict:
    """Every candidate's most recent empirically-derived horizon --
    mirrors replay/state.py::load_horizons exactly. A candidate absent
    here has never had enough data yet; `_open_live_test` falls back to
    the documented placeholder in that case."""
    return _read(HORIZONS_PATH, {})


def save_horizons(horizons: dict) -> None:
    _write(HORIZONS_PATH, horizons)
