"""Shared read/write interface for the two sources of a live entry
signal: the weekly-refreshed candidate battery (`live_battery_state.json`,
written by `candidates/run_battery.py`) and Sonnet-approved manual
signals (`pending_manual_signals.json`, written by the LLM pipeline after
a human explicitly approves a novel-condition test or a direct trade
recommendation). The execution engine reads both through this module so
neither path is ever queried with ad-hoc, one-off file parsing."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent
BATTERY_STATE_PATH = STATE_DIR / "live_battery_state.json"
MANUAL_SIGNALS_PATH = STATE_DIR / "pending_manual_signals.json"
ACTIVE_SIGNALS_PATH = STATE_DIR / "active_manual_signals.json"


def load_battery_state() -> dict:
    if not BATTERY_STATE_PATH.exists():
        return {"generated_at": None, "horizons_days": [], "candidates": {}}
    return json.loads(BATTERY_STATE_PATH.read_text())


def load_manual_signals() -> list[dict]:
    if not MANUAL_SIGNALS_PATH.exists():
        return []
    return json.loads(MANUAL_SIGNALS_PATH.read_text()).get("pending", [])


def push_manual_signal(coin: str, direction: str, tp_mult: float, sl_mult: float,
                        anchors: dict, reasoning: str, approved_by: str, expires_hours: float = 24.0,
                        signal_class: str = "manual") -> None:
    """Called only after an explicit human approval (never by Sonnet on
    its own) -- see llm_pipeline's escalation flow. `anchors` must come
    from a real walk-forward-validated ad-hoc test of the flagged
    condition, not be invented on the spot.

    `signal_class` distinguishes a routine Sonnet-approved trade
    ('manual', the default) from one triggered by the real-time shock
    detector ('shock_reactive', see `llm_pipeline/shock_detector.py`) --
    carried through into the strategy's `enter_tag` so KPI reporting can
    measure critical-phase performance separately from routine trades."""
    pending = load_manual_signals()
    now = datetime.now(timezone.utc)
    pending = [p for p in pending if datetime.fromisoformat(p["expires_at"]) > now]  # drop stale entries
    pending.append({
        "coin": coin, "direction": direction, "tp_mult": tp_mult, "sl_mult": sl_mult, "anchors": anchors,
        "reasoning": reasoning, "approved_by": approved_by, "signal_class": signal_class,
        "issued_at": now.isoformat(), "expires_at": (now + timedelta(hours=expires_hours)).isoformat(),
    })
    MANUAL_SIGNALS_PATH.write_text(json.dumps({"pending": pending}, indent=2))


def peek_pending_manual_signal(coin: str, direction: str) -> dict | None:
    """Read-only lookup used by `populate_entry_trend` to decide whether
    to set enter_long/enter_short for this pair, and to read the
    signal's `signal_class` into the entry tag -- does NOT consume the
    signal (only `confirm_trade_entry`, called once per actual entry,
    does that via `consume_manual_signal`), so it's safe to call on
    every indicator refresh."""
    now = datetime.now(timezone.utc)
    for p in load_manual_signals():
        if p["coin"] == coin and p["direction"] == direction and datetime.fromisoformat(p["expires_at"]) > now:
            return p
    return None


def consume_manual_signal(coin: str, direction: str) -> dict | None:
    """Pops the first matching, unexpired manual signal for (coin,
    direction) and moves it into the ACTIVE store keyed by (coin,
    direction) -- called exactly once, from `confirm_trade_entry`, at the
    moment a trade actually opens. `custom_exit`/`custom_exit_price` read
    the ACTIVE store afterward (`get_active_manual_signal`, read-only) to
    recover this same trade's anchors days later; they must never call
    this function themselves, or they'd silently consume a DIFFERENT,
    unrelated signal that happened to arrive for the same pair since."""
    pending = load_manual_signals()
    now = datetime.now(timezone.utc)
    match_idx = None
    for i, p in enumerate(pending):
        if p["coin"] == coin and p["direction"] == direction and datetime.fromisoformat(p["expires_at"]) > now:
            match_idx = i
            break
    if match_idx is None:
        return None
    matched = pending.pop(match_idx)
    MANUAL_SIGNALS_PATH.write_text(json.dumps({"pending": pending}, indent=2))
    _write_active_signal(coin, direction, matched)
    return matched


def _write_active_signal(coin: str, direction: str, signal: dict) -> None:
    active = json.loads(ACTIVE_SIGNALS_PATH.read_text()) if ACTIVE_SIGNALS_PATH.exists() else {}
    active[f"{coin}:{direction}"] = signal
    ACTIVE_SIGNALS_PATH.write_text(json.dumps(active, indent=2))


def get_active_manual_signal(coin: str, direction: str) -> dict | None:
    """Read-only lookup for a currently-open manually-sourced trade's
    original anchors/tp_mult/sl_mult -- used by `custom_exit` and
    `custom_exit_price`, which must recover the SAME signal used at
    entry, not whatever else happens to be pending now."""
    if not ACTIVE_SIGNALS_PATH.exists():
        return None
    active = json.loads(ACTIVE_SIGNALS_PATH.read_text())
    return active.get(f"{coin}:{direction}")
