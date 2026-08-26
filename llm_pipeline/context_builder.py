"""Builds Sonnet's context and technical snapshot from live state, never
from a hand-maintained file -- a hand-maintained context file is exactly
what went stale in the prior project (it kept describing an architecture
that had already moved on). Every fact here is read fresh, each call."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from execution.signal_store import BATTERY_STATE_PATH, load_battery_state


def build_context_summary() -> str:
    battery = load_battery_state()
    if not battery.get("candidates"):
        candidates_line = "No candidate currently has 'validated' status -- nothing in the battery is allowed to trade unattended right now."
    else:
        names = ", ".join(battery["candidates"].keys())
        candidates_line = f"Currently validated (unattended-trade-eligible) candidates: {names}."
    generated_at = battery.get("generated_at", "never run")
    return (
        f"CANDIDATE BATTERY STATUS (refreshed weekly, last run: {generated_at}):\n"
        f"{candidates_line}\n"
        f"A candidate not listed as validated must never be treated as a live trading signal, "
        f"regardless of how positive it looks in isolation -- it either failed walk-forward "
        f"validation or a per-coin/per-year concentration check. Full status table: "
        f"{BATTERY_STATE_PATH.parent.parent / 'docs/case_study/assets/candidate_battery_status.csv'}."
    )


def build_technical_snapshot(asset: str, freqtrade_db_path: str | Path) -> str:
    """Reads Freqtrade's actual SQLite trade database for real open-
    position state -- the prior project's version of this function was a
    static placeholder that never got wired up; this one is real from
    the start."""
    db_path = Path(freqtrade_db_path)
    if not db_path.exists():
        return f"Asset: {asset}. No Freqtrade database found at {db_path} -- treat as no open positions, but flag this as a setup issue, not a genuine flat-market read."

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT pair, is_short, open_rate, open_date, enter_tag "
            "FROM trades WHERE is_open = 1 AND pair LIKE ?",
            (f"{asset}/%",),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return f"Asset: {asset}. No open position right now."

    lines = [f"Asset: {asset}. {len(rows)} open position(s):"]
    for pair, is_short, open_rate, open_date, enter_tag in rows:
        direction = "SHORT" if is_short else "LONG"
        tag = enter_tag or "manual/Sonnet-approved signal"
        lines.append(f"  - {pair} {direction}, opened {open_date} at {open_rate}, source: {tag}")
    return "\n".join(lines)
