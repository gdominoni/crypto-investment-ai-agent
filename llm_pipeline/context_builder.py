"""Builds Sonnet's context and technical snapshot from live state, never
from a hand-maintained file -- a hand-maintained context file is exactly
what went stale in the prior project (it kept describing an architecture
that had already moved on). Every fact here is read fresh, each call."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from execution.signal_store import BATTERY_STATE_PATH, load_battery_state
from llm_pipeline.dynamic_candidates import rejected_labels


def build_context_summary() -> str:
    battery = load_battery_state()
    if not battery.get("candidates"):
        candidates_line = "No candidate currently has 'validated' status -- nothing in the battery is allowed to trade unattended right now."
    else:
        names = ", ".join(battery["candidates"].keys())
        candidates_line = f"Currently validated (unattended-trade-eligible) candidates: {names}."
    generated_at = battery.get("generated_at", "never run")

    already_tested = rejected_labels()
    tested_line = (
        f"Already tested via 'test it' and NOT currently validated (do not re-propose these without genuinely new "
        f"evidence, not just seeing the same pattern again): {', '.join(sorted(already_tested))}."
        if already_tested else
        "No novel conditions have been tested yet."
    )

    return (
        f"CANDIDATE BATTERY STATUS (refreshed weekly, last run: {generated_at}):\n"
        f"{candidates_line}\n"
        f"A candidate not listed as validated must never be treated as a live trading signal, "
        f"regardless of how positive it looks in isolation -- it either failed walk-forward "
        f"validation or a per-coin/per-year concentration check. Full status table: "
        f"{BATTERY_STATE_PATH.parent.parent / 'docs/case_study/assets/candidate_battery_status.csv'}.\n"
        f"{tested_line}"
    )


def build_technical_snapshot(asset: str, freqtrade_db_path: str | Path) -> str:
    """Reads Freqtrade's actual SQLite trade database for real open-
    position and last-closed-trade state -- the prior project's version
    of this function was a static placeholder that never got wired up;
    this one is real from the start.

    `asset="MARKET"` (or blank) means a broad, non-coin-specific question
    ("how's the market", "any open trades") -- it queries across every
    pair, not `pair LIKE 'MARKET/%'`, which can never match a real pair
    like `BTC/USDT:USDT` and would silently report "no open position"
    even with real positions open."""
    db_path = Path(freqtrade_db_path)
    if not db_path.exists():
        return f"Asset: {asset}. No Freqtrade database found at {db_path} -- treat as no open positions, but flag this as a setup issue, not a genuine flat-market read."

    broad = not asset or asset.upper() == "MARKET"
    conn = sqlite3.connect(str(db_path))
    try:
        if broad:
            open_rows = conn.execute(
                "SELECT pair, is_short, open_rate, open_date, enter_tag FROM trades WHERE is_open = 1"
            ).fetchall()
            last_closed = conn.execute(
                "SELECT pair, is_short, close_profit, close_date, exit_reason FROM trades "
                "WHERE is_open = 0 ORDER BY close_date DESC LIMIT 1"
            ).fetchone()
        else:
            like = f"{asset}/%"
            open_rows = conn.execute(
                "SELECT pair, is_short, open_rate, open_date, enter_tag "
                "FROM trades WHERE is_open = 1 AND pair LIKE ?", (like,),
            ).fetchall()
            last_closed = conn.execute(
                "SELECT pair, is_short, close_profit, close_date, exit_reason FROM trades "
                "WHERE is_open = 0 AND pair LIKE ? ORDER BY close_date DESC LIMIT 1", (like,),
            ).fetchone()
    finally:
        conn.close()

    lines = [f"Asset: {asset}."]
    if open_rows:
        lines.append(f"{len(open_rows)} open position(s):")
        for pair, is_short, open_rate, open_date, enter_tag in open_rows:
            direction = "SHORT" if is_short else "LONG"
            tag = enter_tag or "manual/Sonnet-approved signal"
            lines.append(f"  - {pair} {direction}, opened {open_date} at {open_rate}, source: {tag}")
    else:
        lines.append("No open position right now.")

    if last_closed:
        pair, is_short, close_profit, close_date, exit_reason = last_closed
        direction = "SHORT" if is_short else "LONG"
        lines.append(f"Last closed trade: {pair} {direction}, {close_profit:+.2%}, closed {close_date} (reason: {exit_reason}).")

    return "\n".join(lines)
