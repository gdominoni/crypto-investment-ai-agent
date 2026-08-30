"""Builds Sonnet's context and technical snapshot from live state, never
from a hand-maintained file -- a hand-maintained context file is exactly
what went stale in the prior project (it kept describing an architecture
that had already moved on). Every fact here is read fresh, each call."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from execution import live_test_state
from execution.signal_store import BATTERY_STATE_PATH, load_battery_state
from llm_pipeline.dynamic_candidates import rejected_labels


def build_context_summary() -> str:
    """This project never opens a funded position -- 'accepted' means a
    candidate's own trigger opens an observational live test
    automatically, not that anything is 'traded'. This text is Sonnet's
    own context for free-text answers, so it deliberately avoids
    'trade'/'trading' framing a reader could mistake for real capital
    being at risk (see docs/case_study/methodology-decisions.md)."""
    battery = load_battery_state()
    if not battery.get("candidates"):
        candidates_line = "No candidate currently has 'accepted' status -- nothing is currently opening live tests on its own trigger right now."
    else:
        names = ", ".join(battery["candidates"].keys())
        candidates_line = f"Currently accepted (live-testing automatically on their own trigger): {names}."
    generated_at = battery.get("generated_at", "never run")

    already_tested = rejected_labels()
    tested_line = (
        f"Already tested via 'test it' and NOT currently accepted (do not re-propose these without genuinely new "
        f"evidence, not just seeing the same pattern again): {', '.join(sorted(already_tested))}."
        if already_tested else
        "No novel conditions have been tested yet."
    )

    return (
        f"CANDIDATE BATTERY STATUS (refreshed weekly, last run: {generated_at}):\n"
        f"{candidates_line}\n"
        f"A candidate not listed as accepted has not cleared the statistical bar -- it either failed the "
        f"walk-forward backtest, a per-coin/per-year concentration check, or (if unclassified) hasn't accumulated "
        f"enough historical occurrences yet ('insufficient data'). Full status table: "
        f"{BATTERY_STATE_PATH.parent.parent / 'docs/case_study/assets/candidate_battery_status.csv'}.\n"
        f"{tested_line}"
    )


def build_live_test_summary(top_n: int = 15) -> str:
    """Real forward-return/positive-rate per candidate AND per
    (candidate, coin), computed on production's own resolved LIVE tests
    (execution/live_test_state.py -- no TP/SL, no funded position: this
    project is a pattern-discovery investigation, never an investment
    strategy). Mirrors replay/judgment.py's own two summaries exactly,
    handed to Sonnet as already-computed fact, matching this project's
    "cite only given numbers" discipline.

    Each breakdown is capped at `top_n` rows (ranked by |mean_return|) --
    without this, both blocks grow with the number of candidates/live
    tests ever accumulated, which only ever increases, so every future
    question would cost more just to carry old history nobody asked
    about (an effect measured directly in the replay: ~10,400 tokens of
    context at 96 tracked candidates, 74% of it these two blocks alone).
    A truncation note points to /summary (free, local, no LLM) for the
    full list."""
    import numpy as np

    closed = [t for t in live_test_state.load_trade_log() if t["status"] == "closed"]
    if not closed:
        return "No resolved live tests yet."
    by_candidate: dict[str, list[dict]] = {}
    by_pair: dict[tuple[str, str], list[dict]] = {}
    for t in closed:
        by_candidate.setdefault(t["candidate"], []).append(t)
        by_pair.setdefault((t["candidate"], t["coin"]), []).append(t)

    def _ranked_lines(groups: dict, name_fmt) -> tuple[list[str], int]:
        rows = []
        for key, trades in groups.items():
            n = len(trades)
            positive = sum(1 for t in trades if t["forward_return"] > 0)
            returns = np.array([t["forward_return"] for t in trades])
            rows.append((abs(returns.mean()), f"  {name_fmt(key)}: N={n}, positive_rate={positive / n:.1%}, mean_return={returns.mean():+.2%}"))
        rows.sort(key=lambda r: r[0], reverse=True)
        return [line for _, line in rows[:top_n]], len(rows)

    candidate_lines, n_candidates = _ranked_lines(by_candidate, lambda c: c)
    pair_lines, n_pairs = _ranked_lines(by_pair, lambda p: f"{p[0]} / {p[1]}")

    header1 = "By-candidate breakdown (all resolved live tests)"
    if n_candidates > top_n:
        header1 += f", top {top_n} of {n_candidates} by |mean return| -- ask /summary for the full list"
    header2 = "By-candidate-and-coin breakdown (all resolved live tests)"
    if n_pairs > top_n:
        header2 += f", top {top_n} of {n_pairs} by |mean return| -- ask /summary for the full list"

    return "\n".join([header1 + ":"] + candidate_lines + [header2 + ":"] + pair_lines)


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
        # This is the NORMAL, permanent state of this project, not a
        # misconfiguration: no funded position is ever opened, so
        # Freqtrade never places an order and never creates this
        # database. The message used to say "flag this as a setup
        # issue", which told Sonnet to warn the human their install was
        # broken every single time they asked a free-text market
        # question. Open live tests are real and are passed to Sonnet
        # separately (execution/live_test_state.py); they are what
        # "anything open right now" actually means here.
        return (f"Asset: {asset}. No funded positions exist -- by design, this project never "
                f"places a real order, so there is no trade database. This is expected, NOT a "
                f"setup problem: do not describe it as one. Open observational live tests, if "
                f"any, are listed separately below.")

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
