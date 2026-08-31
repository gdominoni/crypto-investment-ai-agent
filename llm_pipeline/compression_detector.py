"""Live detection of confirmed exits from volatility compression -- the
production counterpart of the replay's only trigger.

Replaces `shock_detector.py` as the market-data escalation path. The reasoning
is in docs/case_study/methodology-decisions.md and is not repeated here, except
for the one line that matters: this project looks for the causes of a TREND, and
a volatility shock is not a trend -- measured, post-shock days produce a defined
trend LESS often than ordinary days. Compression precedes one.

Reuses `candidates.methodology.compression_exit` directly -- the exact function
the replay triggers on -- so "compression exit" means one thing in this project
rather than a second definition that drifts. That discipline is why
`shock_detector` imported `shock_zscore_series` instead of reimplementing it.

DEDUPLICATION IS NOT OPTIONAL HERE, and its absence was a real defect in the
path this replaces. The daemon scans hourly, and `scan_for_shocks` tested the
volatility STATE (`z >= 2`), not a transition -- so every hour of a multi-day
shock re-escalated the same market to Sonnet. The replay avoided this with
`_shock_transition`; production never did. A confirmed compression exit is a
discrete event, but it stays true for the whole day it is detected, so the same
episode would still fire ~24 times without the ledger below.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from candidates.atomic_json import read_json, write_json
from candidates.data_loading import load_daily
from candidates.methodology import COMPRESSION_CONFIRM_DAYS, compression_exit

# One line per (coin, exit date) already escalated. Keyed on the EXIT date, not
# on the detection date: the same episode detected twice must collapse to one
# entry, which is the whole point.
ESCALATED_PATH = Path(__file__).resolve().parent.parent / "compression_escalated.json"

# How long a ledger entry is kept. Long enough that a coin cannot re-escalate the
# same episode, short enough that the file does not grow without bound.
LEDGER_RETENTION_DAYS = 365


def current_compression_exit(symbol: str) -> dict | None:
    """A confirmed compression exit for this coin, as of the latest bar on disk,
    or None.

    The exit itself (point B) is `COMPRESSION_CONFIRM_DAYS` before the latest
    bar; the confirmation window is what makes today the day it can be reported.
    Everything in the returned episode is dated to B."""
    daily = load_daily(symbol)
    if len(daily) == 0:
        return None
    episode = compression_exit(daily, daily.index[-1])
    if episode is None:
        return None
    return {"symbol": symbol, **episode}


def _prune(ledger: dict) -> dict:
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=LEDGER_RETENTION_DAYS)
    return {k: v for k, v in ledger.items()
            if not k.startswith("_") and pd.Timestamp(v.get("b_date", "1970-01-01")) >= cutoff}


def already_escalated(symbol: str, b_date) -> bool:
    return f"{symbol}|{pd.Timestamp(b_date).date()}" in read_json(ESCALATED_PATH, {})


def mark_escalated(symbol: str, b_date) -> None:
    ledger = _prune(read_json(ESCALATED_PATH, {}))
    ledger[f"{symbol}|{pd.Timestamp(b_date).date()}"] = {
        "b_date": str(pd.Timestamp(b_date).date()),
        "escalated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(ESCALATED_PATH, ledger)


def scan_for_compression_exits(coins: list[str]) -> list[dict]:
    """Every coin whose compression exit is confirmed as of today AND has not
    already been escalated. The second half is what keeps an hourly daemon from
    asking the same question twenty-four times."""
    found = []
    for coin in coins:
        try:
            episode = current_compression_exit(coin)
        except Exception:
            # One coin's missing or malformed data must not cost the others their
            # scan -- the same isolation discipline every other batch loop here uses.
            continue
        if episode is None or already_escalated(coin, episode["b_date"]):
            continue
        found.append(episode)
    return found
