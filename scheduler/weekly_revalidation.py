"""Weekly re-validation entry point (intended to run under cron / a
scheduled task). Re-runs the full candidate battery against the latest
data, diffs each candidate's status against the previous run, and
notifies via Telegram only on an actual change -- a candidate that stays
'rejected' week after week doesn't need a repeated notification, but a
'validated' candidate degrading, or a 'watch' candidate clearing, always
does.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from candidates.run_battery import ASSETS_DIR, run_all
from telegram.bot import _send

PREVIOUS_STATUS_PATH = Path(__file__).resolve().parent / "previous_status.json"


def run_weekly_revalidation() -> None:
    result, live_state = run_all()

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(ASSETS_DIR / "candidate_battery_status.csv", index=False)

    from candidates.run_battery import SIGNAL_STORE_PATH
    SIGNAL_STORE_PATH.write_text(json.dumps(live_state, indent=2))

    current = dict(zip(result["candidate"], result["status"]))
    previous = json.loads(PREVIOUS_STATUS_PATH.read_text()) if PREVIOUS_STATUS_PATH.exists() else {}

    changes = [(c, previous.get(c, "never run"), s) for c, s in current.items() if previous.get(c) != s]
    PREVIOUS_STATUS_PATH.write_text(json.dumps(current, indent=2))

    if not changes:
        print("Weekly re-validation: no status changes.")
        return

    lines = ["Weekly re-validation -- status changes:"]
    for candidate, old, new in changes:
        lines.append(f"  {candidate}: {old} -> {new}")
    message = "\n".join(lines)
    print(message)
    _send(message)


if __name__ == "__main__":
    run_weekly_revalidation()
