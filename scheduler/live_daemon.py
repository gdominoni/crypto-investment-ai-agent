"""Single entry point for running this project live -- one process, one
command, no external cron. This is an AI agent project: a human should
only ever need to interact with it through Telegram, never maintain
scheduling infrastructure by hand -- a misconfigured crontab entry
(wrong path, missing environment variables, a timezone mismatch) is
exactly the kind of silent failure this project has been careful to
avoid everywhere else (see PROJECT_MAP.md's "Partial Failures &
Crashes"). This daemon owns the Telegram long-poll loop itself (the
same one `telegram/bot.py::run_bot()` runs standalone) and, between
polls, checks whether it's time to run the hourly scans or the weekly
re-validation -- all three keep running as long as this one process
does.

Run as:
    python3 -m scheduler.live_daemon

Each scheduled job is isolated exactly like every other batch loop in
this project (see PROJECT_MAP.md's "Partial Failures & Crashes"): one
job crashing sends a Telegram alert and is retried on its next normal
cycle, never taking the whole daemon down and never failing silently.
Last-run timestamps persist across restarts (`live_daemon_state.json`,
gitignored -- real runtime state, not a durable record) so a restart
doesn't immediately re-fire a job that already ran recently.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from candidates.atomic_json import write_json
from execution.live_testing import run_once as run_live_testing
from llm_pipeline.haiku_sonnet_pipeline import run_compression_scan
from llm_pipeline.haiku_sonnet_pipeline import run_once as run_headline_scan
from scheduler.weekly_revalidation import run_weekly_revalidation
from telegram.bot import _dispatch_update, _get_updates, _send

STATE_PATH = Path(__file__).resolve().parent / "live_daemon_state.json"

HOURLY_INTERVAL = timedelta(hours=1)
WEEKLY_INTERVAL = timedelta(days=7)
POLL_TIMEOUT = 25  # seconds -- how long each Telegram long-poll waits for a new update
POLL_FAILURE_BACKOFF = 10  # seconds -- avoid hammering Telegram's API during a real outage


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def _save_state(state: dict) -> None:
    write_json(STATE_PATH, state)


def _run_isolated(name: str, fn, alert_on_failure: bool = True) -> None:
    """One scheduled job failing must never kill the daemon itself, and
    must never vanish silently either -- same discipline every other
    batch loop in this project already follows. `alert_on_failure=False`
    for jobs (currently only weekly re-validation) that already send
    their own, more specific Telegram alert internally before
    re-raising -- avoids reporting the same real failure twice."""
    try:
        fn()
    except Exception as e:
        print(f"[live_daemon] {name} failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        if alert_on_failure:
            try:
                _send(f"<b>Live daemon: {name} failed.</b>\n\n{type(e).__name__}: {e}\n\nWill retry on its normal schedule.")
            except Exception:
                pass  # a failed alert must not crash the daemon either


def run_forever() -> None:
    load_dotenv()
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    state = _load_state()
    now = datetime.now(timezone.utc)
    last_hourly = datetime.fromisoformat(state["last_hourly"]) if "last_hourly" in state else now - HOURLY_INTERVAL
    last_weekly = datetime.fromisoformat(state["last_weekly"]) if "last_weekly" in state else now - WEEKLY_INTERVAL

    offset = None
    print("Live daemon started -- Telegram bot, hourly scans, and weekly re-validation all running in this one process.")
    _send("<b>Live daemon started.</b> Hourly scans and weekly re-validation are now running automatically.")

    while True:
        try:
            updates = _get_updates(token, offset, timeout=POLL_TIMEOUT)
            for update in updates:
                offset = update["update_id"] + 1
                try:
                    _dispatch_update(update, client)
                except Exception as e:
                    print(f"[live_daemon] Telegram update dispatch failed: {e}")
        except Exception as e:
            # Telegram's own getUpdates call isn't isolated by _dispatch_update's
            # own try/except (a known gap, documented in PROJECT_MAP.md) -- this
            # is what actually closes it: a network blip or API error here no
            # longer crashes the whole daemon, just this one poll iteration.
            print(f"[live_daemon] Telegram poll failed, will retry: {e}")
            time.sleep(POLL_FAILURE_BACKOFF)

        now = datetime.now(timezone.utc)
        if now - last_hourly >= HOURLY_INTERVAL:
            _run_isolated("mechanical trigger scan", run_live_testing)
            _run_isolated("headline scan", run_headline_scan)
            _run_isolated("compression scan", run_compression_scan)
            last_hourly = now
            state["last_hourly"] = now.isoformat()
            _save_state(state)

        if now - last_weekly >= WEEKLY_INTERVAL:
            _run_isolated("weekly re-validation", run_weekly_revalidation, alert_on_failure=False)
            last_weekly = now
            state["last_weekly"] = now.isoformat()
            _save_state(state)


if __name__ == "__main__":
    run_forever()
