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
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from candidates.atomic_json import write_json
from candidates.run_battery import COINS
from data_ingestion.market_data.binance_fetcher import update_all as update_market_data
from execution.hyperopt_runner import pending_work_reminder
from execution.live_testing import _check_parked_proposals, run_once as run_live_testing, send_monthly_digest
from llm_pipeline.haiku_sonnet_pipeline import run_compression_scan
from scheduler.weekly_revalidation import run_weekly_revalidation
from telegram.bot import _dispatch_update, _get_updates, _send

STATE_PATH = Path(__file__).resolve().parent / "live_daemon_state.json"

HOURLY_INTERVAL = timedelta(hours=1)
DAILY_INTERVAL = timedelta(days=1)
WEEKLY_INTERVAL = timedelta(days=7)
MONTHLY_INTERVAL = timedelta(days=30)
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


def _remind_about_local_work() -> None:
    """Silent when there is nothing to run -- see pending_work_reminder."""
    message = pending_work_reminder()
    if message:
        _send(message)


def _heartbeat() -> None:
    """Pings an external dead-man's switch, if one is configured.

    Everything else in this project reports its own failures, and that works
    because something is still alive to do the reporting. This is the one
    failure that breaks the pattern: a daemon cannot tell you it has stopped,
    and the host dying takes the messenger with the message. Silence then looks
    exactly like a quiet week, which is the normal, healthy state here -- so a
    human would not notice for a long time.

    Only an OUTSIDE observer can catch that, and it has to work by expecting
    something rather than watching for something: a service that alerts when a
    ping does NOT arrive. Set HEARTBEAT_URL to a check URL from any dead-man's
    switch service (healthchecks.io and similar have free tiers) and this pings
    it after every completed hourly cycle; configure that check to alert if it
    goes quiet for a few hours.

    Unset, this does nothing at all -- the whole system works without it, and
    it stays opt-in rather than making a third-party account a requirement.

    Deliberately NOT wired to a Telegram alert on failure: a transient network
    error here is not worth a message, and a real outage is precisely what the
    external service is already going to tell you about.
    """
    url = os.environ.get("HEARTBEAT_URL")
    if not url:
        return
    import requests

    requests.get(url, timeout=10)


def run_forever() -> None:
    load_dotenv()
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    state = _load_state()
    now = datetime.now(timezone.utc)
    last_hourly = datetime.fromisoformat(state["last_hourly"]) if "last_hourly" in state else now - HOURLY_INTERVAL
    last_daily = datetime.fromisoformat(state["last_daily"]) if "last_daily" in state else now - DAILY_INTERVAL
    last_weekly = datetime.fromisoformat(state["last_weekly"]) if "last_weekly" in state else now - WEEKLY_INTERVAL
    # Not backdated on a first run, unlike the others: the digest reports a
    # PERIOD, and firing one immediately would report a period that never ran.
    last_monthly = datetime.fromisoformat(state["last_monthly"]) if "last_monthly" in state else now

    offset = None
    print("Live daemon started -- Telegram bot, hourly scans, daily parked-proposal re-check, and weekly "
          "re-validation all running in this one process.")
    _send("<b>Live daemon started.</b> Hourly scans, the daily parked-proposal re-check, and weekly "
          "re-validation are now running automatically.")

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
            # Refresh FIRST, as its own job. Both scans below read the hourly
            # OHLCV this writes, and the refresh used to live inside the
            # compression scan -- which runs second, so the mechanical scan
            # always read data an hour older than it had to. Harmless in a
            # long-running daemon (a 24-hour window losing its newest hour),
            # but not on a freshly deployed host: the very first mechanical
            # scan would run against whatever stale candles the repo shipped
            # with. Hoisting it also means a market-data outage is reported as
            # a market-data failure, instead of surfacing as a compression
            # scan that quietly scanned old data.
            _run_isolated("market data refresh", lambda: update_market_data(COINS))
            _run_isolated("mechanical trigger scan", run_live_testing)
            # The hourly Haiku headline scan was removed on 2026-09-02: nothing
            # it surfaced could enter a testable hypothesis, and a backfill was
            # measured not to fix that. See llm_pipeline/haiku_sonnet_pipeline.py's
            # module docstring and docs/case_study/methodology-decisions.md.
            _run_isolated("compression scan", lambda: run_compression_scan(refresh=False))
            # Last, and only after the real work: the point is to signal that a
            # full cycle completed, not merely that the process is running. A
            # daemon looping without ever finishing a cycle would still be a
            # dead system, and pinging earlier would report it as healthy.
            _run_isolated("heartbeat", _heartbeat, alert_on_failure=False)
            last_hourly = now
            state["last_hourly"] = now.isoformat()
            _save_state(state)

        if now - last_daily >= DAILY_INTERVAL:
            # Cheap: even a few dozen parked proposals cost seconds, not the
            # hours a compressed nine-year replay would spend unstaggered --
            # see _check_parked_proposals's own docstring.
            _run_isolated("parked-proposal re-check", _check_parked_proposals)
            last_daily = now
            state["last_daily"] = now.isoformat()
            _save_state(state)

        if now - last_monthly >= MONTHLY_INTERVAL:
            _run_isolated("monthly digest", lambda: send_monthly_digest(pd.Timestamp(last_monthly).tz_localize(None)))
            # The one job that needs the human's own machine, and it only asks
            # when there is actually something to run -- a reminder that fires
            # on a schedule regardless is the kind of message this system
            # removed everywhere else. See pending_work_reminder's docstring.
            _run_isolated("local-task reminder", _remind_about_local_work)
            last_monthly = now
            state["last_monthly"] = now.isoformat()
            _save_state(state)

        if now - last_weekly >= WEEKLY_INTERVAL:
            _run_isolated("weekly re-validation", run_weekly_revalidation, alert_on_failure=False)
            last_weekly = now
            state["last_weekly"] = now.isoformat()
            _save_state(state)


if __name__ == "__main__":
    run_forever()
