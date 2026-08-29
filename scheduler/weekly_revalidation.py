"""Weekly re-validation entry point (intended to run under cron / a
scheduled task). Re-runs the full candidate battery against the latest
data, diffs each candidate's status against the previous run, and
notifies via Telegram only on an actual change -- a candidate that stays
'rejected' week after week doesn't need a repeated notification, but an
'accepted' candidate degrading, or a 'watch' candidate clearing, always
does.

Also where two longer-horizon decisions surface: a candidate tracked 2+
years without ever being accepted gets a keep-or-drop proposal (Sonnet's
qualitative opinion, a human's actual decision -- see
candidates/status_history.py for why 2 years, not a re-test count: a
condition tied to a twice-yearly event needs calendar time to accumulate
occurrences, not a fixed number of re-tests), and if literally nothing
is left to test -- every static and dynamic candidate dropped, nothing
new proposed -- the system shuts itself down rather than run forever
with nothing to do.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from candidates.definitions import TRIGGER_DESCRIPTIONS
from candidates.methodology import explain_non_acceptance
from candidates.run_battery import ASSETS_DIR, run_all
from candidates.status_history import PRUNE_YEARS_THRESHOLD, mark_asked
from candidates.status_history import years_tracked as candidate_years_tracked
from data_ingestion.market_data.binance_fetcher import COINS as MARKET_DATA_COINS
from data_ingestion.market_data.binance_fetcher import update_all as update_market_data
from execution.live_testing import check_n50_milestones
from llm_pipeline.dynamic_candidates import registered_specs
from llm_pipeline.haiku_sonnet_pipeline import escape_html, sonnet_prune_advice
from llm_pipeline.novel_condition_tester import condition_desc
from telegram.bot import _send

PREVIOUS_STATUS_PATH = Path(__file__).resolve().parent / "previous_status.json"


def _trigger_description(candidate: str) -> str:
    """What the candidate's entry condition actually tests -- passed to
    Sonnet's prune advice so it reasons from the real trigger instead of
    inventing a mechanism from the label alone (see the "mean-reversion
    setup" story it fabricated for c1_long, a funding-rate-crowding
    trigger, before this existed)."""
    base = candidate.rsplit("_", 1)[0]
    if base in TRIGGER_DESCRIPTIONS:
        return TRIGGER_DESCRIPTIONS[base]
    for spec in registered_specs():
        if spec.label == candidate:
            return f"{condition_desc(spec)} → {spec.direction}"
    return "trigger definition not found -- treat this as missing information, do not guess at it"


PRUNE_KEYBOARD_TEMPLATE = lambda candidate: {
    "inline_keyboard": [[
        {"text": "Keep Testing", "callback_data": f"prune:keep:{candidate}"},
        {"text": "Drop from Batch", "callback_data": f"prune:drop:{candidate}"},
    ]]
}


def run_weekly_revalidation() -> None:
    """Refreshes local OHLCV/funding data from Binance BEFORE re-running
    the battery -- without this, `run_all()` would re-grade the exact
    same frozen snapshot every week and never see anything new. A fetch
    failure (network, exchange outage) is logged but doesn't block the
    run -- re-validating against last week's data is still more useful
    than not re-validating at all, and next week's run gets another
    chance to catch up.

    Everything past the data refresh is wrapped so a crash always sends
    a Telegram alert before re-raising -- without this, "no message this
    week" would be ambiguous between "nothing changed" (normal, expected)
    and "the whole run silently died" (a real failure nobody would
    otherwise notice until they happened to check the logs)."""
    try:
        update_market_data(MARKET_DATA_COINS)
    except Exception as e:
        print(f"Market data refresh failed, continuing with existing data: {e}")

    try:
        _run_weekly_revalidation()
    except Exception as e:
        message = (f"<b>Weekly re-validation crashed and did not complete.</b>\n\n"
                    f"{escape_html(type(e).__name__)}: {escape_html(str(e))}\n\n"
                    f"Nothing was updated this run -- check the process logs for the full traceback.")
        sent = _send(message)
        print(f"Failure alert: {'sent' if sent else 'SEND FAILED'}")
        raise


def _run_weekly_revalidation() -> None:
    result, live_state, meta = run_all()
    if meta.get("already_shut_down"):
        print("Weekly re-validation: system already shut down, nothing to do.")
        return

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(ASSETS_DIR / "candidate_battery_status.csv", index=False)

    from candidates.run_battery import SIGNAL_STORE_PATH
    SIGNAL_STORE_PATH.write_text(json.dumps(live_state, indent=2))

    if meta["shutdown_triggered"]:
        # `result` is empty here -- every candidate was dropped this run,
        # so there's nothing left to diff status against or propose a
        # prune decision for. Just notify and stop.
        sent = _send("<b>Candidates exhausted. Terminating the process.</b>\n\nEvery static and dynamic candidate has been "
                     "dropped or exhausted, and none have been proposed to replace them. Weekly re-validation will no "
                     "longer run until a new candidate is added.")
        print(f"Shutdown notice: {'sent' if sent else 'SEND FAILED'}")
        return

    current = dict(zip(result["candidate"], result["status"]))
    previous = json.loads(PREVIOUS_STATUS_PATH.read_text()) if PREVIOUS_STATUS_PATH.exists() else {}
    changes = [(c, previous.get(c, "never run"), s) for c, s in current.items() if previous.get(c) != s]
    PREVIOUS_STATUS_PATH.write_text(json.dumps(current, indent=2))

    if changes:
        lines = ["Weekly re-validation -- status changes:"]
        for candidate, old, new in changes:
            lines.append(f"  {candidate}: {old} -> {new}")
        message = "\n".join(lines)
        print(message)
        _send(message)
    else:
        print("Weekly re-validation: no status changes.")

    # Mirrors replay/engine.py's own horizon-change notice exactly -- run_all()
    # re-derives (and re-syncs to the file _open_live_test actually reads) every
    # tracked candidate's horizon this run, independent of accepted/watch/rejected;
    # this surfaces it to a human only on the runs it actually changed.
    if "horizon_changed_to" in result.columns:
        for _, r in result[result["horizon_changed_to"].notna()].iterrows():
            _send(f"<b>Horizon updated -- {escape_html(r['candidate'])}</b>\n\n"
                  f"({escape_html(_trigger_description(r['candidate']))})\n\n"
                  f"Now held for <b>{int(r['horizon_changed_to'])}d</b> going forward (empirically "
                  f"re-derived from accumulated history, replacing the previous value).")

    if meta.get("failed_candidates"):
        failed_msg = (f"<b>{len(meta['failed_candidates'])} candidate(s) failed to process this run "
                       f"(will retry next week):</b> {escape_html(', '.join(meta['failed_candidates']))}")
        print(failed_msg)
        _send(failed_msg)

    if meta["prune_candidates"]:
        load_dotenv()
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        for candidate in meta["prune_candidates"]:
            row = result[result["candidate"] == candidate]
            if len(row) and row.iloc[0]["status"] == "error":
                reason = "this week's run failed to process this candidate (see the failed-candidates notice above) -- its long-term status is unavailable this run"
                summary = "no current data (processing error this run)"
            elif len(row):
                r = row.iloc[0]
                reason = explain_non_acceptance(r.to_dict())
                summary = (f"win_rate={r['win_rate']:.1%}, sortino={r['sortino']:.2f}, N={r['n']} "
                           f"(reference TP/SL-structure stats, informational only, don't gate acceptance)"
                           if r["status"] != "insufficient_data" else "not enough historical occurrences yet to compute reference stats")
            else:
                reason = "no current data"
                summary = "no current data"
            trigger_desc = _trigger_description(candidate)
            advice = sonnet_prune_advice(
                candidate, years_tracked=candidate_years_tracked(candidate) or PRUNE_YEARS_THRESHOLD,
                recent_summary=summary, trigger_description=trigger_desc, client=client,
            )
            sent = _send(
                f"<b>Keep-or-drop decision -- {escape_html(candidate)}</b>\n\n"
                f"({escape_html(trigger_desc)})\n\n"
                f"Tracked 2+ years, never reached 'accepted'.\n"
                f"<b>Why:</b> {escape_html(reason)}.\n\n"
                f"For reference: {escape_html(summary)}.\n\n"
                f"Sonnet's opinion (advisory only, not a verified finding): {escape_html(advice)}",
                reply_markup=PRUNE_KEYBOARD_TEMPLATE(candidate),
            )
            print(f"Prune decision requested for '{candidate}': {'sent' if sent else 'SEND FAILED'}")
            mark_asked(candidate)

    load_dotenv()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    status_summary = result.set_index("candidate").to_dict(orient="index")
    check_n50_milestones(status_summary, client)


if __name__ == "__main__":
    run_weekly_revalidation()
