"""Weekly re-validation entry point (intended to run under cron / a
scheduled task). Re-runs the full candidate battery against the latest
data, diffs each candidate's status against the previous run, and
notifies via Telegram only on an actual change -- a candidate that stays
'rejected' week after week doesn't need a repeated notification, but a
'validated' candidate degrading, or a 'watch' candidate clearing, always
does.

Also where two longer-horizon decisions surface: a candidate tracked 2+
years with no validation gets a keep-or-drop proposal (Sonnet's
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
from candidates.run_battery import ASSETS_DIR, run_all
from candidates.status_history import PRUNE_YEARS_THRESHOLD, mark_asked
from candidates.status_history import years_tracked as candidate_years_tracked
from llm_pipeline.dynamic_candidates import registered_specs
from llm_pipeline.haiku_sonnet_pipeline import sonnet_prune_advice
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
            return f"dynamic condition discovered via 'test it': {spec.indicator} {spec.op} {spec.threshold} -> {spec.direction}"
    return "trigger definition not found -- treat this as missing information, do not guess at it"


PRUNE_KEYBOARD_TEMPLATE = lambda candidate: {
    "inline_keyboard": [[
        {"text": "Keep Testing", "callback_data": f"prune:keep:{candidate}"},
        {"text": "Drop from Batch", "callback_data": f"prune:drop:{candidate}"},
    ]]
}


def run_weekly_revalidation() -> None:
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

    if meta["prune_candidates"]:
        load_dotenv()
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        for candidate in meta["prune_candidates"]:
            row = result[result["candidate"] == candidate]
            if len(row):
                r = row.iloc[0]
                summary = (f"win_rate={r['win_rate']:.1%}, sortino={r['sortino']:.2f}, N={r['n']}, "
                           f"{r['max_year_share']:.0%} of trades fell in {r['dominant_year']}, "
                           f"{r['max_coin_share']:.0%} on {r['dominant_coin'] or 'no single dominant coin'}")
            else:
                summary = "no current data"
            trigger_desc = _trigger_description(candidate)
            advice = sonnet_prune_advice(
                candidate, years_tracked=candidate_years_tracked(candidate) or PRUNE_YEARS_THRESHOLD,
                recent_summary=summary, trigger_description=trigger_desc, client=client,
            )
            sent = _send(
                f"<b>Keep-or-drop decision: {candidate}</b>\n\n"
                f"<b>Trigger:</b> {trigger_desc}\n\n"
                f"Tracked 2+ years, never reached 'validated'. Current: {summary}\n\n"
                f"Sonnet's opinion (advisory only, not a verified finding): {advice}",
                reply_markup=PRUNE_KEYBOARD_TEMPLATE(candidate),
            )
            print(f"Prune decision requested for '{candidate}': {'sent' if sent else 'SEND FAILED'}")
            mark_asked(candidate)


if __name__ == "__main__":
    run_weekly_revalidation()
