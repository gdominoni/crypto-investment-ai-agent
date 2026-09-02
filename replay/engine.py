"""The historical replay's day-by-day walker.

Advances a simulated 'current date' forward: re-validates the candidate
battery on a weekly cadence (replay/battery.py, as-of data only), reacts
to real, dated macro releases and volatility shocks with the same
Sonnet judgment production uses (replay/judgment.py), and opens a
resulting LIVE TEST the moment it's decided -- but does NOT resolve it
in the same breath. An accepted candidate isn't traded with a TP/SL
position; a live occurrence is held for exactly the horizon
`pattern_significance` found significant at, and its outcome (forward
return, MFE, MAE) is what actually gets recorded -- the live test of the
same concept the backtest already measured, not a different barrier-
based strategy inspired by it. `_check_live_tests` re-checks every open
live test on every subsequent simulated day and resolves it the moment
its horizon has fully elapsed. Collapsing "opened" and "resolved" into
one instant would look nothing like how this would have actually
unfolded.

Stops immediately whenever a novel-condition test is proposed -- exactly
like production, this needs a real human decision (`resolve_pending_test`
/ `discard_pending_test`, wired to Telegram's "Test It" / "Don't Test It"
buttons) before anything downstream can be trusted -- and otherwise stops
every ~30 simulated days for a human checkpoint.
"""
from __future__ import annotations

import itertools
import os

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

from candidates.data_loading import load_daily, load_funding, load_hourly
from candidates.definitions import CANDIDATE_DIRECTIONS, TRIGGER_DESCRIPTIONS, compute_triggers
from candidates.methodology import (COMPRESSION_CONFIRM_DAYS, COMPRESSION_ZSCORE_THRESHOLD,
                                     compression_exit, format_prune_digest, path_outcome,
                                     MIN_INTERESTING_EFFECT, prospective_split,
                                     prune_recommendation, required_n_for_power,
                                     vol_compression_series)
from candidates.run_battery import COINS
from execution import hyperopt_runner
from llm_pipeline.haiku_sonnet_pipeline import escape_html, format_spec_clauses
from llm_pipeline.novel_condition_tester import (
    ConditionSpec, clause_signal_hourly, clause_to_dict, condition_desc, format_pattern_significance,
    MIN_HISTORICAL_OCCURRENCES, filter_redundant_proposals, count_occurrences, is_testable,
    proposals_from_assessment, relax_to_testable,
    spec_from_dict, spec_from_proposal, spec_to_dict,
    test_novel_condition,
)
from replay import judgment, state
from replay import status_history as sh
from replay.battery import run_replay_battery
from telegram.bot import _send

CHUNK_DAYS = 30
# The shock threshold used to live here, for the trigger. The trigger is gone
# (see _compression_exit); `shock_zscore` survives only as a snapshot reading and
# in the C1/C2/C6 static battery, both of which carry their own constant.


def _normalize_coin(coin: str) -> str | None:
    """The prompt tells Sonnet to use the exact symbol from `COINS`
    (e.g. "BTCUSDT"), but a prompt instruction is a request, not a
    guarantee -- this is the actual enforcement, matching the rest of
    this project's own discipline of never trusting model output where
    code can check it instead. A bare ticker ("BTC") is accepted and
    corrected; anything that still doesn't resolve is rejected rather
    than crashing on a file that doesn't exist."""
    if coin in COINS:
        return coin
    guess = f"{coin.upper()}USDT"
    return guess if guess in COINS else None


PLACEHOLDER_HORIZON_DAYS = 7  # neutral default (middle of HORIZONS_DAYS) -- see docs/case_study/methodology-decisions.md
CONSECUTIVE_FAILURE_ALERT_THRESHOLD = 2  # see docs/case_study/methodology-decisions.md
CONSECUTIVE_FAILURE_CONTEXT_WINDOW = 5


def _open_live_test(candidate: str, coin: str, direction: str, decision_date: pd.Timestamp) -> dict:
    """Entry is the next real bar's open after `decision_date` (same
    causality convention as `methodology.build_events`). This does NOT
    open a position with a TP/SL exit -- no capital, no barrier. It
    records a live occurrence of a TRACKED pattern (accepted, watch,
    rejected, or not yet classified -- testing starts the moment a
    trigger is identified, not only once it's already accepted), to be
    held for exactly the horizon `pattern_significance` found significant
    at (`replay/state.py::load_horizons`), or the documented placeholder
    if that candidate hasn't had enough data yet to derive one
    empirically. Resolved by `_check_live_tests` with the SAME forward-
    return/MFE/MAE measure `pattern_significance` uses -- so live testing
    measures literally the same concept the backtest measures, not a
    different barrier-based strategy inspired by it."""
    coin = _normalize_coin(coin)
    if coin is None:
        return {"opened": False, "message": f"REJECTED: coin not recognized (not in the replay's coin universe) -- refused."}
    horizon = int(state.load_horizons().get(candidate, PLACEHOLDER_HORIZON_DAYS))
    ohlc = load_daily(coin)
    after = ohlc.index[ohlc.index > decision_date]
    if len(after) == 0:
        return {"opened": False, "message": "No further price data available to open this live test yet."}
    entry_loc = ohlc.index.get_loc(after[0])
    entry_date = ohlc.index[entry_loc]
    entry_price = float(ohlc["open"].iloc[entry_loc])
    trade_id = state.append_trade({
        "candidate": candidate, "coin": coin, "direction": direction,
        "decision_date": str(decision_date.date()), "entry_date": str(entry_date.date()), "entry_price": entry_price,
        "entry_loc": int(entry_loc), "horizon": horizon,
    })
    return {"opened": True, "trade_id": trade_id, "coin": coin, "direction": direction, "candidate": candidate,
            "entry_date": entry_date, "horizon": horizon}


def _check_consecutive_failures(candidate: str, d: pd.Timestamp) -> None:
    """Mirrors execution/live_testing.py::_check_consecutive_failures
    exactly, against the replay's own simulated trade log/clock. Fires
    immediately after a live test resolves, only for a CONFIRMED
    candidate (milestone_cleared) -- a fast, purely informational
    early-warning for a genuine losing streak, since a well-established
    candidate's own aggregate significance test is, by design, resistant
    to short-term noise (verified: can absorb 20-30 consecutive
    worst-case-magnitude failures before flipping -- see
    docs/case_study/methodology-decisions.md), which is correct against
    noise but too slow to react to an actual regime change on its own.
    Never changes any candidate's status itself."""
    if not sh.all_latest_statuses().get(candidate, {}).get("milestone_cleared"):
        return
    closed = sorted((t for t in state.load_trade_log() if t["candidate"] == candidate and t["status"] == "closed"),
                     key=lambda t: t["close_date"])
    streak = 0
    for t in reversed(closed):
        if t["forward_return"] >= 0:
            break
        streak += 1
    if streak < CONSECUTIVE_FAILURE_ALERT_THRESHOLD:
        return
    window = closed[-max(streak, CONSECUTIVE_FAILURE_CONTEXT_WINDOW):]
    mean_return = sum(t["forward_return"] for t in window) / len(window)
    mean_mfe = sum(t["mfe"] for t in window) / len(window)
    mean_mae = sum(t["mae"] for t in window) / len(window)
    ratio = mean_mfe / mean_mae if mean_mae else float("nan")
    lines = [
        f"<b>{d.date()}</b>\n",
        f"<b>Consecutive-failure alert -- {escape_html(candidate)}</b>\n",
        f"The last <b>{streak}</b> live test(s) for this CONFIRMED candidate resolved negative in a row.\n",
        f"Last {len(window)} occurrence(s) for context:",
    ]
    for t in window:
        lines.append(f"  {t['close_date']}  {escape_html(t['coin'])}  return={t['forward_return']:+.2%}  "
                      f"MFE={t['mfe']:+.2%}  MAE={t['mae']:+.2%}")
    lines.append(f"\nOver these {len(window)}: mean return={mean_return:+.2%}, MFE/MAE={ratio:.2f} (favorable if > 1.0)")
    lines.append(f"\nInformational only -- this does not change <b>{escape_html(candidate)}</b>'s status. The full "
                 f"aggregate statistics (see /replay_details <b>{escape_html(candidate)}</b>) are far more resistant to a "
                 f"short streak by design; this exists specifically to surface a genuine losing run long before "
                 f"the aggregate ever would.")
    _send("\n".join(lines))



def _market_return_over(entry_loc: int, horizon: int, direction: str) -> float:
    """Equal-weighted forward return of the whole coin universe over the same
    window as one live test, signed by that test's direction.

    The comparison a raw win rate cannot make. "The trend happened" and "the
    trend happened because of this condition" are different claims, and across
    2017-2026 a long-only rule is right most of the time for reasons that have
    nothing to do with any macro release."""
    import numpy as np

    rets = []
    for coin in COINS:
        try:
            ohlc = load_daily(coin)
        except Exception:
            continue
        if entry_loc + horizon >= len(ohlc):
            continue
        entry = float(ohlc["open"].iloc[entry_loc])
        exit_ = float(ohlc["close"].iloc[entry_loc + horizon])
        if entry > 0:
            rets.append(exit_ / entry - 1.0)
    if not rets:
        return float("nan")
    r = float(np.mean(rets))
    return r if direction == "long" else -r




def _confirmation_block(candidate: str) -> str:
    """The running confirmation record, appended to every resolved live test.

    NOT a validation, and the second line is what keeps that honest. At this
    project's horizons `required_n_for_power` puts the sample needed to
    demonstrate a 5% effect at 80% power in the hundreds -- 307 occurrences over
    7 days, 742 over 14 -- against a typical candidate producing about 11
    independent occurrences a year. No count reachable in any realistic tracking
    window gets there. Printing the achieved count AS A FRACTION of the required
    one is what stops an accumulating counter from implying a proof: "23 of 307"
    cannot be misread the way a bare "23" can.

    Never raises. This runs inside the notification loop, so a missing statistic
    degrades to a stated gap rather than taking down the message carrying the
    actual result."""
    import numpy as np

    try:
        closed = [t for t in state.load_trade_log()
                  if t["candidate"] == candidate and t["status"] == "closed"]
        prior = state.load_confirmation_priors().get(candidate, 0)
        n = _effective_milestone_count(candidate, prior, len(closed))
        horizon = int(state.load_horizons().get(candidate, PLACEHOLDER_HORIZON_DAYS))
        sd = ((state.load_battery_status().get("summary") or {})
              .get(candidate, {}).get("pattern_oos_sd"))
        need = required_n_for_power(sd) if isinstance(sd, (int, float)) else float("nan")

        lines = [f"<b>Confirmation record -- {escape_html(candidate)}</b>"]
        if need == need:
            lines.append(f"Occurrence <b>{n} of {need:.0f}</b> "
                         f"(needed for a {MIN_INTERESTING_EFFECT:.0%} effect at 80% power over {horizon}d)")
        else:
            lines.append(f"Occurrence <b>{n}</b> (required sample not yet computable -- "
                         f"no volatility estimate for this candidate)")

        if closed:
            wins = sum(1 for t in closed if t["forward_return"] > 0)
            lines.append(f"Trend materialised: <b>{wins/len(closed):.0%}</b> of {len(closed)} resolved")
            mfe = float(np.mean([t["mfe"] for t in closed]))
            mae = float(np.mean([t["mae"] for t in closed]))
            ratio = mfe / abs(mae) if mae else float("nan")
            lines.append(f"Mean best point {mfe:+.2%}, mean worst {mae:+.2%}"
                         + (f" -- MFE/MAE {ratio:.2f}" if ratio == ratio else ""))
        else:
            lines.append("Trend materialised: no occurrence has resolved yet")
        lines.append(hyperopt_runner.format_result(candidate, short=True))
        return "\n".join(lines)
    except Exception as e:
        # One malformed statistic must not cost the message the outcome it exists
        # to report -- same isolation discipline as every other loop here.
        return f"<i>(confirmation record unavailable: {type(e).__name__})</i>"



def _check_live_tests(d: pd.Timestamp) -> None:
    """Runs on EVERY simulated day (not just event days) -- resolves a
    live test the moment its horizon has fully elapsed, no earlier and no
    barrier check in between: no TP, no SL, nothing that would make this
    measure a different concept than the one `pattern_significance`
    tested in backtest. `path_outcome` recomputes the SAME forward-
    return/MFE/MAE measure that function uses."""
    for trade in state.load_open_trades():
        entry_date = pd.Timestamp(trade["entry_date"])
        if d <= entry_date:
            continue  # entry bar itself excluded, matches methodology.py's own convention
        elapsed = (d - entry_date).days
        if elapsed < trade["horizon"]:
            continue
        ohlc = load_daily(trade["coin"])
        if d not in ohlc.index:
            continue
        outcome = path_outcome(trade["entry_price"], trade["entry_loc"], ohlc, trade["direction"], trade["horizon"])
        if outcome["forward_return"] != outcome["forward_return"]:
            # NaN: full horizon not actually available yet -- leave open, retry
            # next simulated day. Mirrors execution/live_testing.py exactly.
            continue
        state.update_trade(trade["id"], {
            "status": "closed", "close_date": str(d.date()),
            "forward_return": outcome["forward_return"], "mfe": outcome["mfe"], "mae": outcome["mae"],
            # What simply holding the whole coin universe over the same window
            # did, signed the same way. Stored at resolution rather than
            # recomputed later: it is the denominator for "did the trend happen
            # BECAUSE of the condition", and in a rising market a positive long
            # return on its own says very little.
            "baseline_return": _market_return_over(trade["entry_loc"], trade["horizon"], trade["direction"]),
        })
        _send(f"<b>{d.date()}</b>\n\n"
              f"<b>Live test resolved -- {trade['direction'].upper()} {trade['coin']}</b>\n\n"
              f"(candidate <b>{escape_html(trade['candidate'])}</b>: {escape_html(_trigger_description(trade['candidate']))}, "
              f"held {trade['horizon']}d, opened {trade['entry_date']})\n\n"
              f"Forward return: <b>{outcome['forward_return']:+.2%}</b>\n"
              f"Best point reached: {outcome['mfe']:+.2%}\n"
              f"Worst point reached: {outcome['mae']:+.2%}\n\n"
              + _confirmation_block(trade["candidate"]))
        _check_consecutive_failures(trade["candidate"], d)


# Only one replay proposal can ever be pending at a time (the replay
# halts until it's answered -- see advance()'s "waiting_for_human" check),
# so unlike production's PROPOSAL_KEYBOARD_TEMPLATE this needs no id.
REPLAY_PROPOSAL_KEYBOARD = {
    "inline_keyboard": [[
        {"text": "Test It", "callback_data": "replay_propose:test"},
        {"text": "Don't Test It", "callback_data": "replay_propose:skip"},
    ]]
}



def _prepare_proposal(raw: dict, as_of: pd.Timestamp) -> dict | None:
    """Validate one proposed condition and, if it is too rare, loosen it to the
    nearest measurable version. Returns None when it cannot be tested at all.

    Factored out of `_handle_assessment` when a call began returning more than
    one proposal: the per-proposal work is identical and duplicating it is how
    two paths drift into checking different things."""
    spec, err = spec_from_proposal(raw)
    if spec is None:
        print(f"[{as_of.date()}] proposal '{raw.get('label')}' rejected, not tested: {err}")
        return None
    # Rarity, measured rather than approximated. 0.25s of local computation and no
    # API call, so the thing that actually decides whether a hypothesis can
    # produce a result is checked directly.
    why_not = is_testable(spec, COINS, as_of=as_of)
    relax_note = None
    if why_not is not None:
        # "Too rare" is a statement about the THRESHOLDS, not about the idea.
        # Search locally for the nearest measurable version instead.
        #
        # This is a power calculation, not a result search: the search criterion
        # is the occurrence count and nothing else -- no return, no p-value, no
        # outcome is visible to `relax_to_testable`. Loosening until a condition
        # fires often enough to be measured is a sample-size decision; loosening
        # until it becomes significant would be p-hacking.
        relaxed = relax_to_testable(spec, COINS, as_of=as_of)
        if relaxed is None:
            # PARKED, not discarded. The condition is well-formed and on-thesis;
            # it simply has not happened enough times YET. Only 8% of this
            # grammar is testable as of January 2019, so discarding these threw
            # away most of what the first four years of a replay discovers, and
            # threw it away permanently -- nothing stored it.
            #
            # `_check_parked_proposals` re-checks them at the weekly battery
            # refresh, which costs no API call. And the wait makes the eventual
            # test stronger: a hypothesis written in 2019 and tested in 2022 is
            # tested partly on data that did not exist when it was written.
            state.park_proposal({"spec": spec_to_dict(spec), "proposed_at": str(as_of.date()),
                                  "reason": why_not})
            print(f"[{as_of.date()}] proposal '{raw.get('label')}' PARKED, not discarded: "
                  f"{why_not}. Re-checked weekly as history accumulates.")
            return None
        spec, relax_note = relaxed
        # Substituted, not silently: the condition about to be tested is NOT the
        # one proposed, and both the human approving it and the stored record
        # have to say so. `update` preserves keys the proposal carries beyond the
        # spec fields (rationale, and anything added later).
        raw.update(spec_to_dict(spec))
        raw["relaxed_from"] = relax_note
        print(f"[{as_of.date()}] proposal '{raw.get('label')}': {relax_note}")
    return {"spec": spec, "dict": raw, "relaxed_from": relax_note}



def _check_parked_proposals(as_of: pd.Timestamp, live_coin: str | None = None) -> dict | None:
    """Promote the OLDEST parked proposal that has become testable, if any.

    One per refresh, deliberately. The replay holds a single pending slot and
    halts on it, so promoting several at once would need a queue for no benefit
    -- with a weekly refresh there are roughly 470 opportunities across a nine
    year run, far more than the number of proposals that will ever be parked.

    Oldest first, not best first: choosing which parked hypothesis to promote by
    any measured quality would be selecting on the outcome, which is the one
    thing the proposal path must never do."""
    parked = state.load_parked_proposals()
    if not parked:
        return None
    for entry in sorted(parked, key=lambda e: e.get("proposed_at", "")):
        try:
            spec = spec_from_dict(entry["spec"])
        except ValueError:
            # Written under an older grammar and no longer expressible -- drop it
            # rather than let it block the queue forever.
            state.unpark_proposal(entry["spec"].get("label", ""))
            continue
        if is_testable(spec, COINS, as_of=as_of) is not None:
            continue
        state.unpark_proposal(spec.label)
        entry["spec"]["parked_since"] = entry.get("proposed_at")
        print(f"[{as_of.date()}] parked proposal '{spec.label}' is now testable "
              f"(proposed {entry.get('proposed_at')}) -- promoting.")
        return {"specs": [entry["spec"]], "coins": COINS, "live_coin": live_coin,
                "as_of": str(as_of.date()), "resume_from": str(as_of.date()),
                "proposed_at": entry.get("proposed_at")}
    return None



def _handle_assessment(as_of: pd.Timestamp, event_desc: str, assessment: dict, live_coin: str | None = None,
                        resume_from: pd.Timestamp | None = None) -> str | None:
    """Returns "STOP" if the replay must halt for a human decision, else
    None. Sonnet never opens a trade here -- that's the mechanical scan's
    job (see _scan_mechanical_triggers); this only ever proposes a novel
    condition for human approval, or does nothing."""
    # Equal on every path where the replay's clock IS the data cutoff; they
    # differ only for the compression trigger, which asks at C about B.
    resume_from = resume_from if resume_from is not None else as_of
    raw_proposals = proposals_from_assessment(assessment)
    if raw_proposals:
        # Validate BEFORE storing/halting. An off-thesis proposal (no news/macro
        # event clause, too many clauses, a banned indicator) is discarded here
        # rather than parked as pending: halting the walk for a human decision
        # about a hypothesis the system will refuse to test is both a waste and,
        # before this guard, a crash at resolve time.
        prepared = []
        for raw in raw_proposals:
            ready = _prepare_proposal(raw, as_of)
            if ready is not None:
                prepared.append(ready)
        if not prepared:
            return None
        # Two proposals that fire on the same days are one hypothesis wearing two
        # hats, and would spend twice the alpha budget for one piece of
        # information. Checked on BEHAVIOUR, never on shared clauses -- the
        # intended pattern is two proposals sharing their news term.
        specs = [p["spec"] for p in prepared]
        kept, notes = filter_redundant_proposals(specs, COINS, as_of=as_of)
        for note in notes:
            print(f"[{as_of.date()}] {note}")
        prepared = [p for p in prepared if p["spec"] in kept]
        # Stored as a SET: the human approves or dismisses them together, and
        # each is then tested on its own. Splitting one idea into two testable
        # halves only helps if both halves actually get tested.
        # TWO DATES, deliberately, and conflating them cost a whole overnight run.
        # `as_of` is the DATA CUTOFF for the backtest -- point B, the compression
        # exit, five days before the replay's actual position, because the
        # hypothesis must not be tested on the confirmation window that decided
        # whether to ask. `resume_from` is where the replay CLOCK is: point C.
        #
        # They used to be the same field, and resolve_pending_test wrote it
        # straight back as the checkpoint -- which rolled the clock back five
        # days, walked forward into the same compression exit, proposed again,
        # and rolled back again. A deterministic infinite loop: ~300 near-
        # duplicate proposals for one episode, live tests dated after the
        # checkpoint that was supposedly ahead of them, and roughly 3x the
        # expected spend before it was caught.
        state.save_pending_test({
            "specs": [p["dict"] for p in prepared], "coins": COINS,
            "live_coin": live_coin, "as_of": str(as_of.date()),
            "resume_from": str(resume_from.date()),
        })
        assessment["novel_condition_specs"] = [p["dict"] for p in prepared]
        _send(judgment.format_telegram_message(as_of, event_desc, assessment),
              reply_markup=REPLAY_PROPOSAL_KEYBOARD)
        return "STOP"
    # NO MESSAGE on no_action. A "nothing to see here" notification for every
    # routine macro release and every shock -- hundreds over a full replay --
    # trains a human to stop reading the channel, which costs them the
    # notifications that DO matter (a proposal, a resolved live test, a
    # consecutive-failure alert). Still printed to the run log, so the
    # judgment is auditable; it just isn't pushed at anyone.
    print(f"[{as_of.date()}] no_action: {assessment.get('assessment', '')[:120]}")
    return None



def _is_systemic_api_failure(e: Exception) -> str | None:
    """Is this "the model returned something odd" (skip the event and carry on)
    or "we can no longer reach the API at all" (stop the whole replay)?

    The distinction is load-bearing and used to be missing. Both cases were
    caught by the same `except Exception`, which printed "skipping" and then let
    the day advance and be CHECKPOINTED AS DONE. So a replay that ran out of
    credit at, say, 2020 would walk silently through the remaining ~2,000
    simulated days doing no LLM work whatsoever, finish, and leave a checkpoint
    claiming it had reached the present -- and because the checkpoint had
    advanced, resuming later would never revisit those years. Hours of simulated
    time, quietly empty, and nothing in the final state to show it.

    Returns a human-readable reason when the failure is systemic, else None.
    """
    import anthropic

    if isinstance(e, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return "the API key was rejected"
    if isinstance(e, anthropic.APIConnectionError):
        return "could not reach the Anthropic API (network)"
    # Out of credit arrives as a 400, not a dedicated exception type, so it has
    # to be recognised from the message. Checked broadly on purpose: a false
    # positive costs a stopped replay that resumes cleanly, a false negative
    # costs the silent-empty-run above.
    text = str(getattr(e, "message", "") or e).lower()
    if any(k in text for k in ("credit balance", "insufficient", "billing", "quota", "payment")):
        return "the Anthropic account is out of credit"
    return None



# If this many events in a row fail, something is wrong with the WORLD, not with
# one model response. Message-matching alone was not enough: `_is_systemic_api_failure`
# looked for credit/billing wording and sailed straight past a 400 reading
# "`temperature` is deprecated for this model", so a real run skipped every event
# for seven simulated months while advancing and checkpointing normally. A count
# needs no foresight about what the next breaking change will say.
CONSECUTIVE_FAILURE_HALT = 8


def _halt_replay(day, reason: str, events_this_chunk: int) -> dict:
    """Stop cleanly on a systemic API failure, leaving state that resumes correctly.

    Checkpoints the day BEFORE `day`, not `day` itself. The current day was only
    partially processed -- some of its events may have been judged before the
    failure -- so marking it done would silently drop the rest. Redoing one day
    costs a handful of API calls; skipping one loses events with nothing to show
    for it.

    The alert is sent to Telegram as well as printed, because the whole point is
    that this must not be something you discover afterwards from a suspiciously
    cheap run.
    """
    resume_from = (pd.Timestamp(day) - pd.Timedelta(days=1)).date()
    state.save_checkpoint(str(resume_from), status="halted")
    msg = (f"<b>Replay stopped at {pd.Timestamp(day).date()}</b>\n\n"
           f"Reason: {escape_html(reason)}.\n\n"
           f"Nothing has been lost. The checkpoint is set to {resume_from}, so resuming "
           f"re-runs {pd.Timestamp(day).date()} from the start rather than skipping it. "
           f"Processed {events_this_chunk} event(s) before stopping.\n\n"
           f"Fix the cause, then send <b>replay continue</b> to pick up from here.")
    print(f"[replay] HALTED at {pd.Timestamp(day).date()}: {reason}. Resume point: {resume_from}")
    _send(msg)
    return {"stopped": "api_failure", "reason": reason,
            "current_date": str(resume_from), "events": events_this_chunk}


# Both the replay and the live daemon call the SAME episode definition, for the
# reason shock_zscore_series was shared before it: a second, drifted copy of
# "what counts as a compression exit" would mean the replay validates one thing
# and production tracks another.
_compression_exit = compression_exit


def _trigger_description(candidate: str) -> str:
    """What the candidate's entry condition actually tests -- passed to
    Sonnet's prune advice so it reasons from the real trigger instead of
    inventing a mechanism from the label alone (same reasoning as
    scheduler/weekly_revalidation.py's own version of this function)."""
    base = candidate.rsplit("_", 1)[0]
    if base in TRIGGER_DESCRIPTIONS:
        return TRIGGER_DESCRIPTIONS[base]
    spec_dict = state.load_dynamic_candidates().get(candidate)
    if spec_dict:
        return f"{format_spec_clauses(spec_dict)} → {spec_dict['direction']}"
    return "trigger definition not found -- treat this as missing information, do not guess at it"


def _format_live_test_opened(date, direction: str, coin: str, candidate: str, horizon: int) -> str:
    """Mirrors execution/live_testing.py's own version -- bold header
    isolated on its own line, the trigger description its own paragraph,
    the held-for duration bolded, so the message scans at a glance
    instead of reading as one dense run-on sentence.

    "no TP/SL" used to be appended here and was removed: this project never
    opens a funded position at all, so saying it of one particular test
    distinguishes nothing -- it reads as though some other test might have
    one."""
    return (f"<b>{date}</b>\n\n"
            f"<b>Live test opened -- {direction.upper()} {coin}</b>\n\n"
            f"(candidate <b>{escape_html(candidate)}</b>: {escape_html(_trigger_description(candidate))})\n\n"
            f"Held for <b>{horizon}d</b>.")


PRUNE_KEYBOARD_TEMPLATE = lambda candidate: {
    "inline_keyboard": [[
        {"text": "Keep Testing", "callback_data": f"replay_prune:keep:{candidate}"},
        {"text": "Drop from Batch", "callback_data": f"replay_prune:drop:{candidate}"},
    ]]
}


def _check_prune_decisions(as_of: pd.Timestamp, status_summary: dict) -> None:
    """The keep-or-drop review, as ONE digest per year rather than a separate
    message and a separate LLM call per candidate.

    The per-candidate version asked Sonnet for a qualitative opinion on each
    candidate due for review. Measured over a 5.5-year replay that was 665 of
    1203 calls -- 55% of everything the run spent -- to produce advice formed
    from exactly the numbers already in the message, with nothing added and no
    verification behind it. `prune_recommendation` now derives the same
    keep-or-drop call from those numbers directly, using the one piece of
    evidence the opinion could not: whether there was POWER to detect an effect,
    which separates "tested and found nothing" from "never actually asked".

    Annual rather than continuous because the decision is not urgent -- a
    candidate is re-tested every ~7 simulated days regardless -- and because a
    human reviewing thirty candidates once reads them, while a human receiving
    one message per candidate stops reading.
    """
    as_of_str = str(as_of.date())
    last = state.load_checkpoint().get("last_prune_digest")
    if last and as_of.year <= pd.Timestamp(last).year:
        return
    due = sh.candidates_due_for_prune_decision(as_of_str)
    if not due:
        return
    rows = {c: status_summary.get(c, {}) for c in due}
    first_tracked = {c: sh.load_history().get(c, {}).get("first_tracked_at") for c in due}
    digest = format_prune_digest(rows, first_tracked, as_of_str)

    # The replay runs unattended, so the recommendations are APPLIED rather than
    # left waiting for a reply that will never come -- the same treatment the
    # orchestrator gives a test proposal. This is a property of the simulation,
    # not of the system: production sends the identical digest and waits for a
    # human, because there the decision is the human's to make.
    #
    # Only "drop" is acted on. "Keep" is the default for everything, including
    # every candidate not named, so applying it would be a no-op.
    dropped = [c for c in due if prune_recommendation(rows[c])[0] == "drop"]
    for c in dropped:
        sh.drop_candidate(c)
    for c in due:
        sh.mark_asked(c, as_of_str)
    if dropped:
        digest += (f"\n\n<i>Unattended replay: the {len(dropped)} recommended drop(s) were applied "
                   f"automatically. The other {len(due) - len(dropped)} stay under test.</i>")
    else:
        digest += "\n\n<i>Unattended replay: nothing was recommended for dropping; all stay under test.</i>"
    _send(digest)
    print(f"[{as_of_str}] keep-or-drop digest: {len(due)} reviewed, {len(dropped)} dropped automatically")
    cp = state.load_checkpoint()
    cp["last_prune_digest"] = as_of_str
    state._write(state.CHECKPOINT_PATH, cp)


def _resolved_live_test_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in state.load_trade_log():
        if t["status"] == "closed":
            counts[t["candidate"]] = counts.get(t["candidate"], 0) + 1
    return counts


def _effective_milestone_count(candidate: str, prior_confirmations: int | None, live_n: int) -> int:
    """How many occurrences count toward CONFIRMING this candidate.

    An occurrence counts when it happened AFTER the hypothesis was written down.
    That is the whole rule, and it puts two things on the same footing that the
    code used to treat differently:

      * occurrences accumulated while a proposal sat PARKED, waiting for enough
        history to be testable. They are in the backtest, not the trade log, but
        they postdate the hypothesis by construction -- nothing about the
        condition could have been shaped by them. `prior_confirmations` is that
        count, computed once at registration.
      * live tests opened after registration, which are the same thing arriving
        one day at a time.

    WHAT THIS REPLACES, and why the old rule was wrong. Dynamic candidates used
    to be topped up with their FULL backtest count -- so a condition with 120
    historical occurrences reached the checkpoint with zero live evidence, on its
    first day. The justification was that Sonnet never sees this project's
    backtest results, so the look-then-test risk is weak. That is true and it is
    not the point: an occurrence from 2019 cannot confirm a hypothesis written in
    2023, however uncontaminated the model was. The distinction is not who saw
    what, it is which came first.

    Static candidates (C1/C2/C6) count live occurrences only, unchanged. They were
    derived by mining this project's own history, so none of their historical
    occurrences postdates the hypothesis -- the rule above gives them zero prior
    confirmations, which is the same answer their special case gave.
    """
    if candidate in CANDIDATE_DIRECTIONS:
        return live_n
    return int(prior_confirmations or 0) + live_n


def _check_n50_milestones(as_of: pd.Timestamp, status_summary: dict) -> None:
    """NOT one-time -- fires again every time a candidate crosses a NEW
    multiple of sh.MILESTONE_N in its own _effective_milestone_count (50,
    100, 150, ...), states plainly whether it's 'accepted' AT THIS
    CHECKPOINT (fresh each time, not a permanent badge -- it may have
    been accepted at N=50 and drifted to 'watch' by N=100, or the
    reverse, both worth logging explicitly), and asks the human whether
    to keep testing or drop it -- reusing the exact same keep/drop
    buttons _check_prune_decisions uses. This is the ONE place this
    project calls a candidate "confirmed": that word is earned (or lost)
    fresh at each checkpoint by how the candidate actually performed
    over real (or, in the replay, simulated) live occurrences (plus, for
    dynamic candidates only, a rolling backtest top-up -- see
    _effective_milestone_count), not by the instant historical backtest
    alone. Replaces an earlier one-time, purely calendar-based 2-year
    version -- see docs/case_study/methodology-decisions.md."""
    as_of_str = str(as_of.date())
    live_counts = _resolved_live_test_counts()
    priors = state.load_confirmation_priors()
    counts = {c: _effective_milestone_count(c, priors.get(c), live_counts.get(c, 0))
              for c in set(live_counts) | set(status_summary) | set(priors)}
    for candidate in sh.candidates_due_for_milestone(counts):
        n_reached = (counts.get(candidate, 0) // sh.MILESTONE_N) * sh.MILESTONE_N
        live_n = live_counts.get(candidate, 0)
        info = status_summary.get(candidate, {})
        status = info.get("status") or sh.all_latest_statuses().get(candidate, {}).get("status", "unknown")
        cleared = status == "accepted"
        is_static = candidate in CANDIDATE_DIRECTIONS
        if info.get("n") is not None:
            n, sig, p = info["n"], info.get("pattern_significant"), info.get("pattern_p_value")
            criteria = [f"backtest N={n} ({'meets' if n > 50 else 'below'} the minimum of {sh.MILESTONE_N})"]
            if sig is not None:
                criteria.append(f"pattern significance: {'significant' if sig else 'not significant'}" +
                                 (f" (p={p:.3f})" if p is not None else ""))
            criteria_str = "; ".join(criteria)
        else:
            criteria_str = "no current backtest data"
        trigger_desc = _trigger_description(candidate)
        # Same reasoning as the keep-or-drop digest: the model's opinion here was
        # formed from the numbers already in this message, with nothing added.
        # prune_recommendation derives the call from them directly and can use the
        # one thing the opinion could not -- whether there was POWER to detect an
        # effect -- so a "no" means something different from "we could not tell".
        _verdict, advice = prune_recommendation(info)
        # "The trend happened" and "the trend happened BECAUSE of this condition"
        # are different claims. Across 2017-2026 a long-only rule is right most of
        # the time for reasons unrelated to any macro release, so the checkpoint --
        # unlike the per-occurrence message, which stays deliberately short --
        # carries the rate net of what simply holding the market did.
        _closed = [t for t in state.load_trade_log()
                   if t["candidate"] == candidate and t["status"] == "closed"]
        _adj = [t["forward_return"] - t["baseline_return"] for t in _closed
                if isinstance(t.get("baseline_return"), (int, float))
                and t["baseline_return"] == t["baseline_return"]]
        market_line = ""
        if _adj:
            raw_w = sum(1 for t in _closed if t["forward_return"] > 0) / len(_closed)
            adj_w = sum(1 for r in _adj if r > 0) / len(_adj)
            market_line = (f"Trend materialised {raw_w:.0%} of the time; {adj_w:.0%} after subtracting "
                           f"what holding the whole coin universe did over the same windows.\n")
        count_basis = (f"{live_n} live occurrence(s) so far" if is_static else
                       f"{n_reached} recent occurrence(s) so far ({live_n} live, the rest backtest -- static "
                       f"candidates count live occurrences only; this one is Sonnet-proposed, so backtest tops "
                       f"up the count only until it has 50 live occurrences of its own)")
        message = (
            f"<b>{as_of_str}</b>\n\n"
            f"<b>Confirmation checkpoint at {n_reached} occurrences -- {escape_html(candidate)}</b>\n\n"
            f"({escape_html(trigger_desc)})\n\n"
            f"<b>{'CONFIRMED' if cleared else 'NOT confirmed'}</b> at this checkpoint -- "
            f"{'still clears' if cleared else 'no longer clears'} the acceptance bar ({count_basis}).\n"
            f"<i>Confirmed, not validated: at this project's horizons a conclusive test would need "
            f"occurrences in the hundreds (see each live-test message for the number). This says the "
            f"condition has kept occurring and still passes on the enlarged sample -- persistence, "
            f"not proof.</i>\n"
            f"{escape_html(criteria_str)}. (No single coin or period may carry more than 60% of the positive "
            f"return either, for either check to pass.)\n\n"
            f"{market_line}"
            f"Current status: <b>{escape_html(status)}</b>\n"
            f"Re-evaluated fresh at every {sh.MILESTONE_N}-occurrence checkpoint, not a permanent verdict -- re-checked "
            f"again at {n_reached + sh.MILESTONE_N} either way, unless dropped below.\n\n"
            f"{escape_html(hyperopt_runner.format_result(candidate))}\n\n"
            f"Assessment: {escape_html(advice)}"
        )
        _send(message, reply_markup=PRUNE_KEYBOARD_TEMPLATE(candidate))
        sh.mark_milestone_reported(candidate, n_reached, cleared)


def _dynamic_trigger_hourly(spec: ConditionSpec, hourly: pd.DataFrame, daily: pd.DataFrame, funding,
                             symbol: str | None = None) -> "pd.Series":
    """Same AND-of-clauses logic as novel_condition_tester.test_novel_condition,
    but evaluated hour-by-hour for live detection (scale=24 reinterprets
    every day-defined indicator window in hours -- see
    docs/case_study/methodology-decisions.md). `shock_zscore` is no
    longer the ONLY such exception: `DAILY_NATIVE_INDICATORS` now covers
    every indicator that isn't distributionally comparable at scale=24
    (rsi_14d, atr_pct_14d, daily_range_pct, efficiency_ratio_20d too) --
    a real, measured train/serve skew documented at that constant.

    Delegates every clause to `clause_signal_hourly`, the ONE shared
    implementation execution/live_testing.py also uses -- so a sequenced
    or daily-native condition can never mean one thing in the backtest
    that accepted it and another in the scan that tracks it."""
    trigger = pd.Series(True, index=hourly.index)
    for clause in spec.clauses:
        trigger &= clause_signal_hourly(clause, hourly, daily, funding, symbol=symbol)
    return trigger


def _static_triggers_full(hourly_full: dict) -> dict:
    """Precomputes the hourly static-candidate trigger series over each
    coin's ENTIRE available history, once per chunk -- NOT per simulated
    day. Rolling-window indicators (e.g. a 480-hour efficiency ratio)
    need real lookback history to mean anything; slicing to a single
    day's 24 rows BEFORE computing them starves every rolling window of
    its lookback and silently produces nothing but NaN/False (a real bug
    caught by testing this against real data before relying on it). Each
    row's value only ever depends on bars up to and including it (same
    backward-looking guarantee as shock_zscore_series), so precomputing
    over the full series and reading historical rows out of the result
    afterward is exactly as causally safe as computing it fresh each day
    -- just not redundantly expensive."""
    return {coin: compute_triggers(hourly_full[coin], load_funding(coin), scale=24) for coin in COINS}


def _scan_mechanical_triggers(d: pd.Timestamp, hourly_full: dict, ohlc_full: dict, static_triggers_full: dict) -> None:
    """Unattended, no LLM involved -- for every TRACKED candidate (static
    C1/C2/C6 + dynamic, not dropped -- accepted, watch, rejected, or not
    yet classified all included, since testing starts the moment a
    trigger is identified, not only once it's accepted), checks whether
    its own trigger fired on any HOURLY bar within simulated day `d`, for
    every coin, and opens a live test the moment it does. Detection is
    hourly; opening/resolving the live test still uses the existing
    daily-bar machinery (horizon counted in days) -- see
    docs/case_study/methodology-decisions.md for why the two run at
    different granularities. Skips a (candidate, coin) pair that already
    has an open live test, so a persistent multi-hour condition doesn't
    reopen one every hour it stays true."""
    day_start, day_end = d.normalize(), d.normalize() + pd.Timedelta(hours=23)
    open_pairs = {(t["candidate"], t["coin"]) for t in state.load_open_trades()}
    dynamic_registry = state.load_dynamic_candidates()

    for coin in COINS:
        funding = load_funding(coin)
        today_static = static_triggers_full[coin].loc[day_start:day_end]
        if len(today_static):
            for variant, direction in CANDIDATE_DIRECTIONS.items():
                if sh.is_dropped(variant) or (variant, coin) in open_pairs or variant not in today_static.columns:
                    continue
                if not today_static[variant].any():
                    continue
                execution = _open_live_test(variant, coin, direction, d)
                if execution.get("opened"):
                    _send(_format_live_test_opened(d.date(), direction, coin, variant, execution["horizon"]))

        if not dynamic_registry:
            continue
        # Dynamic conditions are recomputed with full lookback (through `d`)
        # each day, unlike the static precompute above -- there are
        # typically few of them, and the registry itself can grow mid-chunk
        # (a new "test it" approval), so a chunk-level precompute can't
        # cover a condition that didn't exist yet when the chunk started.
        hourly_to_date = hourly_full[coin].loc[:day_end]
        for label, spec_dict in dynamic_registry.items():
            if sh.is_dropped(label) or (label, coin) in open_pairs:
                continue
            try:
                spec = spec_from_dict(spec_dict)
            except ValueError:
                # Recorded before a news/macro event clause became a NECESSARY
                # condition -- skip rather than crash, see
                # llm_pipeline/dynamic_candidates.py::registered_specs.
                continue
            trig = _dynamic_trigger_hourly(spec, hourly_to_date, ohlc_full[coin], funding, symbol=coin).loc[day_start:day_end]
            if not trig.any():
                continue
            execution = _open_live_test(label, coin, spec.direction, d)
            if execution.get("opened"):
                _send(_format_live_test_opened(d.date(), spec.direction, coin, label, execution["horizon"]))


def advance(chunk_days: int = CHUNK_DAYS) -> dict:
    checkpoint = state.load_checkpoint()
    if checkpoint["status"] == "waiting_for_human":
        return {"stopped": "waiting_for_human", "current_date": checkpoint["current_date"]}

    load_dotenv()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    if checkpoint["current_date"] is None:
        # Start at the earliest date ANY coin in the universe has data for
        # (not an arbitrary N-years-back offset) -- pattern_significance
        # needs as many yearly folds as it can get for a robust horizon
        # choice and a real shot at each trigger reaching its own N=50
        # live-test checkpoint. Coins with a later start (e.g. DOGE, 2019)
        # simply have fewer years pooled early on -- build_events/
        # walk_forward already handle that per-coin, nothing special
        # needed here. The first several simulated years will mostly show
        # 'insufficient_data' until enough yearly folds accumulate --
        # expected, not a bug.
        current = min(load_daily(c).index.min() for c in COINS)
        run_replay_battery(current)
        # Saved immediately, not just at the end of the chunk -- a crash
        # partway through this first chunk must never leave battery_status.json
        # seeded while the checkpoint still says "not started", which would
        # make answer_market_question() read as self-contradictory.
        state.save_checkpoint(str(current.date()), status="running")
    else:
        current = pd.Timestamp(checkpoint["current_date"])

    today_real = pd.Timestamp.now().normalize()
    chunk_end = min(current + pd.Timedelta(days=chunk_days), today_real)
    ohlc_full = {c: load_daily(c) for c in COINS}  # loaded once per chunk, sliced in-memory below -- not re-read per day/coin
    hourly_full = {c: load_hourly(c) for c in COINS}  # for mechanical trigger detection only -- see _scan_mechanical_triggers
    static_triggers_full = _static_triggers_full(hourly_full)  # precomputed once per chunk, not per simulated day

    events_this_chunk = 0
    consecutive_failures = 0
    last_battery_refresh = current
    d = current + pd.Timedelta(days=1)
    while d <= chunk_end:
        _check_live_tests(d)  # every day, not just event days -- a live test can resolve on any day
        _scan_mechanical_triggers(d, hourly_full, ohlc_full, static_triggers_full)  # unattended, no LLM -- opens a live test the moment any tracked trigger fires
        for message in state.due_reveals(d):  # a held-back test result surfacing on its own scheduled day
            _send(message)

        if (d - last_battery_refresh).days >= 7:
            status_summary = run_replay_battery(d)
            _check_prune_decisions(d, status_summary)
            _check_n50_milestones(d, status_summary)
            for candidate, info in status_summary.items():
                if "horizon_changed_to" in info:
                    _send(f"<b>{d.date()}</b>\n\n"
                          f"<b>Horizon updated -- {escape_html(candidate)}</b>\n\n"
                          f"({escape_html(_trigger_description(candidate))})\n\n"
                          f"Now held for <b>{info['horizon_changed_to']}d</b> going forward (empirically "
                          f"re-derived from accumulated history, replacing the previous value).")
            last_battery_refresh = d
            # A parked proposal whose history has caught up goes through the same
            # human gate a fresh one would -- it was never shown to anyone, since
            # it was refused before it could be.
            if state.load_pending_test() is None:
                promoted = _check_parked_proposals(d)
                if promoted is not None:
                    state.save_pending_test(promoted)
                    _send(judgment.format_telegram_message(
                        d, f"A hypothesis proposed on {promoted['proposed_at']} now has enough "
                           f"history behind it to be tested.",
                        {"assessment": "Parked when first proposed because it had not occurred often "
                                        "enough to measure. It has now.",
                         "recommended_action": "propose_novel_test",
                         "novel_condition_specs": promoted["specs"]}),
                        reply_markup=REPLAY_PROPOSAL_KEYBOARD)
                    state.save_checkpoint(str(d.date()), status="waiting_for_human")
                    return {"stopped": "waiting_for_human", "current_date": str(d.date()),
                            "events": events_this_chunk}

        # THE ONLY TRIGGER. Macro releases and volatility shocks were both
        # removed, on the same principle and each with its own measurement:
        #
        #   * a macro release is one of the CAUSES being sought, so triggering on
        #     it conditions the search on the thing to be explained -- and
        #     measured, release days are statistically indistinguishable from
        #     ordinary days at every horizon and for all three series
        #     (forecast/trigger_value.py);
        #   * a volatility shock is the OUTCOME, not a trend, and it actively
        #     selects against what this project looks for: post-shock days
        #     produce a defined trend 8.8% of the time against an 11.8% baseline.
        #
        # Compression is a precursor instead (16.1% vs 11.1%), and it is the
        # right shape besides: it says a directional move is brewing WITHOUT
        # saying which way, leaving the direction to be explained by the macro
        # context and market state -- the question this pipeline exists to ask.
        for coin in COINS:
            episode = _compression_exit(ohlc_full[coin], d)
            if episode is None:
                continue
            event_desc = judgment.format_compression_event(coin, episode)
            # `as_of` is point B, not today. The confirmation window decided
            # WHETHER to ask; it must not leak into WHAT is shown, and the
            # backtest must not see days the hypothesis is meant to precede.
            as_of_b = episode["b_date"]
            try:
                assessment = judgment.judge_event(event_desc, client, as_of=as_of_b, coin=coin)
            except Exception as e:
                halt = _is_systemic_api_failure(e)
                consecutive_failures += 1
                if not halt and consecutive_failures >= CONSECUTIVE_FAILURE_HALT:
                    halt = (f"{consecutive_failures} events failed in a row -- last error: {e}")
                if halt:
                    return _halt_replay(d, halt, events_this_chunk)
                print(f"Failed to judge event on {d.date()} ({coin} compression), skipping: {e}")
                continue
            consecutive_failures = 0
            events_this_chunk += 1
            if _handle_assessment(as_of_b, event_desc, assessment, live_coin=coin,
                                   resume_from=d) == "STOP":
                state.save_checkpoint(str(d.date()), status="waiting_for_human")
                return {"stopped": "waiting_for_human", "current_date": str(d.date()), "events": events_this_chunk}

        # Saved after EVERY day, not just at the end of the whole chunk --
        # a crash on day N+1 must not force day N's already-sent messages
        # (a reveal, a no_action assessment, whatever fired) to be
        # reprocessed and re-notified when the chunk is retried.
        state.save_checkpoint(str(d.date()), status="running")
        d += pd.Timedelta(days=1)

    state.save_checkpoint(str(chunk_end.date()), status="running")
    reached_end = chunk_end >= today_real
    _send_checkpoint_digest(current, chunk_end, events_this_chunk, reached_end)
    return {"stopped": None, "current_date": str(chunk_end.date()), "events": events_this_chunk, "reached_end": reached_end}


def _send_checkpoint_digest(start: pd.Timestamp, end: pd.Timestamp, n_events: int, reached_end: bool) -> None:
    log = state.load_trade_log()
    opened_this_period = [t for t in log if start.date() <= pd.Timestamp(t["decision_date"]).date() <= end.date()]
    closed_this_period = [t for t in log if t["status"] == "closed"
                           and start.date() <= pd.Timestamp(t["close_date"]).date() <= end.date()]
    positive = sum(1 for t in closed_this_period if t["forward_return"] > 0)
    mean_return = (sum(t["forward_return"] for t in closed_this_period) / len(closed_this_period)) if closed_this_period else 0.0
    still_open = len(state.load_open_trades())

    all_closed = [t for t in log if t["status"] == "closed"]
    all_positive = sum(1 for t in all_closed if t["forward_return"] > 0)
    all_mean_return = (sum(t["forward_return"] for t in all_closed) / len(all_closed)) if all_closed else 0.0

    statuses = sh.all_latest_statuses()
    active = {name: info for name, info in statuses.items() if not info["dropped"]}
    lines = [
        f"<b>Checkpoint</b> {start.date()} -> {end.date()}",
        "",
        f"Events assessed this period: {n_events}",
        f"Live tests opened this period: {len(opened_this_period)}",
        f"Live tests resolved this period: {len(closed_this_period)} ({positive} positive forward return), mean return: {mean_return:+.2%}",
        f"All-time: {len(all_closed)} resolved ({all_positive} positive), mean return: {all_mean_return:+.2%}",
        f"Still open (awaiting resolution): {still_open}",
        f"Total live tests ever opened: {len(log)}",
        "",
        f"Triggers tracked: {len(statuses)} total, {len(active)} still active.",
    ]
    for name, info in sorted(statuses.items()):
        if info["dropped"]:
            tag = "dropped"
        elif info.get("milestone_reported"):
            n_reached = info.get("last_checkpoint_n", sh.MILESTONE_N)
            tag = "confirmed" if info.get("milestone_cleared") else f"not confirmed at its {n_reached}-occurrence checkpoint"
        elif info["status"] == "accepted":
            tag = f"accepted, no confirmation checkpoint yet (first at {sh.MILESTONE_N} post-hypothesis occurrences)"
        else:
            tag = "in progress"
        lines.append(f"  - <b>{escape_html(name)}</b> ({tag}): {escape_html(_trigger_description(name))}")
    if reached_end:
        lines.append("\n<b>Replay has reached the present -- fully caught up.</b>")
    _send("\n".join(lines))


TEST_RESULT_DELAY_DAYS = 3


def resolve_pending_test() -> str | None:
    """Called when the human presses the "Test It" button (never a
    free-text reply) -- acknowledges
    immediately (the backtest itself is computed right away too, since
    it's free -- querying history that already exists, nothing to wait
    for), but the RESULT is queued for `TEST_RESULT_DELAY_DAYS` later and
    only actually sent once the day-by-day walk reaches that date
    (`due_reveals`, checked in `advance()`'s own loop) -- so it surfaces
    interleaved with whatever else happens around that day, not as the
    very first thing the next time simulated time moves at all. State
    changes (the dynamic-candidate registry, an accepted candidate
    joining the battery, a trade opening) are still recorded as of the
    real `as_of` date -- only the notification about them is deferred,
    not the events themselves."""
    pending = state.load_pending_test()
    if pending is None:
        return None
    _send("Received — this will be tested against the market.")

    # Accepts both shapes: `specs` (a proposal SET, the current form) and `spec`
    # (a single proposal, still present in stored state from earlier runs).
    raw_specs = pending.get("specs") or ([pending["spec"]] if pending.get("spec") else [])
    specs = [spec_from_dict(d) for d in raw_specs]
    as_of = pd.Timestamp(pending["as_of"])
    reveal_date = as_of + pd.Timedelta(days=TEST_RESULT_DELAY_DAYS)

    lines, statuses = [], []
    for spec in specs:
        status, block = _resolve_one_proposal(spec, pending, as_of, reveal_date)
        statuses.append(status)
        lines += block + [""]

    # The informational appendix -- see _cooccurrence_appendix. Never a verdict,
    # never a third candidate: a note attached to the two real results.
    lines += _cooccurrence_appendix(specs, pending["coins"], as_of)

    state.save_pending_test(None)
    # The CLOCK, not the data cutoff -- see save_pending_test above for what
    # writing `as_of` here cost. Falls back to `as_of` for a pending entry
    # written before the two were separated.
    state.save_checkpoint(pending.get("resume_from") or pending["as_of"], status="running")
    state.queue_reveal("\n".join(lines), str(reveal_date.date()))
    return statuses[0] if statuses else None


def _cooccurrence_appendix(specs: list, coins: list[str], as_of: pd.Timestamp) -> list[str]:
    """A note on how often the proposals in a set have historically fired
    together. NOT a result, NOT a candidate, and never confirmed.

    What this is for. A future reader deciding how to use these triggers wants to
    know whether they are two views of one situation or two independent ones, and
    whether their conjunction is even a thing that happens. That is genuinely
    useful and it is not a hypothesis test.

    Three rules keep it from becoming one, and each closes a specific failure:

      * IT REPORTS A COUNT, never an outcome statistic. "These co-occurred 12
        times" is a fact about frequency; "these won 8 times out of 10" is a
        performance claim on a sample far too small to make one, and would be
        read as evidence however it were captioned.
      * IT LISTS EVERY PAIR that has ever co-fired, not the interesting ones.
        Reporting only the promising combination shows the tail of a distribution
        without showing the distribution -- and with no number attached, a reader
        cannot even suspect the selection.
      * IT NEVER OPENS A TEST. The conjunction is a three-clause condition in all
        but name, with a measured median of 12 occurrences; a live test on it
        could not confirm anything and would sit in the trade log looking like
        one that could.

    Computed over the FULL history rather than over accumulated live
    co-occurrences, which is what makes the count meaningful immediately instead
    of in several years."""
    if len(specs) < 2:
        return []
    out = ["<i>Appendix -- how these two have historically occurred together. "
           "Informational only: no test is opened on the combination, and nothing "
           "here is confirmed or counts toward confirmation.</i>"]
    for a, b in itertools.combinations(specs, 2):
        together = ConditionSpec(label=f"{a.label}+{b.label}", direction=a.direction,
                                  clauses=tuple(a.clauses) + tuple(b.clauses))
        n_both = count_occurrences(together, coins, as_of=as_of)
        n_a = count_occurrences(a, coins, as_of=as_of)
        n_b = count_occurrences(b, coins, as_of=as_of)
        if n_both == 0:
            out.append(f"• {escape_html(a.label)} and {escape_html(b.label)} have "
                       f"<b>never</b> occurred on the same day ({n_a} and {n_b} times "
                       f"separately). They describe different situations.")
            continue
        out.append(f"• {escape_html(a.label)} and {escape_html(b.label)} have occurred "
                   f"together <b>{n_both}</b> time(s), against {n_a} and {n_b} "
                   f"separately.")
        if n_both < MIN_HISTORICAL_OCCURRENCES:
            out.append(f"  Too few to say anything about the combination -- "
                       f"{MIN_HISTORICAL_OCCURRENCES} occurrences are the minimum this "
                       f"project will draw any conclusion from, and that is why no test "
                       f"is opened on it.")
    return out


def _resolve_one_proposal(spec: "ConditionSpec", pending: dict, as_of: pd.Timestamp,
                           reveal_date: pd.Timestamp) -> "tuple[str, list[str]]":
    """Backtest one proposal from an approved set, record it, and return its
    status plus the message block describing it.

    Split out of `resolve_pending_test` when a call started returning two
    proposals: every proposal gets the identical treatment, and the alternative
    -- a loop body inlined in a function that also does set-level work -- is how
    the two halves drift into being tested differently."""
    result = test_novel_condition(spec, pending["coins"], as_of=as_of)
    status = result["status"]

    # Occurrences that already postdate the hypothesis at registration time. For a
    # proposal that sat parked this is the whole parking period; for one testable
    # immediately it is zero, which is correct -- it has nothing to be confirmed by
    # yet.
    proposed_at_for_prior = (pending.get("proposed_at")
                             or next((d.get("parked_since") for d in (pending.get("specs") or [])
                                      if d.get("label") == spec.label and d.get("parked_since")), None))
    if proposed_at_for_prior:
        prior = prospective_split(spec, pending["coins"], proposed_at_for_prior, as_of=as_of)["n_after"]
    else:
        prior = 0
    state.save_confirmation_prior(spec.label, prior)

    registry = state.load_dynamic_candidates()
    registry[spec.label] = {"label": spec.label,
                             "clauses": [clause_to_dict(c) for c in spec.clauses],
                             "direction": spec.direction, "horizons": list(spec.horizons)}
    state.save_dynamic_candidates(registry)
    # Recorded here, not deferred to the next weekly battery refresh -- without
    # this, a just-discovered candidate is already registered (and can already
    # open live tests via the mechanical scan, which doesn't check status_history
    # at all) but stays invisible to all_latest_statuses()/Sonnet's own context
    # for up to ~7 simulated days.
    sh.record_status(spec.label, status, pending["as_of"])

    condition_str = f"{condition_desc(spec)} → {spec.direction}"
    lines = [f"<b>{reveal_date.date()}</b>", "",
             f"<b>Historical backtest -- {escape_html(spec.label)}</b>", "",
             f"({escape_html(condition_str)})"]
    if pending.get("relaxed_from") or any(d.get("relaxed_from") for d in (pending.get("specs") or [])
                                           if d.get("label") == spec.label):
        note = next((d.get("relaxed_from") for d in (pending.get("specs") or [])
                     if d.get("label") == spec.label and d.get("relaxed_from")),
                    pending.get("relaxed_from"))
        if note:
            lines += ["", f"<i>Thresholds were {escape_html(note)}.</i>"]
    pattern = result.get("pattern_significance") or {}
    if status == "insufficient_data":
        n_raw = result.get("n_raw_triggers", 0)
        lines += ["", f"<b>Verdict:</b> not enough history to judge -- "
                      f"{n_raw} occurrence(s) found, too few for a walk-forward test.",
                  "", "Re-checked automatically every 7 days; this can change as "
                      "more occurrences accumulate."]
    else:
        lines.append("")
        lines.append(format_pattern_significance(pattern))
        lines.append("")
        lines.append(f"<b>Verdict:</b> {escape_html(status.upper())}")
        lines.append("")
        lines.append(f"For reference, trading this with a TP/SL structure over the same history: "
                     f"N={result['n']}, win_rate={result['win_rate']:.1%}, sortino={result['sortino']:.2f} "
                     f"(informational only, doesn't affect the verdict above).")
        lines.append("Testing continues going forward regardless of this verdict -- re-checked automatically "
                      "every 7 days alongside every other tracked trigger.")

    # The genuinely prospective half, when the hypothesis predates part of its own
    # evidence. A parked proposal has this by construction; a freshly-proposed one
    # has none, and says so rather than showing an empty section.
    proposed_at = (pending.get("proposed_at")
                   or next((d.get("parked_since") for d in (pending.get("specs") or [])
                            if d.get("label") == spec.label and d.get("parked_since")), None))
    if proposed_at and status != "insufficient_data":
        split = prospective_split(spec, pending["coins"], proposed_at, as_of=as_of,
                                   horizon=(pattern.get("horizon") or PLACEHOLDER_HORIZON_DAYS))
        if split["n_after"]:
            lines += ["", f"<b>Out-of-sample since the hypothesis was written ({split['proposed_at']}):</b>",
                      f"{split['n_after']} of {split['n_before'] + split['n_after']} occurrences happened "
                      f"AFTER this condition was written down, so nothing about it could have been "
                      f"shaped by them.",
                      f"Mean forward return on those: {split['mean_after']:+.2%} against "
                      f"{split['baseline_after']:+.2%} for simply holding over the same span "
                      f"(excess {split['excess_after']:+.2%}).",
                      f"<i>Reported, not gated: at this count a significance test would usually be "
                      f"underpowered, and a test that cannot detect anything must not be read as a "
                      f"negative result.</i>"]

    if pattern.get("status") == "ok":
        horizons = state.load_horizons()
        horizons[spec.label] = pattern["horizon"]
        state.save_horizons(horizons)
        sh.record_horizon(spec.label, pattern["horizon"])

    if status == "accepted" and result.get("live_anchors"):
        battery = state.load_battery_status()
        battery["candidates"][spec.label] = {
            "direction": spec.direction, "horizon": pattern["horizon"],
            "tp_mult": result["live_tp_mult"], "sl_mult": result["live_sl_mult"],
            "anchors": result["live_anchors"],
        }
        state.save_battery_status(battery)
        lines.append("")
        lines.append("Now accepted into the battery -- its own trigger will fire live tests automatically going "
                     "forward, alongside every other accepted candidate.")

    # The occurrence that prompted this proposal gets its own live test regardless
    # of the verdict -- testing starts the moment a trigger is identified, not
    # only once it's already accepted.
    if pending.get("live_coin") and status != "insufficient_data":
        # Dated to the proposal's own `as_of` (point B), which for a compression
        # trigger is COMPRESSION_CONFIRM_DAYS before the replay's actual position.
        # Backdating is deliberate: the outcome is read at the end of the horizon,
        # never before, and the model never saw the confirmation window. The one
        # case where that would not hold is a horizon shorter than the
        # confirmation window, since the whole outcome would already be history;
        # those start at the confirmation date instead.
        horizon = int(state.load_horizons().get(spec.label, PLACEHOLDER_HORIZON_DAYS))
        open_at = as_of
        if horizon <= COMPRESSION_CONFIRM_DAYS:
            open_at = as_of + pd.Timedelta(days=COMPRESSION_CONFIRM_DAYS)
        execution = _open_live_test(spec.label, pending["live_coin"], spec.direction, open_at)
        if execution.get("opened"):
            lines.append("")
            lines.append(f"<b>Live test opened -- {spec.direction.upper()} {escape_html(pending['live_coin'])}</b>\n\n"
                         f"Held for <b>{execution['horizon']}d</b>, then resolved -- measuring the same pattern the backtest found.")
    return status, lines


def discard_pending_test() -> str | None:
    """Called when the human presses "Don't Test It" -- clears the
    pending proposal without ever running the backtest, and lets the
    replay resume advancing (mirrors resolve_pending_test's own
    checkpoint handling, minus everything test-related). Returns None if
    nothing was actually pending (e.g. a double-tap after it already
    expired)."""
    pending = state.load_pending_test()
    if pending is None:
        return None
    state.save_pending_test(None)
    state.save_checkpoint(pending.get("resume_from") or pending["as_of"], status="running")
    _send("Dismissed -- this condition won't be tested. Reply 'replay continue' to keep going.")
    return "dismissed"
