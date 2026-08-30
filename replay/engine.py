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

import os

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

from candidates.data_loading import load_daily, load_funding, load_hourly
from candidates.definitions import CANDIDATE_DIRECTIONS, TRIGGER_DESCRIPTIONS, compute_triggers
from candidates.macro_vintage import MACRO_SERIES
from candidates.methodology import explain_non_acceptance, path_outcome, shock_zscore_series
from candidates.run_battery import COINS
from execution import hyperopt_runner
from llm_pipeline.haiku_sonnet_pipeline import escape_html, format_spec_clauses, sonnet_prune_advice
from llm_pipeline.novel_condition_tester import (
    ConditionSpec, clause_from_dict, clause_signal_hourly, clause_to_dict, condition_desc, format_pattern_significance,
    spec_from_proposal,
    test_novel_condition,
)
from replay import judgment, state
from replay import status_history as sh
from replay.battery import run_replay_battery
from replay.time_sandbox import latest_release_with_prior, release_dates
from telegram.bot import _send

CHUNK_DAYS = 30
SHOCK_ZSCORE_THRESHOLD = 2.0


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
    immediately after a live test resolves, only for a VALIDATED
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
        f"The last <b>{streak}</b> live test(s) for this VALIDATED candidate resolved negative in a row.\n",
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
        })
        _send(f"<b>{d.date()}</b>\n\n"
              f"<b>Live test resolved -- {trade['direction'].upper()} {trade['coin']}</b>\n\n"
              f"(candidate <b>{escape_html(trade['candidate'])}</b>: {escape_html(_trigger_description(trade['candidate']))}, "
              f"held {trade['horizon']}d, opened {trade['entry_date']})\n\n"
              f"Forward return: <b>{outcome['forward_return']:+.2%}</b>\n"
              f"Best point reached: {outcome['mfe']:+.2%}\n"
              f"Worst point reached: {outcome['mae']:+.2%}")
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


def _handle_assessment(as_of: pd.Timestamp, event_desc: str, assessment: dict, live_coin: str | None = None) -> str | None:
    """Returns "STOP" if the replay must halt for a human decision, else
    None. Sonnet never opens a trade here -- that's the mechanical scan's
    job (see _scan_mechanical_triggers); this only ever proposes a novel
    condition for human approval, or does nothing."""
    if assessment["recommended_action"] == "propose_novel_test" and assessment.get("novel_condition_spec"):
        s = assessment["novel_condition_spec"]
        # Validate BEFORE storing/halting. An off-thesis proposal (no news/macro
        # event clause -- see novel_condition_tester.NEWS_EVENT_INDICATORS) is
        # discarded here rather than parked as pending: halting the walk for a
        # human decision about a hypothesis the system will refuse to test is
        # both a waste and, before this guard, a crash at resolve time.
        _spec, _err = spec_from_proposal(s)
        if _spec is None:
            print(f"[{as_of.date()}] proposal '{s.get('label')}' rejected, not tested: {_err}")
            return None
        state.save_pending_test({"spec": s, "coins": COINS, "live_coin": live_coin, "as_of": str(as_of.date())})
        _send(judgment.format_telegram_message(as_of, event_desc, assessment), reply_markup=REPLAY_PROPOSAL_KEYBOARD)
        return "STOP"
    # NO MESSAGE on no_action. A "nothing to see here" notification for every
    # routine macro release and every shock -- hundreds over a full replay --
    # trains a human to stop reading the channel, which costs them the
    # notifications that DO matter (a proposal, a resolved live test, a
    # consecutive-failure alert). Still printed to the run log, so the
    # judgment is auditable; it just isn't pushed at anyone.
    print(f"[{as_of.date()}] no_action: {assessment.get('assessment', '')[:120]}")
    return None


def _shock_transition(ohlc_full: pd.DataFrame, day: pd.Timestamp) -> tuple[float, str] | None:
    """Fires only on the transition INTO shock regime, not every day a
    multi-day shock persists -- checks `day` and `day - 1` against the
    same in-memory frame, sliced, never re-read from disk per call."""
    today_slice = ohlc_full.loc[:day]
    if len(today_slice) == 0 or today_slice.index[-1] != day:
        return None
    z_today = shock_zscore_series(today_slice).iloc[-1]
    if pd.isna(z_today) or z_today < SHOCK_ZSCORE_THRESHOLD:
        return None
    yesterday = day - pd.Timedelta(days=1)
    yesterday_slice = ohlc_full.loc[:yesterday]
    if len(yesterday_slice):
        z_yesterday = shock_zscore_series(yesterday_slice).iloc[-1]
        if pd.notna(z_yesterday) and z_yesterday >= SHOCK_ZSCORE_THRESHOLD:
            return None  # already in shock yesterday -- don't re-fire every day
    latest_return = today_slice["close"].pct_change().iloc[-1]
    direction = "crash" if latest_return < 0 else "surge"
    return float(z_today), direction


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
    instead of reading as one dense run-on sentence."""
    return (f"<b>{date}</b>\n\n"
            f"<b>Live test opened -- {direction.upper()} {coin}</b>\n\n"
            f"(candidate <b>{escape_html(candidate)}</b>: {escape_html(_trigger_description(candidate))})\n\n"
            f"Held for <b>{horizon}d</b>, no TP/SL.")


PRUNE_KEYBOARD_TEMPLATE = lambda candidate: {
    "inline_keyboard": [[
        {"text": "Keep Testing", "callback_data": f"replay_prune:keep:{candidate}"},
        {"text": "Drop from Batch", "callback_data": f"replay_prune:drop:{candidate}"},
    ]]
}


def _check_prune_decisions(as_of: pd.Timestamp, status_summary: dict, client: Anthropic) -> None:
    """Same mechanism as scheduler/weekly_revalidation.py's real one --
    a candidate tracked 2 simulated years without ever being accepted
    gets a keep-or-drop proposal, Sonnet's advisory opinion attached, a
    human decision via Telegram buttons. Checked every battery refresh
    (~7 simulated days), same cadence the battery itself re-classifies
    status on."""
    as_of_str = str(as_of.date())
    for candidate in sh.candidates_due_for_prune_decision(as_of_str):
        info = status_summary.get(candidate, {})
        if info.get("status") == "error":
            reason = "this refresh failed to process this candidate -- its long-term status is unavailable this time"
            summary = "no current data (processing error)"
        elif info.get("n") is not None:
            reason = explain_non_acceptance(info)
            summary = (f"win_rate={info['win_rate']:.1%}, sortino={info['sortino']:.2f}, N={info['n']} "
                       f"(reference TP/SL-structure stats, informational only, don't gate acceptance)")
        else:
            reason = "no current data" if info.get("status") != "insufficient_data" else explain_non_acceptance({"n": 0})
            summary = "no current data" if info.get("status") != "insufficient_data" else "not enough historical occurrences yet to compute reference stats"
        trigger_desc = _trigger_description(candidate)
        advice = sonnet_prune_advice(
            candidate, years_tracked=sh.years_tracked(candidate, as_of_str) or sh.PRUNE_YEARS_THRESHOLD,
            recent_summary=summary, trigger_description=trigger_desc, client=client,
        )
        message = (
            f"<b>{as_of_str}</b>\n\n"
            f"<b>Keep-or-drop decision -- {escape_html(candidate)}</b>\n\n"
            f"({escape_html(trigger_desc)})\n\n"
            f"Tracked 2+ years, never reached 'accepted' status.\n"
            f"<b>Why:</b> {escape_html(reason)}.\n\n"
            f"Will keep being re-tested automatically every ~7 days either way, unless dropped below.\n"
            f"For reference: {escape_html(summary)}\n\n"
            f"Sonnet's opinion (advisory only, not a verified finding): {escape_html(advice)}"
        )
        _send(message, reply_markup=PRUNE_KEYBOARD_TEMPLATE(candidate))
        sh.mark_asked(candidate, as_of_str)


def _resolved_live_test_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in state.load_trade_log():
        if t["status"] == "closed":
            counts[t["candidate"]] = counts.get(t["candidate"], 0) + 1
    return counts


def _effective_milestone_count(candidate: str, backtest_n: int | None, live_n: int) -> int:
    """Mirrors execution/live_testing.py::_effective_milestone_count
    exactly. Static candidates (C1/C2/C6) were derived by directly mining
    this project's own historical data -- a direct look-then-test risk,
    so only genuinely prospective evidence (real, or here simulated,
    resolved live tests) counts toward validating them. Dynamic
    (Sonnet-proposed) candidates carry only a much weaker, diffuse
    version of that risk (Sonnet never sees this project's own backtest
    results before proposing), so they use a rolling window of the most
    recent 50 occurrences, backtest and live mixed -- equivalent to
    filling the window with live occurrences first and topping up with
    the most recent backtest ones only while live_n hasn't reached 50
    yet. Once a dynamic candidate accumulates 50 live tests on its own,
    this collapses to the exact same live-only rule the static
    candidates always use."""
    if candidate in CANDIDATE_DIRECTIONS or live_n >= sh.MILESTONE_N:
        return live_n
    return min(backtest_n or 0, sh.MILESTONE_N - live_n) + live_n


def _check_n50_milestones(as_of: pd.Timestamp, status_summary: dict, client: Anthropic) -> None:
    """NOT one-time -- fires again every time a candidate crosses a NEW
    multiple of sh.MILESTONE_N in its own _effective_milestone_count (50,
    100, 150, ...), states plainly whether it's 'accepted' AT THIS
    CHECKPOINT (fresh each time, not a permanent badge -- it may have
    been accepted at N=50 and drifted to 'watch' by N=100, or the
    reverse, both worth logging explicitly), and asks the human whether
    to keep testing or drop it -- reusing the exact same keep/drop
    buttons _check_prune_decisions uses. This is the ONE place this
    project calls a candidate "validated": that word is earned (or lost)
    fresh at each checkpoint by how the candidate actually performed
    over real (or, in the replay, simulated) live occurrences (plus, for
    dynamic candidates only, a rolling backtest top-up -- see
    _effective_milestone_count), not by the instant historical backtest
    alone. Replaces an earlier one-time, purely calendar-based 2-year
    version -- see docs/case_study/methodology-decisions.md."""
    as_of_str = str(as_of.date())
    live_counts = _resolved_live_test_counts()
    counts = {c: _effective_milestone_count(c, status_summary.get(c, {}).get("n"), live_counts.get(c, 0))
              for c in set(live_counts) | set(status_summary)}
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
        advice = sonnet_prune_advice(
            candidate, years_tracked=sh.years_tracked(candidate, as_of_str) or 0.0,
            recent_summary=criteria_str, trigger_description=trigger_desc, client=client,
        )
        count_basis = (f"{live_n} live occurrence(s) so far" if is_static else
                       f"{n_reached} recent occurrence(s) so far ({live_n} live, the rest backtest -- static "
                       f"candidates count live occurrences only; this one is Sonnet-proposed, so backtest tops "
                       f"up the count only until it has 50 live occurrences of its own)")
        message = (
            f"<b>{as_of_str}</b>\n\n"
            f"<b>Checkpoint at {n_reached} occurrences -- {escape_html(candidate)}</b>\n\n"
            f"({escape_html(trigger_desc)})\n\n"
            f"<b>{'VALIDATED' if cleared else 'NOT validated'}</b> -- {'cleared' if cleared else 'did not clear'} the "
            f"acceptance bar as of this checkpoint ({count_basis}).\n"
            f"{escape_html(criteria_str)}. (No single coin or period may carry more than 60% of the positive "
            f"return either, for either check to pass.)\n\n"
            f"Current status: <b>{escape_html(status)}</b>\n"
            f"Re-evaluated fresh at every {sh.MILESTONE_N}-occurrence checkpoint, not a permanent verdict -- re-checked "
            f"again at {n_reached + sh.MILESTONE_N} either way, unless dropped below.\n\n"
            f"{escape_html(hyperopt_runner.format_result(candidate))}\n\n"
            f"Sonnet's opinion (advisory only, not a verified finding): {escape_html(advice)}"
        )
        _send(message, reply_markup=PRUNE_KEYBOARD_TEMPLATE(candidate))
        sh.mark_milestone_reported(candidate, n_reached, cleared)


def _dynamic_trigger_hourly(spec: ConditionSpec, hourly: pd.DataFrame, daily: pd.DataFrame, funding) -> "pd.Series":
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
        trigger &= clause_signal_hourly(clause, hourly, daily, funding)
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
                spec = ConditionSpec(label=spec_dict["label"], clauses=tuple(clause_from_dict(c) for c in spec_dict["clauses"]),
                                      direction=spec_dict["direction"], horizons=tuple(spec_dict["horizons"]))
            except ValueError:
                # Recorded before a news/macro event clause became a NECESSARY
                # condition -- skip rather than crash, see
                # llm_pipeline/dynamic_candidates.py::registered_specs.
                continue
            trig = _dynamic_trigger_hourly(spec, hourly_to_date, ohlc_full[coin], funding).loc[day_start:day_end]
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
    last_battery_refresh = current
    d = current + pd.Timedelta(days=1)
    while d <= chunk_end:
        _check_live_tests(d)  # every day, not just event days -- a live test can resolve on any day
        _scan_mechanical_triggers(d, hourly_full, ohlc_full, static_triggers_full)  # unattended, no LLM -- opens a live test the moment any tracked trigger fires
        for message in state.due_reveals(d):  # a held-back test result surfacing on its own scheduled day
            _send(message)

        if (d - last_battery_refresh).days >= 7:
            status_summary = run_replay_battery(d)
            _check_prune_decisions(d, status_summary, client)
            _check_n50_milestones(d, status_summary, client)
            for candidate, info in status_summary.items():
                if "horizon_changed_to" in info:
                    _send(f"<b>{d.date()}</b>\n\n"
                          f"<b>Horizon updated -- {escape_html(candidate)}</b>\n\n"
                          f"({escape_html(_trigger_description(candidate))})\n\n"
                          f"Now held for <b>{info['horizon_changed_to']}d</b> going forward (empirically "
                          f"re-derived from accumulated history, replacing the previous value).")
            last_battery_refresh = d

        for series_key, series_label in MACRO_SERIES.items():
            if len(release_dates(series_key, d, d)) == 0:
                continue
            release = latest_release_with_prior(series_key, d)
            if release is None:
                continue
            event_desc = judgment.format_macro_event(series_label, release)
            try:
                assessment = judgment.judge_event(event_desc, client, as_of=d)
            except Exception as e:
                # A malformed model response for ONE event must not cost
                # every other day already processed this chunk -- same
                # reasoning as llm_pipeline/haiku_sonnet_pipeline.py::run_once().
                print(f"Failed to judge event on {d.date()} ({series_label}), skipping: {e}")
                continue
            events_this_chunk += 1
            if _handle_assessment(d, event_desc, assessment) == "STOP":
                state.save_checkpoint(str(d.date()), status="waiting_for_human")
                return {"stopped": "waiting_for_human", "current_date": str(d.date()), "events": events_this_chunk}

        for coin in COINS:
            transition = _shock_transition(ohlc_full[coin], d)
            if transition is None:
                continue
            z, direction = transition
            event_desc = judgment.format_shock_event(coin, z, direction)
            try:
                assessment = judgment.judge_event(event_desc, client, as_of=d, coin=coin)
            except Exception as e:
                print(f"Failed to judge event on {d.date()} ({coin} shock), skipping: {e}")
                continue
            events_this_chunk += 1
            if _handle_assessment(d, event_desc, assessment, live_coin=coin) == "STOP":
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
            tag = "validated" if info.get("milestone_cleared") else f"did not validate at its {n_reached}-live-test checkpoint"
        elif info["status"] == "accepted":
            tag = f"accepted, not yet validated (first checkpoint at {sh.MILESTONE_N} live tests)"
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

    s = pending["spec"]
    spec = ConditionSpec(label=s["label"], clauses=tuple(clause_from_dict(c) for c in s["clauses"]), direction=s["direction"])
    as_of = pd.Timestamp(pending["as_of"])
    result = test_novel_condition(spec, pending["coins"], as_of=as_of)
    status = result["status"]

    registry = state.load_dynamic_candidates()
    registry[spec.label] = {"label": spec.label,
                             "clauses": [clause_to_dict(c) for c in spec.clauses],
                             "direction": spec.direction, "horizons": list(spec.horizons)}
    state.save_dynamic_candidates(registry)
    # Recorded here, not deferred to the next weekly battery refresh -- without
    # this, a just-discovered candidate is already registered (and can already
    # open live tests via the mechanical scan, which doesn't check status_history
    # at all) but stays completely invisible to all_latest_statuses()/Sonnet's own
    # context for up to ~7 simulated days, since that function skips any candidate
    # with no status_log entry at all (a real, observed case this session: Sonnet
    # correctly said "not listed... so I can't state its status" for a candidate
    # that was already live-testing).
    sh.record_status(spec.label, status, pending["as_of"])
    state.save_pending_test(None)
    state.save_checkpoint(pending["as_of"], status="running")

    condition_str = f"{condition_desc(spec)} → {spec.direction}"
    reveal_date = as_of + pd.Timedelta(days=TEST_RESULT_DELAY_DAYS)
    lines = [f"<b>{reveal_date.date()}</b>", "",
             f"<b>Historical backtest -- {escape_html(spec.label)}</b>", "",
             f"({escape_html(condition_str)})"]
    pattern = result.get("pattern_significance") or {}
    if status not in ("insufficient_data",):
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

    # The occurrence that actually prompted this proposal gets its own live
    # test regardless of the verdict above -- testing starts the moment a
    # trigger is identified, not only once it's already accepted.
    if pending.get("live_coin") and status != "insufficient_data":
        execution = _open_live_test(spec.label, pending["live_coin"], spec.direction, as_of)
        if execution.get("opened"):
            lines.append("")
            lines.append(f"<b>Live test opened -- {spec.direction.upper()} {escape_html(pending['live_coin'])}</b>\n\n"
                         f"Held for <b>{execution['horizon']}d</b>, then resolved -- no TP/SL, this measures the same pattern the backtest found.")
    state.queue_reveal("\n".join(lines), str(reveal_date.date()))
    return status


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
    state.save_checkpoint(pending["as_of"], status="running")
    _send("Dismissed -- this condition won't be tested. Reply 'replay continue' to keep going.")
    return "dismissed"
