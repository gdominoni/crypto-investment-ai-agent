"""Production's own live testing -- the real-data counterpart to
replay/engine.py's day-by-day walker. This project never opens a funded
position (see docs/case_study/methodology-decisions.md): a "live test"
here is the same observational record replay/ produces -- a real, dated
occurrence of an already-tracked trigger, held for the horizon
pattern_significance found significant, resolved by measuring the real
forward return/MFE/MAE, no TP/SL, no Freqtrade order. Intended to run on
a schedule (e.g. hourly, alongside the shock scan) via `run_once()`.

Shares the exact same pure statistical functions replay/engine.py uses
(candidates.methodology.path_outcome, candidates.definitions.compute_triggers,
llm_pipeline.novel_condition_tester's indicator whitelist) -- only the
data source (real, unsandboxed) and the state destination
(execution/live_test_state.py, candidates/status_history.py) differ from
the replay's own isolated versions. The orchestration here intentionally
mirrors replay/engine.py's structure closely so the two stay easy to
compare and keep in sync -- not merged into one shared module, to avoid
risking the replay's own already-verified behavior on a large refactor
(see docs/case_study/methodology-decisions.md for that tradeoff).
"""
from __future__ import annotations

import pandas as pd
from anthropic import Anthropic

from candidates.data_loading import load_daily, load_funding, load_hourly
from candidates.definitions import CANDIDATE_DIRECTIONS, TRIGGER_DESCRIPTIONS, compute_triggers
from candidates.methodology import SIGNIFICANCE_ALPHA, path_outcome, prune_recommendation, required_n_for_power
from candidates.run_battery import COINS
from candidates import status_history as sh
from execution import hyperopt_runner
from execution import live_test_state as state
from execution import signal_store
from llm_pipeline.dynamic_candidates import registered_specs
from llm_pipeline.novel_condition_tester import (
    ConditionSpec, clause_signal_hourly, condition_desc, is_testable, relax_to_testable,
    spec_from_dict, spec_to_dict,
)
from llm_pipeline.haiku_sonnet_pipeline import PROPOSAL_KEYBOARD_TEMPLATE
from llm_pipeline.pending_tests import push_pending_test
from telegram.bot import _send, escape_html, short_id as _short_id

MAX_DIGEST_ROWS = 8  # bounded by construction -- see send_monthly_digest
PLACEHOLDER_HORIZON_DAYS = 7  # neutral default (middle of HORIZONS_DAYS) -- see docs/case_study/methodology-decisions.md
BACKDATE_LOOKBACK_DAYS = 14  # how far back a newly-discovered condition's own triggering occurrence may be backdated


def _normalize_coin(coin: str) -> str | None:
    if coin in COINS:
        return coin
    guess = f"{coin.upper()}USDT"
    return guess if guess in COINS else None


def _trigger_description(candidate: str) -> str:
    base = candidate.rsplit("_", 1)[0]
    if base in TRIGGER_DESCRIPTIONS:
        return TRIGGER_DESCRIPTIONS[base]
    for spec in registered_specs():
        if spec.label == candidate:
            from llm_pipeline.novel_condition_tester import condition_desc
            return f"{condition_desc(spec)} → {spec.direction}"
    return "trigger definition not found -- treat this as missing information, do not guess at it"


def _open_live_test(candidate: str, coin: str, direction: str, decision_date: pd.Timestamp | None = None) -> dict:
    """`decision_date=None` (the default) means "as of right now" -- the
    normal case for a trigger caught by the mechanical scan below.
    Passed explicitly, it backdates entry to a real past bar (see
    `find_backdated_entry`), used ONLY for the specific occurrence that
    prompted a brand-new dynamic condition's discovery -- see that
    function's own docstring for why this is legitimate here (an
    observational record, never a funded order) but would never be for
    real capital."""
    coin = _normalize_coin(coin)
    if coin is None:
        return {"opened": False, "message": "REJECTED: coin not recognized -- refused."}
    horizon = int(state.load_horizons().get(candidate, PLACEHOLDER_HORIZON_DAYS))
    ohlc = load_daily(coin)
    if decision_date is None:
        entry_loc = len(ohlc.index) - 1
    else:
        after = ohlc.index[ohlc.index > decision_date]
        if len(after) == 0:
            return {"opened": False, "message": "No further price data available to open this live test yet."}
        entry_loc = ohlc.index.get_loc(after[0])
    entry_date = ohlc.index[entry_loc]
    entry_price = float(ohlc["open"].iloc[entry_loc]) if decision_date is not None else float(ohlc["close"].iloc[entry_loc])
    trade_id = state.append_trade({
        "candidate": candidate, "coin": coin, "direction": direction,
        "entry_date": str(entry_date.date()), "entry_price": entry_price,
        "entry_loc": int(entry_loc), "horizon": horizon,
        "backdated": decision_date is not None,
    })
    return {"opened": True, "trade_id": trade_id, "coin": coin, "direction": direction,
            "candidate": candidate, "entry_date": entry_date, "horizon": horizon}


def _market_return_over(entry_date, horizon: int, direction: str) -> float:
    """Equal-weighted forward return of the whole coin universe over the same
    window as one live test, signed by that test's direction. Kept identical to
    replay/engine.py's version so production's stored `baseline_return` means
    the same quantity the replay's does -- the two must stay comparable.

    The comparison a raw win rate cannot make. "The trend happened" and "the
    trend happened because of this condition" are different claims, and across
    2017-2026 a long-only rule is right most of the time for reasons that have
    nothing to do with any macro release.

    Keyed on the DATE, never on a positional index: the coins list at different
    times, so the same integer position is a different day on each of them --
    see replay/engine.py's own note for the measured effect that had."""
    import numpy as np

    entry_date = pd.Timestamp(entry_date)
    rets = []
    for coin in COINS:
        try:
            ohlc = load_daily(coin)
        except Exception:
            continue
        loc = int(ohlc.index.searchsorted(entry_date))
        if loc >= len(ohlc) or loc + horizon >= len(ohlc):
            continue
        entry = float(ohlc["open"].iloc[loc])
        exit_ = float(ohlc["close"].iloc[loc + horizon])
        if entry > 0:
            rets.append(exit_ / entry - 1.0)
    if not rets:
        return float("nan")
    r = float(np.mean(rets))
    return r if direction == "long" else -r


def _check_live_tests() -> None:
    """Resolves any open live test whose horizon has fully elapsed as of
    today -- same forward-return/MFE/MAE measure pattern_significance
    uses, no TP/SL, mirrors replay/engine.py::_check_live_tests exactly,
    against real (not simulated) data."""
    today = pd.Timestamp.now().normalize()
    for trade in state.load_open_trades():
        entry_date = pd.Timestamp(trade["entry_date"])
        elapsed = (today - entry_date).days
        if elapsed < trade["horizon"]:
            continue
        ohlc = load_daily(trade["coin"])
        if today not in ohlc.index:
            continue
        outcome = path_outcome(trade["entry_price"], trade["entry_loc"], ohlc, trade["direction"], trade["horizon"])
        if outcome["forward_return"] != outcome["forward_return"]:
            # NaN: the full horizon hasn't actually elapsed in the data yet
            # (path_outcome no longer silently clamps to the last bar -- see its
            # docstring). Leave the test OPEN and retry on the next run rather
            # than recording a partial hold as a resolved, full-horizon result.
            continue
        state.update_trade(trade["id"], {
            "status": "closed", "close_date": str(today.date()),
            "forward_return": outcome["forward_return"], "mfe": outcome["mfe"], "mae": outcome["mae"],
            # What simply holding the whole coin universe over the same window
            # did, signed the same way -- stored at resolution, exactly as
            # replay/engine.py does. Without it `check_n50_milestones`'s
            # market-adjusted line has no denominator and silently disappears,
            # leaving the raw "Trend Realized" figure standing alone, which is
            # the one number that most needs qualifying in a rising market.
            "baseline_return": _market_return_over(trade["entry_date"], trade["horizon"], trade["direction"]),
        })
        # NOT notified individually, and no per-resolution alert: measured on
        # this project's own registry, the open/resolve stream is ~56 messages a
        # day in bursts of 28, which is the same failure the replay already
        # diagnosed and fixed (95% of its traffic, and a rate limit that stalled
        # a real run). The dated record survives in full in the trade log and is
        # reachable through /details <name or id>; the periodic picture is
        # send_monthly_digest() below.


def _dynamic_trigger_hourly(spec: ConditionSpec, hourly: pd.DataFrame, daily: pd.DataFrame, funding,
                             symbol: str | None = None) -> pd.Series:
    """Identical logic to replay/engine.py's own version -- both now
    delegate every clause to `clause_signal_hourly`, the ONE shared
    implementation, which owns both the `DAILY_NATIVE_INDICATORS`
    exception and the `within_days` lag.

    Both used to be hand-rolled here and in the replay, and the
    daily-native set was a single hardcoded `shock_zscore` check -- which
    silently left four other indicators (rsi_14d, atr_pct_14d,
    daily_range_pct, efficiency_ratio_20d) measuring a DIFFERENT
    statistic live than the one the backtest accepted on. See that
    constant's own note for the measured distributions."""
    trigger = pd.Series(True, index=hourly.index)
    for clause in spec.clauses:
        trigger &= clause_signal_hourly(clause, hourly, daily, funding, symbol=symbol)
    return trigger


def find_backdated_entry(spec: ConditionSpec, coin: str, lookback_days: int = BACKDATE_LOOKBACK_DAYS) -> pd.Timestamp | None:
    """A newly-discovered dynamic condition's OWN triggering occurrence
    can't be entered "now" and called faithful to when it actually
    happened -- by the time Sonnet proposes it and a human approves, real
    time has already passed. Since this project never places a real
    order (see module docstring), there's nothing physically stopping an
    HONEST, real-data retroactive read: scan the last `lookback_days` of
    already-recorded hourly price history (kept fresh by
    data_ingestion/market_data/binance_fetcher.py) for the EARLIEST hour
    this exact condition was already true, and use that as the entry
    anchor instead of the discovery moment. Returns None if it wasn't
    true at all within the lookback window (the discovery moment IS the
    earliest known occurrence, nothing to backdate to)."""
    hourly = load_hourly(coin)
    daily = load_daily(coin)
    funding = load_funding(coin)
    # Computed on the FULL hourly series, sliced to the lookback window only
    # AFTER -- rolling-window indicators (e.g. a 720-hour funding z-score)
    # need real lookback history; slicing first starves it and silently
    # produces nothing but NaN/False (the exact bug already caught once in
    # replay/engine.py -- see docs/case_study/methodology-decisions.md).
    window_start = hourly.index.max() - pd.Timedelta(days=lookback_days)
    trig = _dynamic_trigger_hourly(spec, hourly, daily, funding, symbol=coin).loc[window_start:]
    true_hours = trig[trig]
    if len(true_hours) == 0:
        return None
    return true_hours.index[0]


def _static_triggers_full(hourly_full: dict) -> dict:
    """Precomputed once per run_once() call, not re-derived per coin
    inside the loop below -- see replay/engine.py's own version for why
    (rolling-window indicators need real lookback history; slicing to a
    tiny recent window before computing them starves that lookback)."""
    return {coin: compute_triggers(hourly_full[coin], load_funding(coin), scale=24) for coin in COINS}


def _scan_mechanical_triggers(hourly_full: dict, ohlc_full: dict, static_triggers_full: dict) -> None:
    """Unattended, no LLM involved -- for every TRACKED candidate (static
    + dynamic, not dropped, any status), checks whether its own trigger
    fired on any HOURLY bar in the last 24h, for every coin, and opens a
    live test the moment it does. Mirrors
    replay/engine.py::_scan_mechanical_triggers exactly, against real
    data with no `as_of` sandboxing needed."""
    now = pd.Timestamp.now()
    window_start = now - pd.Timedelta(hours=24)
    open_pairs = {(t["candidate"], t["coin"]) for t in state.load_open_trades()}
    dynamic_specs = registered_specs()

    for coin in COINS:
        funding = load_funding(coin)
        recent_static = static_triggers_full[coin].loc[window_start:]
        if len(recent_static):
            for variant, direction in CANDIDATE_DIRECTIONS.items():
                if sh.is_dropped(variant) or (variant, coin) in open_pairs or variant not in recent_static.columns:
                    continue
                if not recent_static[variant].any():
                    continue
                _open_live_test(variant, coin, direction)

        if not dynamic_specs:
            continue
        hourly_to_date = hourly_full[coin]
        for spec in dynamic_specs:
            if sh.is_dropped(spec.label) or (spec.label, coin) in open_pairs:
                continue
            trig = _dynamic_trigger_hourly(spec, hourly_to_date, ohlc_full[coin], funding, symbol=coin).loc[window_start:]
            if not trig.any():
                continue
            _open_live_test(spec.label, coin, spec.direction)


def _resolved_live_test_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in state.load_trade_log():
        if t["status"] == "closed":
            counts[t["candidate"]] = counts.get(t["candidate"], 0) + 1
    return counts


PRUNE_KEYBOARD_TEMPLATE = lambda candidate: {
    "inline_keyboard": [[
        {"text": "Keep Testing", "callback_data": f"prune:keep:{candidate}"},
        {"text": "Drop from Batch", "callback_data": f"prune:drop:{candidate}"},
    ]]
}


def _effective_milestone_count(candidate: str, prior_confirmations: int | None, live_n: int) -> int:
    """How many occurrences count toward CONFIRMING this candidate.

    An occurrence counts when it happened AFTER the hypothesis was written
    down -- the same rule replay/engine.py's own version of this function
    uses. `prior_confirmations` is that count, computed once at registration
    (`telegram/bot.py::handle_test_it_confirmation`, via
    `candidates.methodology.prospective_split`) and stored in
    `execution/live_test_state.py::load_confirmation_priors()`. It is
    nonzero only for a candidate promoted out of the parked-proposals queue
    (see `_check_parked_proposals` below) -- a freshly-tested proposal has
    nothing that could have postdated it yet, so it starts at zero, same as
    a static candidate.

    Static candidates (C1/C2/C6) count live occurrences only: they were
    mined from this project's own history, so none of their historical
    occurrences postdates the hypothesis -- this rule gives them the same
    zero their special case always gave.

    This REPLACES an earlier version that topped a dynamic candidate up
    with its FULL backtest count regardless of when it was written --
    letting a candidate with 120 historical occurrences reach its
    checkpoint on day one with zero live evidence, which is wrong for the
    same reason an occurrence from 2019 cannot confirm a hypothesis
    written in 2023 (see docs/case_study/methodology-decisions.md)."""
    if candidate in CANDIDATE_DIRECTIONS:
        return live_n
    return int(prior_confirmations or 0) + live_n


def _required_n_for(candidate: str, status_summary: dict) -> float:
    """How many occurrences this candidate needs before a null from it would
    mean anything, derived from its own realised volatility. NaN when that
    is not computable yet, in which case the message prints the achieved
    count alone rather than inventing a denominator."""
    sd = status_summary.get(candidate, {}).get("pattern_oos_sd")
    return required_n_for_power(sd) if isinstance(sd, (int, float)) else float("nan")


def check_n50_milestones(status_summary: dict, client: Anthropic) -> None:
    """Mirrors replay/engine.py::_check_n50_milestones exactly -- NOT
    one-time, fires again every time a candidate crosses a NEW multiple
    of sh.MILESTONE_N (20) in its own _effective_milestone_count, each
    time re-asking the human whether to keep testing or drop it, real
    data instead of simulated. `status_summary` is the caller's
    freshly-computed battery result (e.g. weekly_revalidation.py's
    `result` DataFrame keyed by candidate) -- not re-derived here."""
    today_str = str(pd.Timestamp.now().date())
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
        if not cleared:
            # The checkpoint is still RECORDED for every candidate that reaches
            # one -- `mark_milestone_reported` below runs either way, so the next
            # one fires at the right count and nothing about the accounting
            # changes. Only the message is withheld. A checkpoint that did not
            # clear is the ordinary case (most candidates never clear one), so
            # sending it turns the rarest and most meaningful message this
            # system produces into one of many.
            sh.mark_milestone_reported(candidate, n_reached, cleared)
            continue
        trigger_desc = _trigger_description(candidate)
        # Mirrors replay/engine.py: the opinion here was formed from the numbers
        # already in this message, with nothing added and no verification behind
        # it. prune_recommendation derives the assessment from them directly and
        # can use what the opinion could not -- whether there was POWER to detect
        # an effect, which is what separates "no" from "we could not tell".
        _verdict, advice = prune_recommendation(info)
        # "The trend happened" and "the trend happened BECAUSE of this condition"
        # are different claims -- the checkpoint carries the rate net of what
        # simply holding the market did.
        _closed = [t for t in state.load_trade_log()
                   if t["candidate"] == candidate and t["status"] == "closed"]
        _adj = [t["forward_return"] - t["baseline_return"] for t in _closed
                if isinstance(t.get("baseline_return"), (int, float))
                and t["baseline_return"] == t["baseline_return"]]
        raw_w = (sum(1 for t in _closed if t["forward_return"] > 0) / len(_closed)) if _closed else float("nan")
        adj_w = (sum(1 for r in _adj if r > 0) / len(_adj)) if _adj else float("nan")

        need = _required_n_for(candidate, status_summary)
        p_val = info.get("pattern_p_value")
        sig = info.get("pattern_significant")
        if p_val is None:
            sig_line = "Pattern Significance: NOT TESTED (no result on this sample yet)"
        else:
            verdict_word = "SIGNIFICANT" if sig else "NOT SIGNIFICANT"
            sig_line = (f"Pattern Significance: <b>{verdict_word}</b> "
                        f"(p = {p_val:.3f} | Target: p &lt; {SIGNIFICANCE_ALPHA:.3f})")
        if need == need:
            powered = n_reached >= need
            verdict = ("SAMPLE SUFFICIENT -- a null here is a measurement" if powered
                        else "Incomplete Sample for 80% Power")
            power_line = f"Power Progress: {n_reached:,} / {need:,.0f} occurrences ({verdict})"
        else:
            power_line = f"Power Progress: {n_reached:,} occurrences (required sample not yet computable)"

        mfe_mae_line = "MFE / MAE: not enough resolved occurrences yet"
        trend_line = f"Trend Realized: {raw_w:.1%}" if _closed else "Trend Realized: no resolved occurrences yet"
        lift_line = ""
        if _closed:
            _mfe = sum(t["mfe"] for t in _closed) / len(_closed)
            _mae = abs(sum(abs(t["mae"]) for t in _closed) / len(_closed))
            ratio = f" (Ratio: {_mfe / _mae:.2f})" if _mae else ""
            mfe_mae_line = f"MFE / MAE: {_mfe:+.2%} / {-_mae:+.2%}{ratio}"
        if _adj:
            lift_line = (f"\n• Market-Adjusted Excess: {sum(_adj) / len(_adj):+.2%} per occurrence "
                         f"vs universe baseline ({adj_w:.0%} positive after adjustment)")

        message = (
            f"<b>{today_str}</b>\n\n"
            f"<b>STATUS UPDATE: {escape_html(status.upper())}</b>\n"
            f"Candidate: <b>{escape_html(candidate)}</b> <code>{_short_id(candidate)}</code>\n"
            f"<i>({escape_html(trigger_desc)})</i>\n\n"
            f"<b>--- VERDICT &amp; POWER ---</b>\n"
            f"• {sig_line}\n"
            f"• {power_line}\n\n"
            f"<b>--- PERFORMANCE &amp; EXCURSION ---</b>\n"
            f"• {trend_line}{lift_line}\n"
            f"• {mfe_mae_line}\n\n"
            f"<b>--- EXECUTION (HYPEROPT) ---</b>\n"
            f"• {escape_html(hyperopt_runner.format_result(candidate, short=True))}\n\n"
            f"<b>---</b>\n"
            f"Next Checkpoint: {n_reached + sh.MILESTONE_N:,} occurrences "
            f"(re-evaluated fresh each time, never a permanent verdict)\n"
            f"<b>{'CONFIRMED' if cleared else 'NOT confirmed'}</b> at this checkpoint. "
            f"<i>Confirmed, not validated: persistence on an enlarged sample, not proof -- "
            f"a conclusive test needs the occurrence count shown above.</i>\n"
            f"Assessment: {escape_html(advice)}"
        )
        _send(message, reply_markup=PRUNE_KEYBOARD_TEMPLATE(candidate))
        sh.mark_milestone_reported(candidate, n_reached, cleared)


def _check_parked_proposals() -> None:
    """Re-checks every parked proposal (see
    `llm_pipeline/haiku_sonnet_pipeline.py::run_compression_scan`, where a
    too-rare proposal that couldn't even be rescued by
    `relax_to_testable` is parked here rather than discarded) and
    promotes any that have become testable back to the human for
    approval, carrying the proposal's ORIGINAL `proposed_at` date forward
    so a later CONFIRMED checkpoint counts only occurrences that postdate
    the actual hypothesis, not its promotion.

    Intended to run once a day (see scheduler/live_daemon.py's daily
    cadence) -- unlike replay/engine.py's own version, this does NOT
    stagger the queue across several days. Staggering exists there
    because a compressed nine-year replay accumulates up to ~100 parked
    proposals and `is_testable()` costs ~146ms each -- unstaggered, ~9
    hours of wall-clock per simulated day. Production accumulates parked
    proposals at real-world speed, so even a few dozen entries cost only
    seconds once a day; the complexity that problem justified doesn't
    apply here.

    Also unlike the replay (a single pending slot, so only the oldest
    testable proposal is promoted per check), every proposal that becomes
    testable today is promoted: production's pending-test queue already
    supports several proposals awaiting a human answer at once (each with
    its own id and its own buttons -- see llm_pipeline/pending_tests.py),
    so there is no slot to protect."""
    parked = state.load_parked_proposals()
    for entry in sorted(parked, key=lambda e: e.get("proposed_at", "")):
        try:
            spec = spec_from_dict(entry["spec"])
        except ValueError:
            state.unpark_proposal(entry["spec"].get("label", ""))
            continue
        if is_testable(spec, COINS) is not None:
            continue
        state.unpark_proposal(spec.label)
        proposed_at = entry.get("proposed_at")
        pending_id = push_pending_test(spec, COINS, live_coin=None, signal_class="promoted_from_parking",
                                        proposed_at=proposed_at)
        message = (
            f"<b>{str(pd.Timestamp.now().date())}</b>\n\n"
            f"<b>A parked hypothesis now has enough history to test</b>\n\n"
            f"Proposed {proposed_at}, parked because it had not occurred often enough to measure. "
            f"It has now.\n\n"
            f"<b>{escape_html(spec.label)}</b>\n"
            f"({escape_html(condition_desc(spec))} → {spec.direction.upper()})"
        )
        _send(message, reply_markup=PROPOSAL_KEYBOARD_TEMPLATE(pending_id))


def send_monthly_digest(since: pd.Timestamp) -> None:
    """The one periodic message about live testing, replacing the per-test
    stream this module used to send on every open and every resolve.

    Mirrors replay/engine.py::_send_monthly_digest, which replaced the same
    stream there for the same measured reason: over a full replay that stream
    was 15,500 of 16,363 messages, and Telegram answered the volume with a
    rate limit long enough to stall a real run. Production fires roughly 28
    live tests in a day across the tracked battery, so the same design applies
    at a smaller scale -- ~56 messages a day, in bursts, is still a stream
    nobody reads.

    Bounded by construction: a fixed header plus at most MAX_DIGEST_ROWS rows
    and a count for the rest, so it cannot grow into Telegram's 4,096-character
    limit as the battery does."""
    log = state.load_trade_log()
    opened = [t for t in log if pd.Timestamp(t["entry_date"]) >= since]
    closed = [t for t in log if t["status"] == "closed" and pd.Timestamp(t["close_date"]) >= since]
    still_open = len(state.load_open_trades())
    all_closed = [t for t in log if t["status"] == "closed"]

    def _mean(xs):
        return (sum(xs) / len(xs)) if xs else float("nan")

    pos = sum(1 for t in closed if t["forward_return"] > 0)
    all_pos = sum(1 for t in all_closed if t["forward_return"] > 0)

    lines = [f"<b>{'━' * 3} MONTHLY DIGEST -- {pd.Timestamp.now().strftime('%B %Y')} {'━' * 3}</b>", ""]
    lines.append(f"<b>Live tests</b>  {len(opened)} opened - {len(closed)} resolved - {still_open} still open")
    if closed:
        mfe, mae = _mean([t["mfe"] for t in closed]), _mean([abs(t["mae"]) for t in closed])
        ratio = f"{mfe / mae:.2f}" if mae else "n/a"
        lines.append(f"<b>This month</b>  {pos}/{len(closed)} positive ({pos / len(closed):.0%}) - "
                      f"mean {_mean([t['forward_return'] for t in closed]):+.2%} - MFE/MAE {ratio}")
    if all_closed:
        lines.append(f"<b>All time</b>  {len(all_closed)} resolved - {all_pos / len(all_closed):.0%} positive - "
                      f"mean {_mean([t['forward_return'] for t in all_closed]):+.2%}")

    # Ranked by how far along the confirmation count each trigger is, NEVER by
    # how well it has done: ordering by success rate puts the luckiest small
    # sample on top -- measured on a real run, that meant candidates at n=6 with
    # a 100% hit rate whose own backtest status was `rejected`.
    by_candidate: dict[str, list] = {}
    for t in all_closed:
        by_candidate.setdefault(t["candidate"], []).append(t["forward_return"])
    priors = state.load_confirmation_priors()
    statuses = sh.all_latest_statuses()
    battery = signal_store.load_battery_state() or {}
    summary = {k: v for k, v in (battery.get("summary") or {}).items()}
    rows = []
    for name, rets in by_candidate.items():
        if statuses.get(name, {}).get("dropped"):
            continue
        rows.append((_effective_milestone_count(name, priors.get(name), len(rets)), name, rets))
    rows.sort(reverse=True)
    if rows:
        lines.append("")
        lines.append("<b>Confirmation progress</b> (none of these is a result -- the denominator is the point)")
        powered = 0
        for n, name, rets in rows[:MAX_DIGEST_ROWS]:
            need = _required_n_for(name, summary)
            if need == need and n >= need:
                need_txt = f" -- <b>powered</b> (needed {need:.0f})"
                powered += 1
            elif need == need:
                need_txt = f" of {need:.0f} needed for power"
            else:
                need_txt = ""
            wins = sum(1 for r in rets if r > 0)
            status = statuses.get(name, {}).get("status", "?")
            lines.append(f"  <b>{escape_html(name[:34])}</b> <code>{_short_id(name)}</code>  "
                          f"confirmed {n}{need_txt}  -  trend {wins / len(rets):.0%}  -  {status}")
        if len(rows) > MAX_DIGEST_ROWS:
            lines.append(f"  <i>... and {len(rows) - MAX_DIGEST_ROWS} more -- /summary for all of them</i>")
        if powered:
            lines.append(f"  <i>{powered} of the rows above are past their power threshold: for those, "
                          f"'no effect found' is a measurement, not a missing answer.</i>")

    active = [k for k, v in statuses.items() if not v.get("dropped")]
    confirmed = [k for k, v in statuses.items() if v.get("milestone_cleared")]
    reached = [k for k, v in statuses.items() if v.get("milestone_reported")]
    lines.append("")
    lines.append(f"<b>Battery</b>  {len(statuses)} tracked - {len(active)} active - "
                  f"{len(reached)} reached a checkpoint - {len(confirmed)} currently CONFIRMED - "
                  f"{len(state.load_parked_proposals())} parked")
    lines.append("")
    lines.append("<i>Individual live tests are not sent one by one. Every figure above comes from the full "
                  "trade log -- /summary for the table, /details &lt;name or id&gt; for one trigger with its "
                  "last dated occurrences.</i>")
    _send("\n".join(lines))


def run_once() -> None:
    """Entry point for a scheduled job (hourly, alongside the shock
    scan): resolves due live tests, then scans every tracked trigger for
    new occurrences. Doesn't refresh the battery itself or check
    milestones -- those stay on weekly_revalidation.py's own cadence,
    which should call check_n50_milestones(status_summary, client) with
    its own freshly-computed status_summary after each battery refresh."""
    _check_live_tests()
    ohlc_full = {c: load_daily(c) for c in COINS}
    hourly_full = {c: load_hourly(c) for c in COINS}
    static_triggers_full = _static_triggers_full(hourly_full)
    _scan_mechanical_triggers(hourly_full, ohlc_full, static_triggers_full)
