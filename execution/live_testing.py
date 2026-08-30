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
from candidates.methodology import path_outcome
from candidates.run_battery import COINS
from candidates import status_history as sh
from execution import hyperopt_runner
from execution import live_test_state as state
from llm_pipeline.dynamic_candidates import registered_specs
from llm_pipeline.haiku_sonnet_pipeline import sonnet_prune_advice
from llm_pipeline.novel_condition_tester import ConditionSpec, clause_signal_hourly
from telegram.bot import _send, escape_html

PLACEHOLDER_HORIZON_DAYS = 7  # neutral default (middle of HORIZONS_DAYS) -- see docs/case_study/methodology-decisions.md
BACKDATE_LOOKBACK_DAYS = 14  # how far back a newly-discovered condition's own triggering occurrence may be backdated
CONSECUTIVE_FAILURE_ALERT_THRESHOLD = 2  # see docs/case_study/methodology-decisions.md
CONSECUTIVE_FAILURE_CONTEXT_WINDOW = 5


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


def _format_live_test_opened(date, direction: str, coin: str, candidate: str, horizon: int) -> str:
    """Shared by both the static and dynamic branches of
    _scan_mechanical_triggers below -- bold header isolated on its own
    line, the (often long) trigger description on its own paragraph, the
    held-for duration bolded, so the message scans at a glance instead of
    reading as one dense run-on sentence.

    "no TP/SL" used to be appended and was removed here and in
    replay/engine.py together: no funded position is ever opened anywhere in
    this project, so saying it of one test implies some other test might have
    one."""
    return (f"<b>{date}</b>\n\n"
            f"<b>Live test opened -- {direction.upper()} {coin}</b>\n\n"
            f"(candidate <b>{escape_html(candidate)}</b>: {escape_html(_trigger_description(candidate))})\n\n"
            f"Held for <b>{horizon}d</b>.")


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


def _check_consecutive_failures(candidate: str) -> None:
    """Fires immediately after a live test resolves, only for a
    VALIDATED candidate (milestone_cleared -- see status_history.py),
    only for those: a well-established candidate's own aggregate
    significance test is, by design, resistant to a short losing streak
    (verified directly: a strong candidate can absorb 20-30 consecutive
    worst-case-magnitude failures before its p-value or MFE/MAE ratio
    would ever move enough to flip status -- see
    docs/case_study/methodology-decisions.md) -- which is exactly the
    right behavior against ordinary noise, but means the aggregate alone
    would be far too slow to surface a genuine regime change (a real
    shift in market structure, a rule change, an arbitraged-away
    inefficiency). This is a fast, purely informational early-warning a
    human can act on long before the aggregate statistics ever would --
    it never changes any candidate's status itself."""
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
        f"<b>Consecutive-failure alert -- {escape_html(candidate)}</b>\n",
        f"The last <b>{streak}</b> live test(s) for this VALIDATED candidate resolved negative in a row.\n",
        f"Last {len(window)} occurrence(s) for context:",
    ]
    for t in window:
        lines.append(f"  {t['close_date']}  {escape_html(t['coin'])}  return={t['forward_return']:+.2%}  "
                      f"MFE={t['mfe']:+.2%}  MAE={t['mae']:+.2%}")
    lines.append(f"\nOver these {len(window)}: mean return={mean_return:+.2%}, MFE/MAE={ratio:.2f} (favorable if > 1.0)")
    lines.append(f"\nInformational only -- this does not change <b>{escape_html(candidate)}</b>'s status. The full "
                 f"aggregate statistics (see /details <b>{escape_html(candidate)}</b>) are far more resistant to a short "
                 f"streak by design; this exists specifically to surface a genuine losing run long before the "
                 f"aggregate ever would.")
    _send("\n".join(lines))


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
        })
        _send(f"<b>{today.date()}</b>\n\n"
              f"<b>Live test resolved -- {trade['direction'].upper()} {trade['coin']}</b>\n\n"
              f"(candidate <b>{escape_html(trade['candidate'])}</b>: {escape_html(_trigger_description(trade['candidate']))}, "
              f"held {trade['horizon']}d, opened {trade['entry_date']})\n\n"
              f"Forward return: <b>{outcome['forward_return']:+.2%}</b>\n"
              f"Best point reached: {outcome['mfe']:+.2%}\n"
              f"Worst point reached: {outcome['mae']:+.2%}")
        _check_consecutive_failures(trade["candidate"])


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
                execution = _open_live_test(variant, coin, direction)
                if execution.get("opened"):
                    _send(_format_live_test_opened(now.date(), direction, coin, variant, execution["horizon"]))

        if not dynamic_specs:
            continue
        hourly_to_date = hourly_full[coin]
        for spec in dynamic_specs:
            if sh.is_dropped(spec.label) or (spec.label, coin) in open_pairs:
                continue
            trig = _dynamic_trigger_hourly(spec, hourly_to_date, ohlc_full[coin], funding, symbol=coin).loc[window_start:]
            if not trig.any():
                continue
            execution = _open_live_test(spec.label, coin, spec.direction)
            if execution.get("opened"):
                _send(_format_live_test_opened(now.date(), spec.direction, coin, spec.label, execution["horizon"]))


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


def _effective_milestone_count(candidate: str, backtest_n: int | None, live_n: int) -> int:
    """What counts toward a validation checkpoint differs by how the
    candidate was found. Static candidates (C1/C2/C6) were derived by
    directly mining this project's own historical data (a prior,
    dedicated research phase -- see docs/case_study/methodology-decisions.md)
    -- a direct look-then-test risk, so only genuinely prospective
    evidence (real resolved live tests) counts toward validating them,
    unchanged from before. Dynamic (Sonnet-proposed) candidates carry
    only a much weaker, diffuse version of that risk (Sonnet never sees
    this project's own backtest results before proposing -- only a
    live snapshot plus whatever general market-pattern knowledge its
    training absorbed), so they use a rolling window of the most recent
    50 occurrences, backtest and live mixed, chronologically -- since
    every live occurrence is by definition more recent than every
    backtest one, this is equivalent to filling the window with live
    occurrences first and topping up with the most recent backtest ones
    only while live_n hasn't reached 50 yet. Once a dynamic candidate
    accumulates 50 real live tests on its own, backtest contributes
    nothing further -- this collapses to the exact same live-only rule
    the static candidates always use."""
    if candidate in CANDIDATE_DIRECTIONS or live_n >= sh.MILESTONE_N:
        return live_n
    return min(backtest_n or 0, sh.MILESTONE_N - live_n) + live_n


def check_n50_milestones(status_summary: dict, client: Anthropic) -> None:
    """Mirrors replay/engine.py::_check_n50_milestones exactly -- NOT
    one-time, fires again every time a candidate crosses a NEW multiple
    of 50 in its own _effective_milestone_count (50, 100, 150, ...),
    each time re-asking the human whether to keep testing or drop it,
    real data instead of simulated. `status_summary` is the caller's
    freshly-computed battery result (e.g. weekly_revalidation.py's
    `result` DataFrame keyed by candidate) -- not re-derived here."""
    today_str = str(pd.Timestamp.now().date())
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
            candidate, years_tracked=sh.years_tracked(candidate) or 0.0,
            recent_summary=criteria_str, trigger_description=trigger_desc, client=client,
        )
        count_basis = (f"{live_n} real live occurrence(s) so far" if is_static else
                       f"{n_reached} recent occurrence(s) so far ({live_n} real live, the rest backtest -- "
                       f"static candidates count real live occurrences only; this one is Sonnet-proposed, so "
                       f"backtest tops up the count only until it has 50 real live occurrences of its own)")
        message = (
            f"<b>{today_str}</b>\n\n"
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
