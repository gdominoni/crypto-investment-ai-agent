"""Sonnet's judgment calls for the historical replay -- structurally the
same decision this project's live pipeline makes (llm_pipeline/haiku_sonnet_pipeline.py::sonnet_strategist),
adapted for a REAL, dated, structured event (a macro release, a
volatility shock) instead of a news headline, since the replay
deliberately avoids needing any invented content: every event fed in
here is real and dated, and the battery/history context is built from
the replay's OWN isolated state (replay/state.py), never production's.
"""
from __future__ import annotations

import json

import pandas as pd
from anthropic import Anthropic

from candidates.macro_vintage import recent_releases_summary
from candidates.methodology import STATUS_PLAIN
from candidates.run_battery import COINS
from llm_pipeline.haiku_sonnet_pipeline import SONNET_MODEL, _strip_fences, cached_system, escape_html, extract_text, format_spec_clauses
from llm_pipeline.novel_condition_tester import SUPPORTED_INDICATORS, build_indicator_leadup, build_indicator_snapshot
from replay import state
from replay import status_history as sh
from replay.time_sandbox import daily_as_of
from llm_pipeline import usage as _usage

REPLAY_SYSTEM_PROMPT = f"""You are a market strategist for a crypto trading system, in a historical replay: \
you are being asked to make the same real-time judgment call the live system would have made on a specific \
past date, using ONLY what was actually public knowledge as of that date -- a real, dated macro data release \
or a real, dated volatility shock, never a news headline (this replay deliberately doesn't use any invented \
or curated content).

You will be given: (1) the event itself, (2) the replay's own candidate battery status as of this date \
(which candidates, if any, are 'accepted' as of TODAY in the simulation -- meaning they cleared the \
historical/backtest bar; this is not a claim of a live track record), (3) a short summary of this \
replay's trade history so far. Trades are never yours to open -- an unattended, purely mechanical scan \
already opens a live test the moment an accepted candidate's own trigger condition is met, with no LLM \
judgment involved. Your only job here is to decide whether this event suggests a genuinely new, \
untested pattern -- especially a SPECIFIC combination of what you were shown (an indicator reading, a \
recent macro release) rather than the event in isolation -- not covered by any existing candidate, \
using ONLY these whitelisted indicators: {list(SUPPORTED_INDICATORS)}. Ground any compound hypothesis \
ONLY in indicator readings/releases you were actually given -- never invent one. A single clause is \
fine when nothing else in the given context looks relevant.

Return ONLY a JSON object with exactly these fields:
- "assessment": 1-2 sentences
- "recommended_action": one of "no_action", "propose_novel_test"
- "novel_condition_spec": null, or {{"label": "...", "clauses": [{{"indicator": "...", \
"op": "<"/">"/"<="/">=", "threshold": <number>, "within_days": <integer 0-14, optional>}}, ...], \
"direction": "long"/"short", "coins": <optional list>, "outcome": "raw"/"market_relative" (optional)}}

CONDITIONS MAY BE SEQUENCED, not just simultaneous. Each clause takes an optional "within_days" \
(0-14, default 0). 0 means "true on the day the condition fires"; K means "was true at any point in \
the last K days". This is what expresses an ORDERING rather than a coincidence, and the two are \
genuinely different hypotheses:
  - crash FIRST, then the release:  shock_zscore >= 2 with within_days=3, AND today's condition
  - release FIRST, then the move:   is_macro_day >= 1 with within_days=2, AND today's condition
The lead-up table you are given exists precisely so you can tell these apart -- use it.

IS THIS EVENT ABOUT ONE COIN OR THE WHOLE MARKET? The two need different settings:
  - MARKET-WIDE (a CPI print, an FOMC decision): omit "coins", leave "outcome" as "raw". The whole \
    market moving together IS the effect; measuring each coin against the market would subtract the \
    very thing being tested.
  - COIN-SPECIFIC (something affecting one asset and not its peers): set "coins" to just that coin, \
    e.g. ["XRPUSDT"], and set "outcome" to "market_relative". About 54% of any coin's move is simply \
    the market's move, so removing it isolates what is specific to this one.
If unsure, omit both and the whole-market defaults apply.

"prior_weight" (optional, 0.25-4.0, default 1.0) is HOW PLAUSIBLE you think this hypothesis is \
BEFORE it is tested, and it must be justified by the mechanism you can actually articulate -- not by \
how much you would like it to be true. 1.0 is neutral. Use above 1.0 only when there is a specific \
reason to expect this effect (a plausible causal story linking THIS event type to THIS market state \
in THIS direction); use below 1.0 for a speculative combination you are proposing mainly to rule it \
out. This does NOT make a hypothesis easier to accept on its own: the family shares one fixed error \
budget, so weighting one condition up makes every other condition tested alongside it harder to \
accept. Marking everything highly plausible therefore achieves exactly nothing. Your weight is \
recorded now and never revised after the result is seen.

No prose, no markdown fences, just the JSON object."""


def _battery_context() -> str:
    battery = state.load_battery_status()
    candidates = battery.get("candidates", {})
    if not candidates:
        return ("No candidate currently has 'accepted' status as of this date -- nothing is currently having its "
                "own trigger opened as a live test right now (this project never places a funded trade at all; "
                "'accepted' only means a candidate's own trigger opens an observational live test automatically).")
    return (f"Currently accepted (as of this date, live-testing automatically on their own trigger): "
            f"{', '.join(candidates.keys())}.")


def _all_candidates_status_summary() -> str:
    """Every candidate ever tracked (all 6 static C1/C2/C6 variants, plus
    any dynamic ones discovered via "test it") and its latest status --
    rejected and watch included, not just the accepted subset
    `_battery_context()` uses for trade decisions. This is what a
    question like "what's been tested, anything accepted or dropped?"
    actually needs to answer completely. `status` is the raw, ongoing
    technical classification (accepted/watch/rejected); a separate
    "validated" tag is added only for candidates that have actually
    crossed their 50-live-test milestone while accepted -- see
    replay/status_history.py::mark_milestone_reported.

    'insufficient_data' candidates -- almost always the large majority
    once the dynamic registry grows (67 of 96 tracked, as of one real
    checkpoint this case study reached) -- are collapsed to a count plus
    name list instead of one full line each: there's rarely anything
    candidate-specific worth saying about a trigger that simply hasn't
    fired enough times yet, so the per-line detail costs tokens on every
    future call without adding information most questions would ever
    use. accepted/watch/rejected/dropped candidates (the ones a question
    is actually likely to be about) keep their full per-line detail."""
    statuses = sh.all_latest_statuses()
    if not statuses:
        return "No candidates tracked yet."
    lines = ["All tracked candidates and their latest status:"]
    insufficient = []
    for name, info in sorted(statuses.items()):
        if info["status"] == "insufficient_data" and not info["dropped"]:
            insufficient.append(name)
            continue
        tags = []
        if info["dropped"]:
            tags.append("dropped")
        if info.get("milestone_reported"):
            n_reached = info.get("last_checkpoint_n", sh.MILESTONE_N)
            tags.append("validated" if info.get("milestone_cleared") else f"did not validate at its {n_reached}-live-test checkpoint")
        tag = f" ({', '.join(tags)})" if tags else ""
        lines.append(f"{name}: {STATUS_PLAIN.get(info['status'], info['status'])}{tag}")
    if insufficient:
        lines.append(f"Insufficient data, not enough historical occurrences yet to test ({len(insufficient)}): " + ", ".join(insufficient))
    return "\n".join(lines)


def _history_summary() -> str:
    log = state.load_trade_log()
    if not log:
        return "No live tests yet this replay."
    positive = sum(1 for t in log if t.get("status") == "closed" and t.get("forward_return", 0) > 0)
    return f"{len(log)} live test(s) opened so far this replay, {positive} resolved with a positive forward return."


def format_macro_event(name: str, release: dict) -> str:
    change = release["value"] - release["prior_value"] if release.get("prior_value") is not None else None
    change_str = f", change from prior release: {change:+.3f}" if change is not None else " (first known release)"
    return (f"MACRO RELEASE: {name}, published {release['realtime_start'].date()}, "
            f"for period {release['period'].date()}: value={release['value']}{change_str}")


def format_shock_event(symbol: str, shock_z: float, direction: str) -> str:
    return f"VOLATILITY SHOCK: {symbol}, direction={direction}, shock_z={shock_z:.2f} (roughly the top ~2% most extreme volatility episodes for this coin, as of this date)"


def judge_event(event_description: str, client: Anthropic, as_of: pd.Timestamp | None = None,
                 coin: str | None = None) -> dict:
    """`as_of`, when given (always passed by the day-by-day walker), adds
    real, time-sandboxed indicator readings and the real macro releases
    from the last 10 (simulated) days -- the same real, dated context
    production's shock handling gets, so Sonnet can ground a compound
    novel_condition_spec in actual numbers instead of guessing. A
    pattern doesn't require a shock to exist -- Sonnet's discovery role
    is never limited to shock-triggered events, so a macro event (no
    single coin of its own) gets the indicator snapshot for EVERY
    tracked coin, not none at all; a shock event (already about one
    specific coin) gets just that coin's, the more targeted read. Both
    are skipped only when `as_of` itself isn't given (`judge_event` is
    also used from contexts -- tests, ad-hoc questions -- that don't
    need it)."""
    parts = [event_description]
    if as_of is not None:
        if coin is not None:
            parts.append(f"INDICATOR SNAPSHOT:\n{build_indicator_snapshot(coin, as_of)}\n\n"
                          f"{build_indicator_leadup(coin, as_of=as_of)}")
        else:
            snapshots = "\n".join(build_indicator_snapshot(c, as_of) for c in COINS)
            parts.append(f"INDICATOR SNAPSHOT (every tracked coin):\n{snapshots}")
        parts.append(f"RECENT MACRO RELEASES (last 10 simulated days):\n{recent_releases_summary(as_of)}")
    parts.append(f"BATTERY CONTEXT:\n{_battery_context()}")
    parts.append(f"REPLAY HISTORY:\n{_history_summary()}")
    user_content = "\n\n".join(parts)
    response = client.messages.create(
        # 2000, not 700 -- observed live (2026-08-28 replay run): the model emits a
        # thinking block even though `thinking` is never requested here, and at 700
        # it sometimes consumed the entire budget before any text block existed at
        # all (extract_text raising "No text block"), or truncated the JSON mid-
        # string (json.loads raising). Headroom fixes both, whatever the model's
        # exact reason for thinking is -- see docs/case_study/methodology-decisions.md.
        model=SONNET_MODEL, max_tokens=2000, system=cached_system(REPLAY_SYSTEM_PROMPT),
        messages=[{"role": "user", "content": user_content}],
    )
    _usage.record(response, "replay.shock" if coin else "replay.macro", SONNET_MODEL)
    return json.loads(_strip_fences(extract_text(response)))


MARKET_CHECK_SYSTEM_PROMPT = """You are a market-check assistant for a crypto pattern-discovery system \
(NOT a trading system -- no funded position is ever opened; "accepted" only means a candidate's own \
trigger opens an observational live test), answering a question about a historical replay's current \
simulated state (a specific past date, walked forward day by day -- not live). Given real technical/ \
portfolio state as of that simulated date and the live candidate battery context, answer the user's \
question in 2-4 sentences. Cite only the numbers given to you in the state below -- never invent a \
price, a percentage, or a live-test detail. The reader may not know this project's internal vocabulary \
(status codes like "insufficient_data", "watch"; terms like "validated" vs. "accepted") -- briefly \
explain any such term you use in plain language rather than stating it bare, the way you'd explain a \
piece of jargon to someone unfamiliar with the system. If nothing in the given state answers the \
question, say so plainly rather than guessing."""


def _open_positions_summary() -> str:
    """No real position is ever open -- these are live tests of an
    already-accepted pattern (see replay/engine.py::_open_live_test),
    held for a fixed horizon with no TP/SL, not a funded trade."""
    open_tests = state.load_open_trades()
    if not open_tests:
        return "No live tests in progress right now."
    lines = [f"{len(open_tests)} live test(s) in progress:"]
    for t in open_tests:
        lines.append(f"  - {t['coin']} {t['direction'].upper()}, opened {t['entry_date']} at {t['entry_price']}, "
                     f"candidate {t['candidate']}, held for {t['horizon']}d")
    return "\n".join(lines)


def _last_closed_summary() -> str:
    closed = [t for t in state.load_trade_log() if t["status"] == "closed"]
    if not closed:
        return "No resolved live tests yet this replay."
    last = max(closed, key=lambda t: t["close_date"])
    return (f"Last resolved live test: {last['coin']} {last['direction'].upper()}, forward return {last['forward_return']:+.2%} "
            f"(MFE {last['mfe']:+.2%}, MAE {last['mae']:+.2%}), closed {last['close_date']}.")


def _trades_by_candidate_summary(top_n: int = 15) -> str:
    """Real forward-return/Sortino per candidate, computed on the LIVE
    test outcomes with the exact same sortino_ratio() the rest of this
    project uses everywhere else -- handed to Sonnet as already-computed
    fact to relay, not something it estimates itself, matching this
    project's "cite only given numbers" discipline for anything
    quantitative. "win_rate" here just means "fraction with a positive
    forward return", not the formal win/loss/timeout classification a
    TP/SL structure would produce -- there's no barrier here.

    Capped at `top_n` candidates (ranked by |mean_return|, the ones most
    worth a human's attention either way) -- without this, this block's
    size scales with the number of candidates ever discovered, which
    only ever grows, making every SUBSEQUENT judge_event/market-question
    call more expensive than the last regardless of that specific
    question's own content (a real, observed effect: this text alone
    reached ~1,400 tokens by the time 96 candidates had been discovered,
    growing further every week). A truncation note directs anyone who
    genuinely needs the full list to /replay_summary (or /summary in
    production) -- a free, local, no-LLM command built for exactly
    that, rather than paying for the same exhaustive dump on every call
    whether or not it's the one being asked about."""
    import numpy as np
    from candidates.methodology import sortino_ratio

    closed = [t for t in state.load_trade_log() if t["status"] == "closed"]
    if not closed:
        return "No resolved live tests yet this replay."
    by_candidate: dict[str, list[dict]] = {}
    for t in closed:
        by_candidate.setdefault(t["candidate"], []).append(t)
    rows = []
    for candidate, trades in by_candidate.items():
        n = len(trades)
        positive = sum(1 for t in trades if t["forward_return"] > 0)
        returns = np.array([t["forward_return"] for t in trades])
        sortino = sortino_ratio(returns)
        sortino_str = f"{sortino:.2f}" if not np.isnan(sortino) else "n/a (no negative return yet to measure downside against)"
        rows.append((abs(returns.mean()), f"{candidate}: N={n}, positive_rate={positive / n:.1%}, "
                                           f"mean_return={returns.mean():+.2%}, sortino={sortino_str}"))
    rows.sort(key=lambda r: r[0], reverse=True)
    lines = [line for _, line in rows[:top_n]]
    header = "By-candidate breakdown (all resolved live tests)"
    if len(rows) > top_n:
        header += f", top {top_n} of {len(rows)} by |mean return| -- ask /replay_summary for the full list"
    return header + ":\n" + "\n".join(lines)


def _trades_by_candidate_and_coin_summary(top_n: int = 15) -> str:
    """Same real numbers as _trades_by_candidate_summary, one level more
    granular -- per (candidate, coin), for a question like "give me the
    per-coin split of X's results" (the concentration_check gate already
    checks per-coin dominance internally during backtesting; this is the
    same idea surfaced for a human asking about it directly, over LIVE
    test outcomes specifically). Capped at `top_n` pairs the same way and
    for the same reason as _trades_by_candidate_summary above -- this was
    in fact the single largest block in the whole context (~5,200 tokens
    at 96 tracked candidates), since candidate x coin pairs grow faster
    than candidates alone."""
    import numpy as np

    closed = [t for t in state.load_trade_log() if t["status"] == "closed"]
    if not closed:
        return "No resolved live tests yet this replay."
    by_pair: dict[tuple[str, str], list[dict]] = {}
    for t in closed:
        by_pair.setdefault((t["candidate"], t["coin"]), []).append(t)
    rows = []
    for (candidate, coin), trades in by_pair.items():
        n = len(trades)
        positive = sum(1 for t in trades if t["forward_return"] > 0)
        returns = np.array([t["forward_return"] for t in trades])
        rows.append((abs(returns.mean()), f"{candidate} / {coin}: N={n}, positive_rate={positive / n:.1%}, "
                                           f"mean_return={returns.mean():+.2%}"))
    rows.sort(key=lambda r: r[0], reverse=True)
    lines = [line for _, line in rows[:top_n]]
    header = "By-candidate-and-coin breakdown (all resolved live tests)"
    if len(rows) > top_n:
        header += f", top {top_n} of {len(rows)} by |mean return| -- ask /replay_summary for the full list"
    return header + ":\n" + "\n".join(lines)


def _price_snapshot(as_of: pd.Timestamp) -> str:
    """1-day and 7-day price change per coin, as of the simulated date --
    without this, a broad question like "how's the market going" has no
    real price data to answer from, and Sonnet (correctly) refuses to
    guess rather than invent a number."""
    lines = []
    for coin in COINS:
        df = daily_as_of(coin, as_of)
        if len(df) < 8:
            continue
        chg_1d = df["close"].iloc[-1] / df["close"].iloc[-2] - 1
        chg_7d = df["close"].iloc[-1] / df["close"].iloc[-8] - 1
        lines.append(f"{coin}: {chg_1d:+.1%} (1d), {chg_7d:+.1%} (7d)")
    return "Price snapshot -- " + "; ".join(lines) if lines else "No price data available."


def answer_market_question(question: str, client: Anthropic) -> str:
    checkpoint = state.load_checkpoint()
    as_of_str = checkpoint.get("current_date") or "not started yet"
    as_of = pd.Timestamp(as_of_str) if checkpoint.get("current_date") else None
    price_line = _price_snapshot(as_of) if as_of is not None else "Replay not started yet -- no price data."
    snapshot = (f"Simulated date: {as_of_str}.\n{price_line}\n{_open_positions_summary()}\n"
                f"{_last_closed_summary()}\n\n{_trades_by_candidate_summary()}\n\n{_trades_by_candidate_and_coin_summary()}")
    context = (f"CANDIDATE BATTERY STATUS (as of {as_of_str}):\n{_battery_context()}\n\n"
               f"{_all_candidates_status_summary()}")
    response = client.messages.create(
        # 2000, not 800 -- 800 was STILL observed live (2026-08-28) to truncate
        # mid-answer (stop_reason="max_tokens") on a longer, itemized question
        # ("list every open live test, split by trigger") -- thinking alone used
        # over half that budget. Same root cause as judge_event's own comment,
        # just needed more headroom than the earlier bump gave it.
        model=SONNET_MODEL, max_tokens=2000, system=MARKET_CHECK_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"USER QUESTION: {question}\n\nSTATE:\n{snapshot}\n\n{context}"}],
    )
    _usage.record(response, "replay.market_check", SONNET_MODEL)
    return escape_html(extract_text(response))


def format_telegram_message(as_of, event_description: str, assessment: dict) -> str:
    """Deliberately formatted to look exactly like what the live pipeline
    itself would send (llm_pipeline/haiku_sonnet_pipeline.py::format_sonnet_message/
    format_shock_message) -- no "this is a simulation" framing inside the
    message body. That disclosure belongs in the surrounding write-up
    this replay is presented in, not baked into every individual message
    the way a UI mockup doesn't stamp "MOCKUP" on every button. The date
    is still shown, plainly, since messages fire in rapid real succession
    here but represent dates spread across a year -- without it the
    sequence wouldn't read as a coherent history at all.

    Same reasoning as format_sonnet_message on the action label: it
    depends on what's actually happening, not a uniform "Recommended
    action: X" line -- only propose_novel_test is genuinely a decision
    pending on the human."""
    base = (
        f"<b>{as_of.date()}</b>\n\n"
        f"<b>Event:</b> {escape_html(event_description)}\n\n"
        f"<b>Assessment:</b> {escape_html(assessment['assessment'])}"
    )
    action = assessment["recommended_action"]
    if action == "propose_novel_test" and assessment.get("novel_condition_spec"):
        spec = assessment["novel_condition_spec"]
        base += (
            f"\n\n<b>This needs your input.</b>\n\n"
            f"<b>Proposed test: \"{escape_html(spec['label'])}\"</b>\n\n"
            f"<i>Exactly what would be tested:</i> {format_spec_clauses(spec)} → {escape_html(spec['direction'])}\n\n"
            f"Test It runs a real walk-forward backtest of this condition before it's tracked as a live test "
            f"(no real money is ever placed on it). Don't Test It dismisses this proposal."
        )
    else:
        base += "\n\n<i>No action taken -- logged, nothing further needed.</i>"
    return base
