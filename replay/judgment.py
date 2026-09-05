"""Sonnet's judgment calls for the historical replay -- structurally the
same decision this project's live pipeline makes (llm_pipeline/haiku_sonnet_pipeline.py::sonnet_compression_response),
adapted for a REAL, dated, structured event (a confirmed exit from a
volatility-compression episode), since the replay
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
from llm_pipeline.novel_condition_tester import (EVENT_INDICATORS, INDICATOR_PLAIN_NAMES,
                                                  MIN_HISTORICAL_OCCURRENCES, build_indicator_leadup,
                                                  build_indicator_snapshot, proposable_indicators)
from replay import state
from replay import status_history as sh
from replay.time_sandbox import daily_as_of
from llm_pipeline import usage as _usage

REPLAY_SYSTEM_PROMPT = f"""You are a quantitative researcher, not a trader. Your subject is a single question: \
does a real-world macro or news EVENT produce a measurable, repeatable change in crypto prices \
over the following days? Nothing here is a trading system -- no position is ever opened, no \
money is ever at risk, and there is no entry signal to find.

That distinction decides what a good answer looks like. A chart setup -- oversold RSI, a \
Bollinger touch, a volume spike -- is NOT a hypothesis in this project, \
however well it would work as a trade. Those readings are CONTEXT: they describe the state the \
market happened to be in when the event landed. The event is the subject; the market state \
merely says under what circumstances you think it matters. If your idea would still make sense \
with the macro release deleted from it, it is a chart pattern and does not belong here.

The instinct to reach for the technical indicators first is the right one for a market \
strategist and the wrong one for this task. Start from the EVENT -- what was published, how far \
it moved from what was expected -- and only then ask which market conditions would make its \
effect visible.

WHEN YOU ARE ASKED, and why it matters for your answer. You are consulted at exactly one kind of \
moment: a period of unusually LOW volatility for this coin has just ended. The market coiled, stayed \
quiet for some days, and has now begun to move again.

That moment is chosen because a compressed market is more likely than an ordinary one to be followed \
by a sustained directional move -- but nothing about the compression says WHICH DIRECTION. That is \
the question put to you. Do not try to explain why the market went quiet; the quiet is the setting, \
not the subject. Ask instead: given what was published while it was coiling, and the state it coiled \
into, is the resolution predictable?

The exit day's own move is NOT evidence of direction. Measured on 214 of these episodes, the \
direction of the first bar out of compression has no relationship to where price is two weeks later. \
Do not extrapolate from it.

You are working in a historical replay: you are being asked to make the same real-time judgment call \
the live system would have made on a specific past date, using ONLY what was actually public \
knowledge as of that date (this replay deliberately doesn't use any invented or curated content).

You will be given: (1) the compression episode, told in three phases: PHASE A (when it began and how quiet it got), \
PHASE MIDDLE (how long it lasted, how price drifted, and every macro release published while it \
lasted -- each as a CHANGE from the prior print and as a SURPRISE in standard deviations of that \
series' usual move, not as a bare level), and PHASE B (the day it ended), (2) the replay's own candidate battery status as of this date \
(which candidates, if any, are 'accepted' as of TODAY in the simulation -- meaning they cleared the \
historical/backtest bar; this is not a claim of a live track record), (3) a short summary of this \
replay's trade history so far. Trades are never yours to open -- an unattended, purely mechanical scan \
already opens a live test the moment an accepted candidate's own trigger condition is met, with no LLM \
judgment involved. Your only job here is to decide whether this event suggests a genuinely new, \
untested pattern -- especially a SPECIFIC combination of what you were shown (an indicator reading, a \
recent macro release) rather than the event in isolation -- not covered by any existing candidate, \
using ONLY these whitelisted indicators: {proposable_indicators()}. Ground any compound hypothesis \
ONLY in indicator readings/releases you were actually given -- never invent one. A single clause is \
fine when nothing else in the given context looks relevant.

Return ONLY a JSON object with exactly these fields:
- "assessment": 1-2 sentences
- "recommended_action": one of "no_action", "propose_novel_test"
- "novel_condition_specs": null, or a list of ONE OR TWO specs, each \
{{"label": "...", "clauses": [{{"indicator": "...", "op": "<"/">"/"<="/">=", "threshold": <number>, \
"within_days": <integer 0-14, optional>}}, ...], "direction": "long"/"short", \
"coins": <optional list>, "outcome": "raw"/"market_relative" (optional)}}

AT MOST TWO CLAUSES PER SPEC -- one news/macro term and one market-state term. This is not a style \
preference, it is what can be measured: each extra clause divides the number of historical \
occurrences by roughly eight, and a three-clause condition has a median of 12 occurrences where \
{MIN_HISTORICAL_OCCURRENCES} are needed to test anything at all. A spec with three clauses is rejected by code.

PROPOSE TWO SPECS RATHER THAN ONE DEEPER CONDITION when the evidence supports more than one idea. \
An idea you would have written as "rate cut in the last 7 days AND funding negative AND RSI below \
30" should be sent as two: "rate cut in the last 7 days AND funding negative", and "rate cut in \
the last 7 days AND RSI below 30". The three-clause version is one hypothesis that almost certainly \
cannot be measured; the two are both measurable, and they are separately informative -- if only one \
survives, that is a finding the conjunction would have hidden.

The two must be GENUINELY DIFFERENT hypotheses, not one idea restated. Use your judgement about how \
markets work to choose two mechanisms you have real reason to think might each matter, rather than \
the same condition with a threshold nudged. Two specs that fire on the same days are checked for in \
code and the second is discarded, so a near-duplicate simply wastes the slot. One good spec is \
better than one good spec plus filler.

CHOOSE within_days FOR WHAT THE HYPOTHESIS MEANS, not for how often it fires. "The print came out \
today and the market was already oversold" and "a print came out this week, and the market is \
oversold now" are different claims about how an effect travels, and only you can say which one you \
mean. A lookback of 0 says the two things coincided; a lookback of K says the news came first and \
the market condition followed within K days.

Do not reach for a longer window to make a condition occur more often. Occurrences on consecutive \
days are one episode counted several times, and the gate that decides whether a condition can be \
tested counts EPISODES -- so a wider window buys almost no additional evidence, and the extra \
firings it produces are the same evidence repeated.

CONDITIONS MAY BE SEQUENCED, not just simultaneous. Each clause takes an optional "within_days" \
(0-14, default 0). 0 means "true on the day the condition fires"; K means "was true at any point in \
the last K days". This is what expresses an ORDERING rather than a coincidence, and the two are \
genuinely different hypotheses:
  - crash FIRST, then the release:  close_return_5d <= -0.10 with within_days=3, AND today's condition
  - release FIRST, then the move:   cpi_surprise >= 1 with within_days=2, AND today's condition
The lead-up table you are given exists precisely so you can tell these apart -- use it.

IS THIS EVENT ABOUT ONE COIN OR THE WHOLE MARKET? The two need different settings:
  - MARKET-WIDE (a CPI print, an FOMC decision): omit "coins", leave "outcome" as "raw". The whole \
    market moving together IS the effect; measuring each coin against the market would subtract the \
    very thing being tested.
  - COIN-SPECIFIC (something affecting one asset and not its peers): set "coins" to just that coin, \
    e.g. ["XRPUSDT"], and set "outcome" to "market_relative". About 54% of any coin's move is simply \
    the market's move, so removing it isolates what is specific to this one.
If unsure, omit both and the whole-market defaults apply.

KEEP IT WIDE, AND KEEP IT SHORT. The most common way a proposal fails here is not \
that the evidence contradicts it -- it is that the condition almost never occurred, so \
there is nothing to measure either way. A hypothesis that has never happened cannot be \
confirmed or denied.

Two rules, and the second matters more than the first:
  1. AT MOST 3 clauses, including the mandatory news/macro one. Four or more is \
     refused outright.
  2. Prefer MODERATE thresholds over dramatic ones. On this data, for \
     "macro day AND 5-day fall AND RSI below X":
         5-day fall < -20%, RSI < 40  ->    51 usable occurrences
         5-day fall < -20%, RSI < 50  ->    56
         5-day fall < -10%, RSI < 50  ->   220
         5-day fall <  -5%, RSI < 50  ->   508
     Widening the RSI barely moved it. Relaxing the price move from -20% to -10% \
     quadrupled it. A 20% five-day fall is a once-in-years event; asking for it \
     alongside anything else produces a condition that essentially never fires.

So: pick the ONE market-state term that carries your actual idea, set its threshold \
where it fires often enough to measure (a normal bad week, not a historic crash), and \
stop. If a second state term genuinely adds something, make it a LOOSE one -- a \
condition that is true often. Two clauses that fire regularly beat four that describe \
a single day in 2020 perfectly and never recur.

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


def _short_id(candidate: str) -> str:
    """Lazy import so this module keeps no import-time dependency on the bot."""
    from telegram.bot import short_id
    return short_id(candidate)


def _all_candidates_status_summary() -> str:
    """Every candidate ever tracked (all 6 static C1/C2/C6 variants, plus
    any dynamic ones discovered via "test it") and its latest status --
    rejected and watch included, not just the accepted subset
    `_battery_context()` uses for trade decisions. This is what a
    question like "what's been tested, anything accepted or dropped?"
    actually needs to answer completely. `status` is the raw, ongoing
    technical classification (accepted/watch/rejected); a separate
    "CONFIRMED at its checkpoint" tag is added only for candidates that
    were still accepted when they crossed a milestone -- see
    replay/status_history.py::mark_milestone_reported. Each line is
    prefixed with the candidate's 4-character id, the handle a reader
    types into /replay_details.

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
            # CONFIRMED, never "validated". The distinction is the project's own
            # and is documented at length: no reachable occurrence count
            # demonstrates an effect of interesting size, so the checkpoint
            # asserts persistence. Sonnet repeats the vocabulary it is given, and
            # given "validated" it wrote "validated" -- in a message that would
            # have gone in the README beside the claim that the word is never
            # used here.
            tags.append("CONFIRMED at its checkpoint"
                        if info.get("milestone_cleared")
                        else f"not confirmed at its {n_reached}-occurrence checkpoint")
        tag = f" ({', '.join(tags)})" if tags else ""
        # The 4-char id goes in every line: it is what a reader types to ask for
        # detail, and a Sonnet answer that names a 50-character condition without
        # it leaves them nothing to act on.
        lines.append(f"[{_short_id(name)}] {name}: "
                      f"{STATUS_PLAIN.get(info['status'], info['status'])}{tag}")
    if insufficient:
        listed = ", ".join(f"[{_short_id(n)}] {n}" for n in insufficient)
        lines.append(f"Insufficient data, not enough historical occurrences yet to test "
                      f"({len(insufficient)}): {listed}")
    return "\n".join(lines)


def _history_summary() -> str:
    log = state.load_trade_log()
    if not log:
        return "No live tests yet this replay."
    positive = sum(1 for t in log if t.get("status") == "closed" and t.get("forward_return", 0) > 0)
    return f"{len(log)} live test(s) opened so far this replay, {positive} resolved with a positive forward return."




def _macro_number(value: float, signed: bool = False) -> str:
    """Format a macro figure at a scale a reader can take in.

    These series do not share units: initial jobless claims are counts in the
    hundreds of thousands, CPI is an index, the Fed funds rate is a percentage.
    One format cannot serve all three -- "+14000.000" was being printed for a
    claims delta, three decimals of false precision on a number that is only
    ever reported in thousands."""
    sign = "+" if signed and value >= 0 else ""
    if abs(value) >= 10_000:
        return f"{sign}{value / 1000:,.0f}k"
    if abs(value) >= 100:
        return f"{sign}{value:,.0f}"
    return f"{sign}{value:.2f}".rstrip("0").rstrip(".") if value % 1 else f"{sign}{value:,.0f}"


def _macro_during(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Every macro release published between two dates, as DELTA and SURPRISE
    rather than as a level.

    A level ("CPI = 3.1") is close to useless to reason from: it needs the whole
    history to interpret. What moves a market is how a print compares with what
    was already known -- the change from the prior release, and how large that
    change is against how much this series usually moves. Both are already
    computed elsewhere in this project (`latest_release_with_prior`,
    `surprise_series`); this presents them together, per release, over a window.

    The surprise here is measured against the SERIES' OWN recent behaviour, not
    against an analyst consensus -- this project has no consensus data source,
    and saying so plainly is better than letting "surprise" be read as the
    market's standard meaning."""
    from candidates.macro_vintage import (MACRO_SERIES, latest_release_with_prior,
                                           release_dates, surprise_series)

    lines = []
    for key, label in MACRO_SERIES.items():
        dates = release_dates(key, start, end, new_periods_only=True)
        if len(dates) == 0:
            continue
        surprises = surprise_series(key, pd.DatetimeIndex(dates))
        for d in dates:
            rel = latest_release_with_prior(key, d)
            if rel is None:
                continue
            delta = (rel["value"] - rel["prior_value"]) if rel.get("prior_value") is not None else None
            z = surprises.get(d)
            parts = [f"  {d.date()} {label}: {_macro_number(rel['value'])}"]
            if delta is not None:
                parts.append(f"change vs prior {_macro_number(delta, signed=True)}")
            if z is not None and z == z:
                parts.append(f"surprise {z:+.1f} sd vs this series' usual move")
            lines.append(", ".join(parts))
    return "\n".join(lines) if lines else "  (no macro releases in this window)"


def format_compression_event(symbol: str, episode: dict) -> str:
    """The trigger's event description: a volatility-compression episode that has
    just ended, told as the story of how it formed.

    Structured as three phases because the model is being asked to explain a
    RESOLUTION, and a resolution has a run-up. What was published while the
    market was coiling is the candidate cause; the state it coiled into is the
    circumstance. Both are needed, and neither is visible from a single day's
    snapshot.

    Deliberately does NOT characterise the breakout direction beyond reporting
    the day's move. Measured on 214 episodes, the direction of the exit bar has
    no relationship to the direction 14 days later (rho = -0.051, p = 0.46), so
    presenting it as a signal would invite the model to extrapolate from noise --
    and the direction is precisely what it is being asked to reason about from
    the macro and market evidence instead."""
    a, b = episode["a_date"], episode["b_date"]
    return (
        f"VOLATILITY COMPRESSION RESOLVED: {symbol}\n"
        f"PHASE A -- compression began {a.date()} "
        f"(volatility {episode['z_at_a']:.2f} sd below this coin's own normal)\n"
        f"PHASE MIDDLE -- it stayed compressed for {episode['duration']} day(s); "
        f"price moved {episode['squeeze_return']:+.2%} across the squeeze.\n"
        f"Macro published while the market was coiling:\n{_macro_during(a, b)}\n"
        f"PHASE B -- compression ended {b.date()}; that day's move was "
        f"{episode['b_return']:+.2%}. Whether this resolves into a sustained move, "
        f"and in which direction, is what the evidence has to explain -- the exit "
        f"bar's own direction carries no information about it."
    )


def indicator_values(coin: str, as_of, names: "tuple[str, ...]") -> dict:
    """Named indicator readings as FLOATS, dated to `as_of`.

    `build_indicator_snapshot` renders every whitelisted indicator as prose for a
    model prompt; this returns the handful a human report actually formats, so
    the report is not parsing a string meant for something else. Never raises: a
    missing or NaN reading comes back absent, and the caller omits that line
    rather than printing a number it does not have."""
    from candidates.data_loading import load_daily, load_funding
    from llm_pipeline.novel_condition_tester import SUPPORTED_INDICATORS

    out: dict = {}
    try:
        daily = load_daily(coin)
        if as_of is not None:
            daily = daily.loc[:as_of]
        if len(daily) == 0:
            return out
        funding = load_funding(coin)
        if funding is not None and as_of is not None:
            funding = funding.loc[:as_of]
        for name in names:
            fn = SUPPORTED_INDICATORS.get(name)
            if fn is None:
                continue
            try:
                # TWO positional args only, exactly as build_indicator_snapshot
                # calls them. The functions in SUPPORTED_INDICATORS do NOT share
                # a signature past that point -- `_rsi` takes (df, funding,
                # scale, window, symbol) while the surprise series take
                # (df, funding, scale, symbol) -- so passing the coin as the
                # fourth positional set window="DOGEUSDT" and every reading came
                # back empty. Silently: the except below swallowed it.
                series = fn(daily, funding).dropna()
            except Exception:
                continue
            if len(series):
                value = float(series.iloc[-1])
                if value == value:
                    out[name] = value
    except Exception:
        return out
    return out


def format_compression_report(symbol: str, episode: dict, specs: "list[dict] | None" = None) -> str:
    """The HUMAN-facing rendering of a compression exit.

    Deliberately separate from `format_compression_event`, which builds the same
    episode as a PROMPT for Sonnet. The two have different jobs -- one has to be
    reasoned from, the other scanned -- and merging them would mean every change
    to how a human reads this silently rewrites what the model is asked."""
    a, b = episode["a_date"], episode["b_date"]
    lines = [f"<b>EVENT ALERT: VOLATILITY COMPRESSION RESOLVED</b>",
             f"Asset: <b>{escape_html(symbol)}</b>",
             f"Period: {a.date()} to {b.date()} ({episode['duration']} Days Coiling)",
             "", "<b>--- SQUEEZE METRICS ---</b>",
             f"• Start Volatility: {episode['z_at_a']:.2f} SD below normal",
             f"• Mid-Squeeze Drift: {episode['squeeze_return']:+.2%}",
             f"• Exit Bar Move ({b.date()}): {episode['b_return']:+.2%}"]
    # The readings shown are the ones SONNET actually reasoned from -- the
    # market-state indicators appearing in the conditions it proposed -- not a
    # fixed pair chosen here. A hardcoded RSI/Bollinger line would print two
    # numbers unrelated to the hypothesis underneath it whenever Sonnet reached
    # for funding, volume or efficiency instead, which is most of the time.
    # Event indicators are excluded: their values are already the MACRO section.
    wanted: list[str] = []
    for spec in (specs or []):
        for clause in spec.get("clauses", []):
            name = clause.get("indicator")
            if name and name not in wanted and name not in EVENT_INDICATORS:
                wanted.append(name)
    readings = indicator_values(symbol, b, tuple(wanted)) if wanted else {}
    if readings:
        # Short label here, full plain-English name on the Logic line below. The
        # names in INDICATOR_PLAIN_NAMES carry a parenthetical gloss ("14-day RSI
        # (momentum, 0-100 scale)") which earns its length where the condition is
        # spelled out and swamps a metrics line where four of them sit side by side.
        bits = [f"{INDICATOR_PLAIN_NAMES.get(n, n).split(' (')[0]}: {readings[n]:.2f}"
                for n in wanted if n in readings]
        # Only named when RSI is among what was proposed, and only at the
        # textbook thresholds: a label is a claim, and "oversold" asserted from
        # a reading of 49 -- or from an indicator that says nothing about being
        # oversold at all -- would be one this project has not earned.
        rsi = readings.get("rsi_14d")
        state_word = ""
        if rsi is not None:
            state_word = ("Oversold " if rsi <= 35 else "Overbought " if rsi >= 65 else "Neutral ")
        lines.append(f"• Post-Squeeze State: {state_word}({' | '.join(bits)})".replace(": (", ": ("))
        if not state_word:
            lines[-1] = f"• Post-Squeeze State: {' | '.join(bits)}"
    lines += ["", "<b>--- MACRO CONTEXT DURING SQUEEZE ---</b>", _macro_during(a, b).rstrip()]
    # The exit bar's own direction is reported and NOT characterised: measured on
    # 214 episodes its direction has no relationship to where price is 14 days
    # later (rho = -0.051, p = 0.46), so calling it bullish or bearish here would
    # invite exactly the extrapolation the trigger design rules out.
    return "\n".join(lines)




def judge_event(event_description: str, client: Anthropic, as_of: pd.Timestamp | None = None,
                 coin: str | None = None, model: str = SONNET_MODEL) -> dict:
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
        # `model` is a parameter only so the same prompt, the same context and the
        # same scoring can be pointed at a cheaper model for comparison
        # (forecast/model_comparison.py). The replay itself never overrides it.
        # 4000, raised from 2000 on 2026-09-01. Measured: Sonnet's mean output on
        # this prompt is 1,438 tokens, so a 2,000 cap sat barely above the
        # average and 35% of calls (14 of 40) hit it -- returning a thinking
        # block with no text at all, which `extract_text` can only reject. Haiku
        # on the same prompt averages 276 and never hit it.
        #
        # The cap grew because the prompt did: a three-phase episode narrative,
        # a clause cap, two proposals, and the within_days explanation. Raising
        # it costs nothing unless the model actually uses the room -- max_tokens
        # is a ceiling, not a request, which is the same distinction that made
        # an early cost forecast in this project 4x too high.
        model=model, max_tokens=4000, system=cached_system(REPLAY_SYSTEM_PROMPT),
        messages=[{"role": "user", "content": user_content}],
    )
    # Tagged `replay.compression`, the trigger that actually fires. It read
    # `replay.shock`/`replay.macro` until 2026-09-02 -- names from the two
    # triggers removed months earlier, so the historical entries under those
    # keys in llm_usage.json belong to the OLD architecture and must not be
    # read as the current design's cost. The `coin` branch is kept because a
    # coin-less judgment is still expressible, not because macro days trigger
    # anything any more.
    _usage.record(response, ("replay.compression" if coin else "replay.judgment")
                            + ("" if model == SONNET_MODEL else f".{model.split('-')[1]}"), model)
    return json.loads(_strip_fences(extract_text(response)))


MARKET_CHECK_SYSTEM_PROMPT = """You are a market-check assistant for a crypto pattern-discovery system \
(NOT a trading system -- no funded position is ever opened; "accepted" only means a candidate's own \
trigger opens an observational live test), answering a question about a historical replay's current \
simulated state (a specific past date, walked forward day by day -- not live). Given real technical/ \
portfolio state as of that simulated date and the live candidate battery context, answer the user's \
question in 2-4 sentences. Cite only the numbers given to you in the state below -- never invent a \
price, a percentage, or a live-test detail. The reader may not know this project's internal vocabulary \
(status codes like "insufficient_data", "watch") -- briefly explain any such term you use in plain \
language rather than stating it bare, the way you'd explain a piece of jargon to someone unfamiliar \
with the system.

ACCEPTED AND CONFIRMED ARE DIFFERENT CLAIMS AND MUST NEVER BE TREATED AS SYNONYMS. "accepted" means \
the HISTORICAL backtest cleared every gate (significance, direction, concentration, risk path) -- it \
says nothing about what happened afterwards. "confirmed" means the condition kept occurring AFTER the \
hypothesis was written down and still held 'accepted' when it reached a checkpoint of 20 such \
occurrences -- it is persistence, re-earned at each checkpoint, never a permanent badge and never a \
proof. A candidate can be accepted and never confirmed, or reach a checkpoint while in 'watch' and so \
not earn the word. Writing "confirmed (that is, accepted)" or otherwise equating them is wrong. \
Neither word means an effect has been demonstrated: at these horizons a demonstration needs occurrences \
in the hundreds, so when you cite a count, cite what would be required alongside it.

WHEN YOU NAME A CANDIDATE, GIVE ITS 4-CHARACTER ID TOO. Every candidate in the state below is \
listed as "[id] name". These conditions have long descriptive names and the id is what a reader \
types to ask for detail, so naming one without it leaves them nothing to act on. Write it as \
`name` (id), e.g. `hawkish_claims_then_volume_spike` (44fb).

NEVER WRITE "VALIDATED". The word this project uses is CONFIRMED, and the difference is the point: \
at these horizons a conclusive test needs occurrences in the hundreds, so a checkpoint asserts that \
a condition kept occurring and still passes on the enlarged sample -- persistence, not proof. \
"Validated" claims something no reachable count here demonstrates.

If nothing in the given state answers the question, say so plainly rather than guessing."""


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
        model=SONNET_MODEL, max_tokens=4000, system=MARKET_CHECK_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"USER QUESTION: {question}\n\nSTATE:\n{snapshot}\n\n{context}"}],
    )
    _usage.record(response, "replay.market_check", SONNET_MODEL)
    return escape_html(extract_text(response))


def format_telegram_message(as_of, event_description: str, assessment: dict,
                             episode: dict | None = None, symbol: str | None = None) -> str:
    """The human-facing proposal message, in scannable sections.

    Deliberately formatted to look exactly like what the live pipeline itself
    would send (llm_pipeline/haiku_sonnet_pipeline.py::format_compression_message)
    -- no "this is a simulation" framing inside the message body. That disclosure
    belongs in the surrounding write-up this replay is presented in, not baked
    into every message, the way a UI mockup doesn't stamp "MOCKUP" on every
    button. The date is still shown, since messages fire in rapid succession here
    but represent dates spread across years.

    `episode`/`symbol` are optional: with them the event renders as the sectioned
    EVENT ALERT report, without them (a promoted parked proposal, which has no
    episode of its own) it falls back to the plain description it was handed."""
    action = assessment["recommended_action"]
    specs = assessment.get("novel_condition_specs")
    if specs is None and assessment.get("novel_condition_spec"):
        specs = [assessment["novel_condition_spec"]]

    # Resolved BEFORE the header: the report's indicator readings are the ones
    # the proposals actually reference, so it cannot be built until they are known.
    if episode is not None and symbol is not None:
        header = format_compression_report(symbol, episode, specs)
    else:
        header = f"<b>Event:</b> {escape_html(event_description)}"
    base = f"<b>{as_of.date()}</b>\n\n{header}\n\n"

    assessment_text = escape_html(assessment["assessment"])
    if action == "propose_novel_test" and specs:
        n = len(specs)
        base += (f"<b>--- ASSESSMENT ---</b>\n{assessment_text}\n"
                 f"Sonnet generated <b>{n} testable hypothes{'es' if n > 1 else 'is'}</b>.\n")
        if n > 1:
            # Two measurable halves rather than one deeper conjunction that could
            # not be measured at all -- and one approval covers the SET, since
            # splitting an idea only helps if both halves are actually tested.
            base += ("<i>Two conditions rather than one deeper combination: each extra clause "
                     "divides historical occurrences by roughly eight. One button approves both.</i>\n")
        for i, spec in enumerate(specs, 1):
            base += (f"\n\n<b>PROPOSAL {i}: {escape_html(spec['label'])}</b>\n"
                     f"Status: <b>PROPOSED</b> (Awaiting Human Gating)\n\n"
                     f"<b>--- CONDITION &amp; PARAMETERS ---</b>\n"
                     f"• Logic: {format_spec_clauses(spec)} → {escape_html(spec['direction'].upper())}\n")
            # The tested condition is not always the proposed one. When thresholds
            # were loosened to reach a measurable sample, the human approving has
            # to see that BEFORE pressing the button -- the clause line above
            # shows the new numbers but not that they changed, and a silent
            # substitution would make the approval meaningless.
            if spec.get("relaxed_from"):
                base += (f"• Thresholds: {escape_html(spec['relaxed_from'])}\n"
                         f"  <i>*Note: nearest testable version of the proposed hypothesis, "
                         f"not the original one.</i>\n")
        base += (f"\n\n<b>--- ACTION REQUIRED (HUMAN-IN-THE-LOOP) ---</b>\n"
                 # Worded from what the code actually does. NOT "parks for live
                 # tracking": in this codebase PARKED means a proposal that never
                 # reached a human for want of occurrences, which is the opposite
                 # of what pressing this does. And NOT "purges from the pipeline":
                 # discard_pending_test clears the pending slot and records
                 # nothing, so nothing is purged -- there is no registry entry to
                 # remove, and the condition remains re-proposable later.
                 f"[ <b>Test It</b> ] → Runs the full walk-forward backtest on history up to this "
                 f"date, registers {'both conditions' if n > 1 else 'the condition'}, and tracks "
                 f"{'them' if n > 1 else 'it'} forward as observational live tests. No capital, ever.\n"
                 f"[ <b>Don't Test It</b> ] → Dismisses {'them' if n > 1 else 'it'} untested and "
                 f"resumes the replay. Nothing is recorded, so the same idea can surface again later.")
    else:
        base += f"<b>--- ASSESSMENT ---</b>\n{assessment_text}\n\n<i>No action taken -- logged, nothing further needed.</i>"
    return base
