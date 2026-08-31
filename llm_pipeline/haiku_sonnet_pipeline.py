"""Haiku Scout -> Sonnet Strategist -> Telegram. Haiku classifies each
headline against the live candidate battery; only genuinely unmatched,
significant conditions escalate to Sonnet.

Sonnet's live role is narrow, by design: (1) decide whether a headline/
shock suggests a genuinely new, untested pattern worth a human's "test
it" (`propose_novel_test`), and (2) answer natural-language questions
about system state. It does NOT decide to open a trade -- that's a
mechanical, unattended scan (see scheduler/ and replay/engine.py) that
fires identically for every occurrence of an `accepted` candidate's own
trigger condition, with no per-event LLM judgment involved. Removing
that decision from Sonnet's hands isn't a loss of capability: a routine
trade never needed Sonnet's opinion in the first place -- the anchors
and the trigger condition were already fixed by the backtest, so an LLM
judgment call per occurrence would only add unattributable variance,
not real decision-making, to something already fully determined by
already-computed statistics.
"""
from __future__ import annotations

import json
import os

import requests
from anthropic import Anthropic
from dotenv import load_dotenv

from candidates.macro_vintage import recent_releases_summary
from data_ingestion.news_sentiment.cryptocompare_fetcher import fetch_cryptocompare_news
from llm_pipeline.context_builder import build_context_summary, build_technical_snapshot
from llm_pipeline.novel_condition_tester import (
    INDICATOR_PLAIN_NAMES, OPERATOR_PLAIN, ConditionSpec, build_indicator_leadup, proposable_indicators,
    build_indicator_snapshot, clause_from_dict,
)
from llm_pipeline.pending_tests import push_pending_test
from llm_pipeline.shock_detector import scan_for_shocks
from llm_pipeline import usage as _usage

HAIKU_MODEL = "claude-haiku-4-5"
SONNET_MODEL = "claude-sonnet-5"
ESCALATION_MAGNITUDE_THRESHOLD = 4
FREQTRADE_DB_PATH = os.environ.get("FREQTRADE_DB_PATH", "execution/tradesv3.sqlite")
SHOCK_SCAN_COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LTCUSDT"]

HAIKU_SYSTEM_PROMPT = """You are a crypto market news sentiment extractor. Given a batch of \
headlines, return a JSON array where each item has exactly these fields:
- "headline": the original headline (verbatim, truncated to 100 chars)
- "asset": the primary crypto asset affected (e.g. "BTC", "ETH", "MARKET" if broad/unclear)
- "sentiment": one of "bullish", "bearish", "neutral"
- "magnitude": integer 1-5, how market-moving this is likely to be (5 = major, 1 = negligible)
- "event_type": one of "regulatory", "macro", "hack_exploit", "adoption", "market_structure", "other"

Return ONLY the JSON array, no prose, no markdown fences."""

SONNET_SYSTEM_PROMPT = f"""You are a quantitative researcher, not a trader. Your subject is a single question: \
does a real-world macro or news EVENT produce a measurable, repeatable change in crypto prices \
over the following days? Nothing here is a trading system -- no position is ever opened, no \
money is ever at risk, and there is no entry signal to find.

That distinction decides what a good answer looks like. A chart setup -- oversold RSI, a \
Bollinger touch, a volume spike, a volatility shock -- is NOT a hypothesis in this project, \
however well it would work as a trade. Those readings are CONTEXT: they describe the state the \
market happened to be in when the event landed. The event is the subject; the market state \
merely says under what circumstances you think it matters. If your idea would still make sense \
with the macro release deleted from it, it is a chart pattern and does not belong here.

The instinct to reach for the technical indicators first is the right one for a market \
strategist and the wrong one for this task. Start from the EVENT -- what was published, how far \
it moved from what was expected -- and only then ask which market conditions would make its \
effect visible.

You will be \
given: (1) a flagged news headline, (2) real current readings on every whitelisted indicator (this \
coin's if the headline names one, every tracked coin's if it's broad/unclear) and real macro \
releases from the last several days -- a pattern doesn't require a volatility shock to exist, so \
this context is given for every headline, not only shock-triggered ones, (3) real technical/ \
portfolio state, (4) the live status of this system's candidate battery, including which candidates \
currently carry 'accepted' status and their coin/direction ('accepted' means it cleared the \
historical/backtest bar -- it is not a claim of a live track record). Ground your reasoning ONLY in \
real numbers/releases you were actually given -- never invent an indicator reading or a release you \
weren't shown. Trades are never yours to open -- an unattended, purely mechanical scan already fires \
a live test the moment an accepted candidate's own trigger condition is met, with no LLM judgment \
involved. Your only two jobs are:

1. If this headline suggests a genuinely new, untested pattern -- not covered by any existing \
candidate and not already logged as rejected -- propose a novel-condition test (this DOES wait \
for human approval, since it spends real compute testing something unproven), using ONLY one of \
these whitelisted indicators (nothing else is buildable): {proposable_indicators()}. Ground any \
compound hypothesis ONLY in evidence you were actually given -- never invent an indicator reading \
or a release you weren't shown.
2. Otherwise, if there's nothing new to propose, say so plainly.

HARD REQUIREMENT -- every proposal MUST contain at least one of these event indicators: \
cpi_surprise, rate_surprise, jobless_claims_surprise. This is not a preference, it is \
what this system exists to test: whether specific MARKET CONDITIONS combined with a REAL-WORLD EVENT \
produce a repeatable pattern. A condition built only from price/volume/funding indicators is a chart \
pattern -- it does not answer that question, and a spec without an event clause is REJECTED by code \
before it is ever tested. A violent price move is a MARKET event, not news, and \
"the price moved, then the price did something" is the tautology this rule exists to exclude.

The volatility shock itself is NOT available as a clause. It is the thing your hypothesis has to \
EXPLAIN, not part of the explanation -- you are being asked about this moment precisely because a \
shock occurred, so every proposal already sits on one and naming it adds nothing. Describe what the \
market looked like and what had been published; the shock is the outcome, not the setup.

Your label must also describe what the clauses actually test. Do not name a condition "post-CPI ..." \
unless a CPI/macro clause is genuinely in it -- that is checked in code too.

Return ONLY a JSON object with exactly these fields:
- "assessment": 1-2 sentences
- "recommended_action": one of "no_action", "propose_novel_test"
- "novel_condition_spec": null, or {{"label": "...", "clauses": [{{"indicator": "...", \
  "op": "<"/">"/"<="/">=", "threshold": <number>, "within_days": <integer 0-14, optional>}}, ...], \
  "direction": "long"/"short", "coins": <optional list, see below>, \
  "outcome": "raw"/"market_relative" (optional, default "raw", see below)}} \
  (one clause is fine for a simple condition; multiple clauses are ANDed together for a compound \
  one, e.g. an oversold technical reading combined with a macro surprise -- use as many as the \
  actual evidence supports, not for its own sake)

IMPORTANT -- is this event about ONE coin, or about the whole market? The two need different \
settings, and getting it wrong wastes the test:
  - MARKET-WIDE (a CPI print, an FOMC decision, a broad risk-off move): omit "coins", and leave \
    "outcome" as "raw". The whole market moving together IS the effect here; measuring each coin \
    against the market would subtract the very thing being tested and guarantee a null result.
  - COIN-SPECIFIC (a lawsuit against one issuer, an exchange listing or delisting, a protocol \
    incident): set "coins" to just the affected coin(s), e.g. ["XRPUSDT"], and set "outcome" to \
    "market_relative". Testing an XRP-specific claim on DOGE adds noise rather than evidence, and \
    measuring the coin against the market isolates what is specific to it -- roughly 54% of any \
    coin's move is simply the market's move, so removing it removes mostly noise.
Only name coins in "coins" if the event genuinely is specific to them. If you are unsure, omit it.

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

IMPORTANT -- conditions may be SEQUENCED, not just simultaneous. Each clause takes an optional \
"within_days" (integer, 0-14, default 0). 0 means "true on the day the condition fires"; K means \
"was true at any point in the last K days". This is what lets you express an ORDERING rather than a \
coincidence, and the two are genuinely different hypotheses:
  - crash FIRST, then the news:  close_return_5d <= -0.10 with within_days=3, AND today's condition
  - news FIRST, then the move:   cpi_surprise >= 1 with within_days=2, AND today's condition
  - both on the same day:        leave within_days at 0 on both
You are shown a day-by-day LEAD-UP table (the last several days of key indicators, oldest first) \
precisely so you can tell these apart. Read it before choosing: if the violent move is two rows \
above the last one, the honest hypothesis is a sequenced one, not a same-day conjunction.

Prefer a hypothesis where the EVENT (a macro release, a shock) is combined with MARKET STATE \
(RSI, Bollinger position, volume, funding). A proposal built only from market-state indicators is \
a chart pattern, not the market-conditions-plus-event question this system exists to answer -- and \
one built only from an event term is tested against ordinary days rather than against the same \
market state without the event, which is a much weaker claim.

No prose, no markdown fences, just the JSON object."""

SHOCK_SYSTEM_PROMPT = f"""You are a quantitative researcher, not a trader. Your subject is a single question: \
does a real-world macro or news EVENT produce a measurable, repeatable change in crypto prices \
over the following days? Nothing here is a trading system -- no position is ever opened, no \
money is ever at risk, and there is no entry signal to find.

That distinction decides what a good answer looks like. A chart setup -- oversold RSI, a \
Bollinger touch, a volume spike, a volatility shock -- is NOT a hypothesis in this project, \
however well it would work as a trade. Those readings are CONTEXT: they describe the state the \
market happened to be in when the event landed. The event is the subject; the market state \
merely says under what circumstances you think it matters. If your idea would still make sense \
with the macro release deleted from it, it is a chart pattern and does not belong here.

The instinct to reach for the technical indicators first is the right one for a market \
strategist and the wrong one for this task. Start from the EVENT -- what was published, how far \
it moved from what was expected -- and only then ask which market conditions would make its \
effect visible.

Here the trigger that brought this to you IS a market event -- a coin's \
short-term realized volatility has just spiked into roughly the top ~2% most extreme episodes for \
that coin (this project's own Phase 1 methodology excludes exactly this population from the static \
candidate battery's fitting, because a handful of crashes shouldn't distort barriers meant for \
ordinary conditions). \
This is deliberately the harder case a fixed rule set can't pre-classify -- you're being asked \
whether this specific instance is worth reacting to at all, not whether shocks in general are \
tradeable.

You will be given the shock's real, computed severity (a z-score) and direction (crash/surge), \
this coin's CURRENT real readings on every whitelisted indicator (technical: RSI, ATR%, Donchian \
channel position, Bollinger %B, volume z-score, trend efficiency; financial: funding-rate z-score, \
whether today is a macro-release day), any real macro releases from the last few days, and recent \
real news headlines, plus real technical/portfolio state and the candidate battery's status. Ground \
your reasoning ONLY in these real numbers/headlines you were actually given -- never invent an \
indicator reading, a headline, or a release you weren't shown. Recommend one of:
- "no_action": noise, not worth a human's attention.
- "propose_novel_test": worth finding out if a SPECIFIC combination of what you were just given, \
  historically, shows a real reversal or continuation pattern -- e.g. you notice the shock \
  coincided with RSI already deeply oversold AND a hotter-than-prior inflation print in the last \
  few days, so you propose testing exactly that combination, not the shock in isolation. Don't force a \
  compound story where the evidence doesn't support one. Use ONLY indicators you were \
  actually shown a reading for. Example: {{"label": "hot_cpi_into_oversold_<coin>", "clauses": \
  [{{"indicator": "cpi_surprise", "op": ">=", "threshold": 1.0, "within_days": 2}}, \
  {{"indicator": "rsi_14d", "op": "<", "threshold": 30}}], "direction": "long" or "short"}} -- note \
  "within_days" (integer 0-14, default 0): 0 = true on the day the condition fires, K = true at any \
  point in the last K days. Use it to express that the RELEASE CAME FIRST and something else followed, \
  which is a different hypothesis from both happening on the same day. The day-by-day LEAD-UP table \
  you are shown exists so you can tell which of the two you are actually looking at. If the human approves and it's \
  accepted, the resulting anchors are used for a LIVE trade on THIS occurrence, tagged separately \
  from routine trades so how the system actually performed reacting to real shocks, in real time, \
  can be measured on its own.


HARD REQUIREMENT -- every proposal MUST contain at least one of these event indicators: \
cpi_surprise, rate_surprise, jobless_claims_surprise. This is not a preference, it is \
what this system exists to test: whether specific MARKET CONDITIONS combined with a REAL-WORLD EVENT \
produce a repeatable pattern. A condition built only from price/volume/funding indicators is a chart \
pattern -- it does not answer that question, and a spec without an event clause is REJECTED by code \
before it is ever tested. A violent price move is a MARKET event, not news, and \
"the price moved, then the price did something" is the tautology this rule exists to exclude.

The volatility shock itself is NOT available as a clause. It is the thing your hypothesis has to \
EXPLAIN, not part of the explanation -- you are being asked about this moment precisely because a \
shock occurred, so every proposal already sits on one and naming it adds nothing. Describe what the \
market looked like and what had been published; the shock is the outcome, not the setup.

Your label must also describe what the clauses actually test. Do not name a condition "post-CPI ..." \
unless a CPI/macro clause is genuinely in it -- that is checked in code too.

TP/SL are never yours to set -- they only ever come from an accepted anchor set, never invented \
here. Return ONLY a JSON object: "assessment" (1-2 sentences), "recommended_action" \
("no_action"/"propose_novel_test"), "novel_condition_spec" (null or the spec above). No prose, no \
markdown fences."""

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def escape_html(text: str) -> str:
    """The only three characters Telegram's HTML parse_mode actually
    requires escaping -- far fewer sharp edges than MarkdownV2 (which is
    why this project uses HTML for any formatted message, not Markdown:
    a stray '_' or '*' in LLM-generated free text broke MarkdownV2
    parsing outright in an earlier version of this pipeline)."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_spec_clauses(spec: dict) -> str:
    """Renders a raw novel_condition_spec dict's (possibly multi-clause)
    'clauses' list as one readable string, e.g. '14-day RSI (momentum,
    0-100 scale) below 30 AND shock z-score (how extreme today's price
    move is vs. this coin's own history) at least 2.0' -- shared by every
    message (production and replay) that shows a proposed or tested
    condition to a human, so a compound spec is never silently rendered
    as if it only had one clause, the indicator name is never shown as a
    raw variable name a non-technical reader couldn't parse (see
    INDICATOR_PLAIN_NAMES), and the comparison is never a raw "<"/">"
    symbol (see OPERATOR_PLAIN's own docstring for why -- a real,
    observed Telegram rendering bug, not just a style choice)."""
    from llm_pipeline.novel_condition_tester import _within_phrase
    return " AND ".join(
        f"{escape_html(INDICATOR_PLAIN_NAMES.get(c['indicator'], c['indicator']))} {OPERATOR_PLAIN.get(c['op'], c['op'])} {c['threshold']}"
        f"{_within_phrase(int(c.get('within_days') or 0))}"
        for c in spec["clauses"]
    )


# Minimum cacheable prefix: 1024 tokens for Sonnet, 2048 for Haiku. Measured with
# the API's own count_tokens rather than guessed:
#     SONNET_SYSTEM_PROMPT   2408   cached
#     REPLAY_SYSTEM_PROMPT   1575   cached
#     SHOCK_SYSTEM_PROMPT    1399   cached
#     MARKET_CHECK_PROMPT     316   too short
#     HAIKU_SYSTEM_PROMPT     187   too short (and Haiku's floor is 2048)
# The short ones are deliberately NOT marked: below the floor the breakpoint is
# silently ignored, which would leave code that looks cached and isn't.
CACHE_MIN_TOKENS_SONNET = 1024


def cached_system(prompt: str) -> list[dict]:
    """The system prompt as a cache breakpoint.

    The breakpoint goes HERE, on the system block, and never on the user
    message. Cache prefixes are built tools -> system -> messages, so the system
    block is the last position identical across calls: the user content carries
    the date, the coin and the indicator readings and differs every single time.
    Marking the varying block instead is the documented classic mistake -- every
    request would compute a new prefix hash, find no prior entry to read, and pay
    for a fresh cache WRITE at 1.25x forever, which is worse than not caching.

    One breakpoint is enough here. There is no growing conversation to push it
    out of the 20-block lookback window: every call is a fresh single-turn
    request with the same static system prefix.
    """
    return [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]


def extract_text(response) -> str:
    """A response's content blocks aren't always [text] -- a model can
    emit a ThinkingBlock (or other non-text block) before its actual
    answer, so content[0] is not reliably the text. Finds the first block
    that actually has text instead of assuming position."""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise ValueError(f"No text block in response.content: {[getattr(b, 'type', type(b)) for b in response.content]}")


def haiku_scout(articles: list[dict], client: Anthropic) -> list[dict]:
    if not articles:
        return []
    headlines_block = "\n".join(f"- {a['headline']}" for a in articles)
    response = client.messages.create(
        model=HAIKU_MODEL, max_tokens=2048, system=HAIKU_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": headlines_block}],
    )
    _usage.record(response, "prod.haiku_scout", HAIKU_MODEL)
    return json.loads(_strip_fences(extract_text(response)))


def sonnet_strategist(flagged: dict, client: Anthropic) -> dict:
    """A pattern doesn't require a shock to exist -- Sonnet's discovery
    role (propose_novel_test) is never limited to shock-triggered
    events, so a routine headline gets the same real indicator/macro
    grounding a shock does: this coin's current readings if the headline
    names one, every tracked coin's if it's broad/unclear ("MARKET"),
    plus recent real macro releases -- never just the headline text
    alone."""
    technical_snapshot = build_technical_snapshot(flagged.get("asset", "MARKET"), FREQTRADE_DB_PATH)
    context_summary = build_context_summary()
    coin = _asset_to_coin(flagged.get("asset", ""))
    if coin is not None:
        # Snapshot AND the run-up: an ordered hypothesis (crash three days ago,
        # news today) is unreasonable to ask for from a single instant's numbers.
        indicator_snapshot = build_indicator_snapshot(coin) + "\n\n" + build_indicator_leadup(coin)
    else:
        indicator_snapshot = "\n".join(build_indicator_snapshot(c) for c in SHOCK_SCAN_COINS)
    macro_summary = recent_releases_summary()
    user_content = (
        f"HEADLINE: {flagged['headline']} (asset={flagged['asset']}, "
        f"sentiment={flagged['sentiment']}, magnitude={flagged['magnitude']}, "
        f"event_type={flagged['event_type']})\n\n"
        f"INDICATOR SNAPSHOT:\n{indicator_snapshot}\n\n"
        f"RECENT MACRO RELEASES (last 10 days):\n{macro_summary}\n\n"
        f"TECHNICAL SNAPSHOT:\n{technical_snapshot}\n\n"
        f"CANDIDATE BATTERY CONTEXT:\n{context_summary}"
    )
    response = client.messages.create(
        # 2000, not 700 -- see replay/judgment.py::judge_event's comment: observed
        # live, the model emits a thinking block even though `thinking` is never
        # requested, and 700 sometimes left no budget for the actual JSON answer.
        model=SONNET_MODEL, max_tokens=2000, system=cached_system(SONNET_SYSTEM_PROMPT),
        messages=[{"role": "user", "content": user_content}],
    )
    _usage.record(response, "prod.sonnet_strategist", SONNET_MODEL)
    return json.loads(_strip_fences(extract_text(response)))


def _recent_headlines_summary(limit: int = 5) -> str:
    """A few of the most recent real news headlines -- best-effort: a
    fetch failure here must not block the shock judgment itself, since
    Sonnet can still reason from indicators/macro alone."""
    try:
        articles = fetch_cryptocompare_news(limit=limit)
    except Exception as e:
        return f"Headlines unavailable ({e})."
    if not articles:
        return "No recent headlines available."
    return "\n".join(f"- {a['headline']} ({a['published_at']})" for a in articles[:limit])


def sonnet_shock_response(shock: dict, client: Anthropic) -> dict:
    technical_snapshot = build_technical_snapshot(shock["symbol"], FREQTRADE_DB_PATH)
    context_summary = build_context_summary()
    indicator_snapshot = (build_indicator_snapshot(shock["symbol"]) + "\n\n"
                          + build_indicator_leadup(shock["symbol"]))
    macro_summary = recent_releases_summary()
    headlines_summary = _recent_headlines_summary()
    user_content = (
        f"SHOCK DETECTED: {shock['symbol']}, direction={shock['direction']}, "
        f"shock_z={shock['shock_z']:.2f}, latest_return={shock.get('latest_return')}\n\n"
        f"INDICATOR SNAPSHOT:\n{indicator_snapshot}\n\n"
        f"RECENT MACRO RELEASES (last 10 days):\n{macro_summary}\n\n"
        f"RECENT NEWS HEADLINES:\n{headlines_summary}\n\n"
        f"TECHNICAL SNAPSHOT:\n{technical_snapshot}\n\n"
        f"CANDIDATE BATTERY CONTEXT:\n{context_summary}"
    )
    response = client.messages.create(
        # 2000, not 700 -- see sonnet_strategist's comment above / replay/judgment.py.
        model=SONNET_MODEL, max_tokens=2000, system=cached_system(SHOCK_SYSTEM_PROMPT),
        messages=[{"role": "user", "content": user_content}],
    )
    _usage.record(response, "prod.shock", SONNET_MODEL)
    return json.loads(_strip_fences(extract_text(response)))


# Attached to every proposal message (Sonnet strategist alert or shock
# alert) that needs a human decision -- replaces the old free-text
# "reply 'test it'" convention, which silently did nothing for any
# phrasing that didn't match one of a few exact strings. A fixed,
# small set of valid answers is always presented as buttons in this
# project now, never left to free-text matching -- see
# llm_pipeline/pending_tests.py's own module docstring.
PROPOSAL_KEYBOARD_TEMPLATE = lambda pending_id: {
    "inline_keyboard": [[
        {"text": "Test It", "callback_data": f"propose:test:{pending_id}"},
        {"text": "Don't Test It", "callback_data": f"propose:skip:{pending_id}"},
    ]]
}


def format_shock_message(shock: dict, assessment: dict) -> str:
    base = (
        f"<b>{'━' * 4} SHOCK ALERT: {escape_html(shock['symbol'])} {'━' * 4}</b>\n\n"
        f"<b>Direction:</b> {escape_html(shock['direction'])} (z={shock['shock_z']:.2f}, roughly the top ~2% most "
        f"extreme volatility episodes for this coin -- excluded from routine conditions)\n\n"
        f"<b>Assessment:</b> {escape_html(assessment['assessment'])}"
    )
    if assessment["recommended_action"] == "propose_novel_test" and assessment.get("novel_condition_spec"):
        spec = assessment["novel_condition_spec"]
        base += (
            f"\n\n<b>This needs your input.</b>\n\n"
            f"<b>Proposed test: \"{escape_html(spec['label'])}\"</b>\n\n"
            f"({format_spec_clauses(spec)} → {escape_html(spec['direction'])})\n\n"
            f"Test It runs a real walk-forward backtest of this coin's own historical shocks -- if it validates, "
            f"the result also tracks THIS live occurrence (tagged shock_reactive), so we can measure how the "
            f"system actually did reacting in real time. Don't Test It dismisses this proposal."
        )
    else:
        base += "\n\n<i>No action taken -- noise, not worth escalating further.</i>"
    return base


def run_shock_scan(coins: list[str] | None = None) -> None:
    """Refreshes local OHLCV data from Binance first -- `scan_for_shocks`
    reads the last bar of what's on disk (`candidates/data_loading.py`),
    so without this the "real-time" shock detector would be checking a
    frozen snapshot instead of the market as it actually stands right
    now. A fetch failure is logged, not fatal -- the scan still runs
    against whatever data is already on disk."""
    try:
        # Imported lazily, inside the one function that uses it. At module level
        # this pulled `ccxt` into everything downstream -- importing telegram/bot.py
        # to test message chunking dragged in a whole exchange client, and CI failed
        # on it. Same convention execution/hyperopt_runner.py already uses for
        # Freqtrade: a heavy dependency needed by one call site should not become a
        # hard dependency of every module that transitively imports it.
        from data_ingestion.market_data.binance_fetcher import update_all as update_market_data

        update_market_data(coins or SHOCK_SCAN_COINS)
    except Exception as e:
        print(f"Market data refresh failed, scanning with existing data: {e}")

    load_dotenv()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    scan_coins = coins or SHOCK_SCAN_COINS
    for shock in scan_for_shocks(scan_coins):
        try:
            assessment = sonnet_shock_response(shock, client)
            reply_markup = None
            if assessment["recommended_action"] == "propose_novel_test" and assessment.get("novel_condition_spec"):
                s = assessment["novel_condition_spec"]
                spec = spec_from_dict(s)
                pending_id = push_pending_test(spec, scan_coins, live_coin=shock["symbol"], signal_class="shock_reactive")
                reply_markup = PROPOSAL_KEYBOARD_TEMPLATE(pending_id)
            if assessment["recommended_action"] != "propose_novel_test":
                # Nothing happened -- log it, do not notify. See replay/engine.py
                # ::_handle_assessment for why silence is the right default here.
                print(f"no_action: {assessment.get('assessment', '')[:120]}")
                continue
            message = format_shock_message(shock, assessment)
            sent = send_telegram(message, reply_markup=reply_markup)
            status = "notified" if sent else "notify FAILED, see error above"
            print(f"Shock escalated + {status}: {shock['symbol']} {shock['direction']} z={shock['shock_z']:.2f}")
        except Exception as e:
            # One malformed Sonnet response for ONE shock must not cost every
            # OTHER shock this scan found its own escalation -- same isolation
            # discipline run_once() already applies per headline. Without this,
            # a single truncated/malformed response silently killed the whole
            # scan (observed live during a replay run, same root cause the
            # max_tokens bump above addresses -- this is the second, independent
            # layer of defense against it).
            print(f"Failed to process shock '{shock.get('symbol', '?')}', skipping: {e}")


def send_telegram(message: str, reply_markup: dict | None = None) -> bool:
    load_dotenv()
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    if reply_markup is not None:
        payload["reply_markup"] = json.dumps(reply_markup)
    resp = requests.post(url, json=payload, timeout=15)
    if not resp.ok:
        print(f"Telegram send FAILED ({resp.status_code}): {resp.text[:300]}")
        return False
    return True


def format_sonnet_message(item: dict, assessment: dict) -> str:
    """The label shown for each `recommended_action` value is deliberately
    NOT a uniform "Recommended action: X" line -- "no_action" (nothing to
    do) reads as a plain statement; only "propose_novel_test" is
    genuinely a decision pending on the human, so it's the only case
    phrased as one."""
    base = (
        f"<b>Sonnet Strategist Alert</b>\n\n"
        f"<b>Headline:</b> {escape_html(item['headline'])}\n"
        f"<b>Asset:</b> {escape_html(item['asset'])} | <b>Magnitude:</b> {item['magnitude']}/5\n\n"
        f"<b>Assessment:</b> {escape_html(assessment['assessment'])}"
    )
    action = assessment["recommended_action"]
    if action == "propose_novel_test" and assessment.get("novel_condition_spec"):
        spec = assessment["novel_condition_spec"]
        base += (
            f"\n\n<b>This needs your input.</b> This looks like a condition we haven't tested before.\n\n"
            f"<b>Proposed test: \"{escape_html(spec['label'])}\"</b>\n\n"
            f"({format_spec_clauses(spec)} → {escape_html(spec['direction'])})\n\n"
            f"Test It runs a real walk-forward backtest of this condition before it's tracked as a live test "
            f"(no real money is ever placed on it). Don't Test It dismisses this proposal."
        )
    else:
        base += "\n\n<i>No action taken.</i>"
    return base


def _asset_to_coin(asset: str) -> str | None:
    if not asset or asset.upper() in ("MARKET", ""):
        return None  # a broad/unclear headline has no single coin to trade the live occurrence on
    symbol = f"{asset.upper()}USDT"
    return symbol if symbol in SHOCK_SCAN_COINS else None


def run_once() -> None:
    load_dotenv()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    articles = fetch_cryptocompare_news(limit=20)
    scored = haiku_scout(articles, client)

    for item in scored:
        if item.get("magnitude", 0) < ESCALATION_MAGNITUDE_THRESHOLD:
            print(f"Not escalated (magnitude {item.get('magnitude')}): {item['headline'][:80]}")
            continue
        try:
            assessment = sonnet_strategist(item, client)
            reply_markup = None
            if assessment["recommended_action"] == "propose_novel_test" and assessment.get("novel_condition_spec"):
                s = assessment["novel_condition_spec"]
                spec = spec_from_dict(s)
                pending_id = push_pending_test(spec, SHOCK_SCAN_COINS, live_coin=_asset_to_coin(item.get("asset", "")), signal_class="manual")
                reply_markup = PROPOSAL_KEYBOARD_TEMPLATE(pending_id)
            if assessment["recommended_action"] != "propose_novel_test":
                # Nothing happened -- log it, do not notify. See replay/engine.py
                # ::_handle_assessment for why silence is the right default here.
                print(f"no_action: {assessment.get('assessment', '')[:120]}")
                continue
            message = format_sonnet_message(item, assessment)
            sent = send_telegram(message, reply_markup=reply_markup)
            status = "notified" if sent else "notify FAILED, see error above"
            print(f"Escalated + {status}: {item['headline'][:80]}")
        except Exception as e:
            # One malformed Sonnet response (e.g. invalid JSON) must not
            # cost every OTHER headline in this batch its own escalation --
            # without this, a single bad item silently stops the whole run.
            print(f"Failed to process escalated headline '{item.get('headline', '?')[:80]}', skipping: {e}")


if __name__ == "__main__":
    try:
        run_once()
    except Exception as e:
        print(f"run_once() failed: {e}")
    try:
        run_shock_scan()
    except Exception as e:
        print(f"run_shock_scan() failed: {e}")
