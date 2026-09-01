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
    INDICATOR_PLAIN_NAMES, MIN_HISTORICAL_OCCURRENCES, OPERATOR_PLAIN, ConditionSpec,
    build_indicator_leadup, build_indicator_snapshot, clause_from_dict,
    filter_redundant_proposals, proposable_indicators, proposals_from_assessment,
    spec_from_proposal, spec_to_dict,
)
from llm_pipeline.pending_tests import push_pending_test
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
Bollinger touch, a volume spike -- is NOT a hypothesis in this project, \
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

Neither the volatility shock nor the compression measure is available as a clause. A shock is a \
price OUTCOME, not a market state -- "the price moved hard, then the price did something" is the \
tautology this rule exists to exclude. The compression measure is the reason you are being asked \
at all, so it is fixed at every proposal by construction and can distinguish nothing. Describe \
what the market looked like and what had been published.

Your label must also describe what the clauses actually test. Do not name a condition "post-CPI ..." \
unless a CPI/macro clause is genuinely in it -- that is checked in code too.

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

COMPRESSION_SYSTEM_PROMPT = f"""You are a quantitative researcher, not a trader. Your subject is a single question: \
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
WHEN YOU ARE ASKED, and why it matters for your answer. You are consulted at exactly one kind of \
moment: a period of unusually LOW volatility for this coin has just ended, and the exit has held \
for several days. The market coiled, stayed quiet, and has now begun to move again.

That moment is chosen because a compressed market is more likely than an ordinary one to be \
followed by a sustained directional move -- but nothing about the compression says WHICH \
DIRECTION. That is the question put to you. Do not try to explain why the market went quiet; the \
quiet is the setting, not the subject. Ask instead: given what was published while it was coiling, \
and the state it coiled into, is the resolution predictable?

The exit day's own move is NOT evidence of direction. Measured on 214 of these episodes, the \
direction of the first bar out of compression has no relationship to where price is two weeks \
later. Do not extrapolate from it.

You will be given the episode in three phases -- PHASE A (when it began and how quiet it got), \
PHASE MIDDLE (how long it lasted, how price drifted, and every macro release published while it \
lasted, each as a CHANGE from the prior print and a SURPRISE in standard deviations of that \
series' usual move), and PHASE B (the day it ended) -- plus this coin's real indicator readings \
AS OF THE EXIT, recent real news headlines, real technical/portfolio state, and the candidate \
battery's status.

Conditions may be built ONLY from these whitelisted indicators (nothing else is buildable): \
{proposable_indicators()}. The list is derived from the code that validates your proposal, so an \
indicator missing from it will be rejected before it is ever tested. Ground \
your reasoning ONLY in these real numbers/headlines you were actually given -- never invent an \
indicator reading, a headline, or a release you weren't shown. Recommend one of:
- "no_action": noise, not worth a human's attention.
- "propose_novel_test": worth finding out if a SPECIFIC combination of what you were just given, \
  historically, shows a real reversal or continuation pattern -- e.g. you notice the market coiled \
  while inflation prints ran hotter each time AND it broke out with RSI already deeply oversold, so \
  you propose testing exactly that combination. Don't force a \
  compound story where the evidence doesn't support one. Use ONLY indicators you were \
  actually shown a reading for. Example: {{"label": "hot_cpi_into_oversold_<coin>", "clauses": \
  [{{"indicator": "cpi_surprise", "op": ">=", "threshold": 1.0, "within_days": 2}}, \
  {{"indicator": "rsi_14d", "op": "<", "threshold": 30}}], "direction": "long" or "short"}} -- note \
  "within_days" (integer 0-14, default 0): 0 = true on the day the condition fires, K = true at any \
  point in the last K days. Use it to express that the RELEASE CAME FIRST and something else followed, \
  which is a different hypothesis from both happening on the same day. The day-by-day LEAD-UP table \
  you are shown exists so you can tell which of the two you are actually looking at. If the human approves and it's \
  accepted, the resulting condition is tracked going forward and this occurrence becomes its first \
  live test, so how the system actually performed on a real resolution, in real time, can be \
  measured on its own.

AT MOST TWO CLAUSES PER SPEC, and PROPOSE TWO SPECS rather than one deeper condition when the \
evidence supports more than one idea. Each extra clause divides the number of historical \
occurrences by roughly eight: a three-clause condition has a median of 12 where {MIN_HISTORICAL_OCCURRENCES} are \
needed to measure anything, and is rejected by code. An idea you would have written as "rate cut in \
the last 7 days AND funding negative AND RSI below 30" should be sent as two specs: "rate cut in \
the last 7 days AND funding negative", and "rate cut in the last 7 days AND RSI below 30". The two \
must be genuinely different hypotheses -- two specs firing on the same days are detected in code \
and the second discarded, so a near-duplicate wastes the slot. One good spec beats one good spec \
plus filler.

CHOOSE within_days FOR WHAT THE HYPOTHESIS MEANS, not for how often it fires. "The print came out \
today and the market was already oversold" and "a print came out this week, and the market is \
oversold now" are different claims about how an effect travels, and only you can say which one you \
mean. A lookback of 0 says the two things coincided; a lookback of K says the news came first and \
the market condition followed within K days.

Do not reach for a longer window to make a condition occur more often. Occurrences on consecutive \
days are one episode counted several times, and the gate that decides whether a condition can be \
tested counts EPISODES -- so a wider window buys almost no additional evidence, and the extra \
firings it produces are the same evidence repeated.


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
("no_action"/"propose_novel_test"), "novel_condition_specs" (null, or a list of one or two specs). No prose, no \
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
#     COMPRESSION_SYSTEM_PROMPT  ~1500  cached
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


def sonnet_compression_response(episode: dict, client: Anthropic) -> dict:
    """The live counterpart of replay/judgment.py::judge_event for a confirmed
    compression exit. Same three-phase framing, same question, so a hypothesis
    discovered live and one discovered in the replay are answers to the same
    prompt rather than to two that happen to look similar."""
    from replay.judgment import format_compression_event

    symbol = episode["symbol"]
    technical_snapshot = build_technical_snapshot(symbol, FREQTRADE_DB_PATH)
    context_summary = build_context_summary()
    # Dated to the EXIT (point B), not to today. The confirmation window decided
    # whether to ask; it must not leak into what the model is shown, exactly as
    # in the replay.
    b_date = episode["b_date"]
    indicator_snapshot = (build_indicator_snapshot(symbol, as_of=b_date) + "\n\n"
                          + build_indicator_leadup(symbol, as_of=b_date))
    user_content = (
        f"{format_compression_event(symbol, episode)}\n\n"
        f"INDICATOR SNAPSHOT (as of the exit):\n{indicator_snapshot}\n\n"
        f"RECENT NEWS HEADLINES:\n{_recent_headlines_summary()}\n\n"
        f"TECHNICAL SNAPSHOT:\n{technical_snapshot}\n\n"
        f"CANDIDATE BATTERY CONTEXT:\n{context_summary}"
    )
    response = client.messages.create(
        model=SONNET_MODEL, max_tokens=2000, system=cached_system(COMPRESSION_SYSTEM_PROMPT),
        messages=[{"role": "user", "content": user_content}],
    )
    _usage.record(response, "prod.compression", SONNET_MODEL)
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


def format_compression_message(episode: dict, assessment: dict) -> str:
    base = (
        f"<b>{'━' * 4} COMPRESSION RESOLVED: {escape_html(episode['symbol'])} {'━' * 4}</b>\n\n"
        f"<b>Quiet since:</b> {episode['a_date'].date()} ({episode['duration']} day(s), "
        f"volatility {episode['z_at_a']:.2f} sd below this coin's own normal)\n"
        f"<b>Broke out:</b> {episode['b_date'].date()} ({episode['b_return']:+.2%} that day)\n\n"
        f"<b>Assessment:</b> {escape_html(assessment['assessment'])}"
    )
    specs = assessment.get("novel_condition_specs") or (
        [assessment["novel_condition_spec"]] if assessment.get("novel_condition_spec") else [])
    if assessment["recommended_action"] == "propose_novel_test" and specs:
        plural = len(specs) > 1
        base += f"\n\n<b>{'These need' if plural else 'This needs'} your input.</b>\n\n"
        for i, spec in enumerate(specs, 1):
            head = f"<b>{i}. \"{escape_html(spec['label'])}\"</b>" if plural else \
                   f"<b>Proposed test: \"{escape_html(spec['label'])}\"</b>"
            base += f"{head}\n\n({format_spec_clauses(spec)} → {escape_html(spec['direction'])})\n\n"
        base += (f"Test It runs a real walk-forward backtest of "
                 f"{'each condition' if plural else 'this condition'} before "
                 f"{'they are' if plural else "it's"} tracked as "
                 f"{'live tests' if plural else 'a live test'} (no real money is ever placed). "
                 f"Don't Test It dismisses {'them' if plural else 'this proposal'}.")
    else:
        base += "\n\n<i>No action taken -- nothing here worth testing.</i>"
    return base


def run_compression_scan(coins: list[str] | None = None) -> None:
    """Refreshes local OHLCV from Binance first, then escalates each confirmed
    compression exit that has not already been escalated.

    Replaces `run_shock_scan`, and fixes a defect it carried: `scan_for_shocks`
    tested the volatility STATE, so an hourly daemon re-escalated the same
    multi-day shock every hour. `scan_for_compression_exits` keeps a ledger and
    fires once per episode.
    """
    from llm_pipeline.compression_detector import mark_escalated, scan_for_compression_exits

    try:
        # Lazy, for the same reason as before: importing ccxt at module level made
        # a heavy exchange client a hard dependency of everything downstream.
        from data_ingestion.market_data.binance_fetcher import update_all as update_market_data

        update_market_data(coins or SHOCK_SCAN_COINS)
    except Exception as e:
        print(f"Market data refresh failed, scanning with existing data: {e}")

    load_dotenv()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    scan_coins = coins or SHOCK_SCAN_COINS
    for episode in scan_for_compression_exits(scan_coins):
        try:
            assessment = sonnet_compression_response(episode, client)
            reply_markup = None
            specs = []
            for raw in proposals_from_assessment(assessment):
                spec, err = spec_from_proposal(raw)
                if spec is None:
                    print(f"Proposal rejected, not queued ({episode['symbol']}): {err}")
                    continue
                specs.append(spec)
            # Two proposals that fire on the same days are one hypothesis wearing
            # two hats and would spend twice the alpha budget for one piece of
            # information. Checked on behaviour, never on shared clauses -- the
            # intended pattern is two proposals sharing their news term.
            specs, notes = filter_redundant_proposals(specs, scan_coins)
            for note in notes:
                print(f"({episode['symbol']}) {note}")
            if specs:
                pending_id = push_pending_test(specs, scan_coins, live_coin=episode["symbol"],
                                                signal_class="compression_exit")
                reply_markup = PROPOSAL_KEYBOARD_TEMPLATE(pending_id)
            elif assessment["recommended_action"] == "propose_novel_test":
                mark_escalated(episode["symbol"], episode["b_date"])
                continue
            assessment["novel_condition_specs"] = [spec_to_dict(sp) for sp in specs]
            # Marked BEFORE the notification can fail: a send error must not leave
            # the episode un-ledgered and re-escalating every hour afterwards.
            mark_escalated(episode["symbol"], episode["b_date"])
            if assessment["recommended_action"] != "propose_novel_test":
                print(f"no_action: {assessment.get('assessment', '')[:120]}")
                continue
            sent = send_telegram(format_compression_message(episode, assessment), reply_markup=reply_markup)
            print(f"Compression escalated + {'notified' if sent else 'notify FAILED'}: "
                  f"{episode['symbol']} exit {episode['b_date'].date()}")
        except Exception as e:
            # One malformed response must not cost the other coins their scan.
            print(f"Failed to process compression exit '{episode.get('symbol', '?')}', skipping: {e}")


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
        run_compression_scan()
    except Exception as e:
        print(f"run_compression_scan() failed: {e}")
