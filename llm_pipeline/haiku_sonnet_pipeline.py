"""Sonnet Strategist -> Telegram, on one trigger: a confirmed exit from a
volatility-compression episode.

THE HAIKU HEADLINE PATH WAS REMOVED (2026-09-02), and the reasoning is the
point rather than a footnote. Haiku screened live CryptoCompare headlines and
escalated the significant ones to Sonnet. Nothing it surfaced could ever enter
a testable hypothesis: `proposable_indicators()` contains no headline or
sentiment term, and every proposal must carry one of `cpi_surprise`,
`rate_surprise`, `jobless_claims_surprise` -- all three FRED series. A headline
could only ever prompt a question whose answer had to be phrased in the exact
vocabulary the compression trigger already uses, and the project's own log
recorded the consequence: of 771 live tests opened for Sonnet-discovered
candidates, ZERO were news-linked.

The obvious repair -- backfill news history so a sentiment clause becomes
testable -- was measured rather than assumed, in `forecast/sentiment_power.py`,
which models sentiment as a continuous daily score parameterised by `rho`, its
correlation with the forward return. Accepted conditions out of 57, per feed
quality: 2 at rho=0.0 (the pure-noise floor), 3 at rho=0.04 (what real news
sentiment achieves), 5 at 0.08, 20 at 0.15, 23 at 0.30. A realistic feed is
indistinguishable from noise; detection needs a feed 3-4x better than published
work reports. So the backfill would not have rescued it either.

Both branches are closed by measurement: a rare discrete news event cannot
reach `MIN_HISTORICAL_EPISODES` in nine years across seven coins, and the
continuous form is undetectable at achievable quality. The component was
deleted rather than kept for the badge. See
docs/case_study/methodology-decisions.md.

Sonnet's live role is narrow, by design: (1) decide whether a confirmed
compression exit suggests a genuinely new, untested pattern worth a human's
"test it" (`propose_novel_test`), and (2) answer natural-language questions
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

from llm_pipeline.context_builder import build_context_summary, build_technical_snapshot
from llm_pipeline.novel_condition_tester import (
    INDICATOR_PLAIN_NAMES, MIN_HISTORICAL_OCCURRENCES, OPERATOR_PLAIN, ConditionSpec,
    build_indicator_leadup, build_indicator_snapshot, clause_from_dict,
    filter_redundant_proposals, proposable_indicators, proposals_from_assessment,
    spec_from_proposal, spec_to_dict,
)
from llm_pipeline.pending_tests import push_pending_test
from llm_pipeline import usage as _usage

SONNET_MODEL = "claude-sonnet-5"
FREQTRADE_DB_PATH = os.environ.get("FREQTRADE_DB_PATH", "execution/tradesv3.sqlite")
SHOCK_SCAN_COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LTCUSDT"]

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
AS OF THE EXIT, real technical/portfolio state, and the candidate battery's status.

Conditions may be built ONLY from these whitelisted indicators (nothing else is buildable): \
{proposable_indicators()}. The list is derived from the code that validates your proposal, so an \
indicator missing from it will be rejected before it is ever tested. Ground \
your reasoning ONLY in these real numbers you were actually given -- never invent an \
indicator reading or a release you weren't shown. Recommend one of:
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


# Minimum cacheable prefix: 1024 tokens for Sonnet. Re-measured 2026-09-04 with
# the API's own count_tokens -- the previous figures here were stale by a wide
# margin (REPLAY was recorded as 1575 and is 3481), and since this table is what
# the cache/no-cache decision is read off, a stale number here is how a prompt
# ends up marked and silently uncached:
#     REPLAY_SYSTEM_PROMPT        3481   cached
#     COMPRESSION_SYSTEM_PROMPT   2588   cached
#     MARKET_CHECK_PROMPT          630   too short -- NOT marked
# The short one is deliberately left unmarked: below the floor the breakpoint is
# silently ignored, which would leave code that looks cached and isn't.
# (SONNET_SYSTEM_PROMPT and HAIKU_SYSTEM_PROMPT were measured here too; both
# belonged to the removed headline path.)
#
# MEASURED BEHAVIOUR, worth knowing before tuning this. Over 73 real calls the
# cache halved input cost ($0.68 -> $0.34), but it was REWRITTEN ~20 times
# rather than once: `ephemeral` is the 5-minute tier, and compression exits are
# rare enough that minutes of local compute pass between calls, expiring it. The
# 1-hour tier would cover those gaps but writes at 2.0x instead of 1.25x; net,
# it saves roughly $0.30 over a full run -- real, and small enough that it is
# recorded here rather than acted on. Switching tiers also means updating the
# multipliers in llm_pipeline/usage.py, which that module's own comment flags.
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


def sonnet_compression_response(episode: dict, client: Anthropic) -> dict:
    """The live counterpart of replay/judgment.py::judge_event for a confirmed
    compression exit. Same three-phase framing, same question, so a hypothesis
    discovered live and one discovered in the replay are answers to the same
    prompt rather than to two that happen to look similar.

    The RECENT NEWS HEADLINES block was removed from this context along with
    the Haiku path, and that closed a live train/serve gap rather than merely
    tidying up: production showed Sonnet headlines here while the replay's
    `judge_event` never did, so the two were NOT answering the same prompt on
    the primary trigger despite the docstring above claiming they were. Since
    no headline can appear in any proposable clause, the block could only
    invite reasoning the model was then unable to express."""
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
        f"TECHNICAL SNAPSHOT:\n{technical_snapshot}\n\n"
        f"CANDIDATE BATTERY CONTEXT:\n{context_summary}"
    )
    response = client.messages.create(
        model=SONNET_MODEL, max_tokens=4000, system=cached_system(COMPRESSION_SYSTEM_PROMPT),
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
        # Bound outside the f-string rather than inlined. "it's" carries an
        # apostrophe, so inlining it needs a double quote nested inside a
        # double-quoted f-string -- legal only from Python 3.12 (PEP 701) and a
        # SyntaxError on 3.11, which is what CI runs and what the README
        # promises. It is a syntax error at IMPORT, so it took down the whole
        # test collection, not one test. See tests/test_python_compatibility.py.
        they_are = "they are" if plural else "it's"
        base += (f"Test It runs a real walk-forward backtest of "
                 f"{'each condition' if plural else 'this condition'} before "
                 f"{they_are} tracked as "
                 f"{'live tests' if plural else 'a live test'} (no real money is ever placed). "
                 f"Don't Test It dismisses {'them' if plural else 'this proposal'}.")
    else:
        base += "\n\n<i>No action taken -- nothing here worth testing.</i>"
    return base


def send_telegram(message: str, reply_markup: dict | None = None) -> bool:
    """THE function whose absence broke the compression trigger, which is
    this project's primary discovery path -- deleted by commit 20b134f
    (2026-08-31) alongside `format_sonnet_message` and `_asset_to_coin`
    while that commit rewrote the neighbouring shock->compression code.

    `run_compression_scan()` calls this on its last line, AFTER it has
    already queued the proposal and called `mark_escalated()`. So the
    NameError left the worst possible state: the episode permanently
    ledgered as escalated (never retried), a pending test sitting behind
    buttons no human ever saw, and the queue entry expiring silently 48h
    later. Every live compression exit since 2026-08-31 was lost that way.
    `run_once()`'s headline path was broken by the same deletion, but it
    keeps no ledger, so it merely lost the message.

    Invisible for three compounding reasons: each call site's own broad
    `except Exception` printed the NameError instead of raising it; no test
    invoked either function end-to-end (now closed -- see
    tests/test_haiku_sonnet_pipeline.py and the end-to-end case in
    tests/test_compression_detector.py); and the replay, which is what
    actually gets run and watched, sends via telegram/bot.py::_send and
    never touches this function at all.

    KNOWN GAP, carried over deliberately from the deleted original rather
    than fixed here: this does not chunk messages past Telegram's 4,096-char
    limit the way telegram/bot.py::_send does (which splits at 3,500 after
    that exact limit silently dropped a 6,880-char /replay_summary -- see
    docs/case_study/methodology-decisions.md). A proposal message is a few
    hundred characters, so it is not reachable today; it is the same latent
    bug in a second sender."""
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


if __name__ == "__main__":
    try:
        run_once()
    except Exception as e:
        print(f"run_once() failed: {e}")
    try:
        run_compression_scan()
    except Exception as e:
        print(f"run_compression_scan() failed: {e}")
