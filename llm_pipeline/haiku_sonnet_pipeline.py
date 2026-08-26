"""Haiku Scout -> Sonnet Strategist -> Telegram. Haiku classifies each
headline against the live candidate battery; only genuinely unmatched,
significant conditions escalate to Sonnet, which reads real technical
state (not a placeholder) and proposes a concrete, checkable action --
never a freehand trade or a freehand number.

The human gate sits on exactly one kind of decision: whether to spend
compute validating a genuinely untested pattern (`propose_novel_test`,
and the shock-detection path in `run_shock_scan`). A routine trade
proposal (`propose_trade`) is never gated on human confirmation -- it can
only reference a candidate the battery has already validated, so the
anchors are never invented and there is nothing left for a human to
approve beyond what walk-forward validation already did. Requiring a
human to also bless each individual trade would defeat this project's own
purpose: testing whether the LLM layer's judgment holds up on its own,
not testing a human's judgment about the LLM's judgment.
"""
from __future__ import annotations

import json
import os

import requests
from anthropic import Anthropic
from dotenv import load_dotenv

from data_ingestion.news_sentiment.cryptocompare_fetcher import fetch_cryptocompare_news
from execution.signal_store import load_battery_state, push_manual_signal
from llm_pipeline.context_builder import build_context_summary, build_technical_snapshot
from llm_pipeline.novel_condition_tester import SUPPORTED_INDICATORS, ConditionSpec
from llm_pipeline.pending_tests import push_pending_test
from llm_pipeline.shock_detector import scan_for_shocks

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

SONNET_SYSTEM_PROMPT = f"""You are a market strategist for a crypto trading system. You will be \
given: (1) a flagged news headline, (2) real technical/portfolio state, (3) the live status of \
this system's candidate battery, including which candidates currently carry 'validated' status \
and their coin/direction. Treat a candidate's status as load-bearing: only 'validated' candidates \
may justify a trade; everything else is context, not a signal.

"propose_trade" is a routine, unattended decision -- it fires immediately, with no human \
confirmation, so it may ONLY reference a candidate CURRENTLY listed as 'validated' in the battery \
context you were given. You are not inventing a new pattern here, only recognizing that current \
conditions match one already proven -- if no validated candidate actually applies, use \
"propose_novel_test" or "no_action" instead, never stretch a trade_proposal to fit.

If this headline suggests a genuinely new, untested pattern -- not covered by any existing \
candidate and not already logged as rejected -- propose a novel-condition test instead (this DOES \
wait for human approval, since it spends real compute validating something unproven), using ONLY \
one of these whitelisted indicators (nothing else is buildable): {list(SUPPORTED_INDICATORS)}.

Return ONLY a JSON object with exactly these fields:
- "assessment": 1-2 sentences
- "recommended_action": one of "no_action", "watch", "propose_trade", "propose_novel_test", "exit_now"
- "watch_condition": a concrete, checkable condition (or null)
- "trade_proposal": null, or {{"candidate": "<must be a currently-validated candidate name>", \
  "coin": "...", "direction": "long"/"short", "reasoning": "..."}} \
  (TP/SL are never Sonnet's to set -- always drawn from that candidate's own validated anchors)
- "novel_condition_spec": null, or {{"label": "...", "indicator": "...", "op": "<"/">"/"<="/">=", \
  "threshold": <number>, "direction": "long"/"short"}}

No prose, no markdown fences, just the JSON object."""

SHOCK_SYSTEM_PROMPT = f"""You are a market strategist evaluating a real-time SHOCK EVENT -- a coin's \
short-term realized volatility has just spiked to a statistical extreme (this project's own \
Phase 1 methodology excludes exactly this kind of event from the static candidate battery's \
fitting, because a handful of crashes shouldn't distort barriers meant for ordinary conditions). \
This is deliberately the harder case a fixed rule set can't pre-classify -- you're being asked \
whether this specific instance is worth reacting to at all, not whether shocks in general are \
tradeable.

You will be given the shock's real, computed severity (a z-score) and direction (crash/surge), \
plus real technical/portfolio state and the candidate battery's status. Recommend one of:
- "no_action": noise, not worth a human's attention.
- "propose_novel_test": worth finding out if THIS coin's shocks, historically, show a real \
  reversal or continuation pattern -- if so, propose testing it with the "shock_zscore" indicator \
  (the only whitelisted way to test this; nothing else measures the same thing), e.g. \
  {{"label": "shock_reactive_<coin>", "indicator": "shock_zscore", "op": ">=", "threshold": 3.0, \
  "direction": "long" or "short"}}. If the human approves and it validates, the resulting anchors \
  are used for a LIVE trade on THIS occurrence, tagged separately from routine trades so how the \
  system actually performed reacting to real shocks, in real time, can be measured on its own.

TP/SL are never yours to set -- they only ever come from a validated anchor set, never invented \
here. Return ONLY a JSON object: "assessment" (1-2 sentences), "recommended_action" \
("no_action"/"propose_novel_test"), "novel_condition_spec" (null or the spec above). No prose, no \
markdown fences."""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def haiku_scout(articles: list[dict], client: Anthropic) -> list[dict]:
    if not articles:
        return []
    headlines_block = "\n".join(f"- {a['headline']}" for a in articles)
    response = client.messages.create(
        model=HAIKU_MODEL, max_tokens=2048, system=HAIKU_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": headlines_block}],
    )
    return json.loads(_strip_fences(response.content[0].text))


def sonnet_strategist(flagged: dict, client: Anthropic) -> dict:
    technical_snapshot = build_technical_snapshot(flagged.get("asset", "MARKET"), FREQTRADE_DB_PATH)
    context_summary = build_context_summary()
    user_content = (
        f"HEADLINE: {flagged['headline']} (asset={flagged['asset']}, "
        f"sentiment={flagged['sentiment']}, magnitude={flagged['magnitude']}, "
        f"event_type={flagged['event_type']})\n\n"
        f"TECHNICAL SNAPSHOT:\n{technical_snapshot}\n\n"
        f"CANDIDATE BATTERY CONTEXT:\n{context_summary}"
    )
    response = client.messages.create(
        model=SONNET_MODEL, max_tokens=700, system=SONNET_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    return json.loads(_strip_fences(response.content[0].text))


def sonnet_shock_response(shock: dict, client: Anthropic) -> dict:
    technical_snapshot = build_technical_snapshot(shock["symbol"], FREQTRADE_DB_PATH)
    context_summary = build_context_summary()
    user_content = (
        f"SHOCK DETECTED: {shock['symbol']}, direction={shock['direction']}, "
        f"shock_z={shock['shock_z']:.2f}, latest_return={shock.get('latest_return')}\n\n"
        f"TECHNICAL SNAPSHOT:\n{technical_snapshot}\n\n"
        f"CANDIDATE BATTERY CONTEXT:\n{context_summary}"
    )
    response = client.messages.create(
        model=SONNET_MODEL, max_tokens=500, system=SHOCK_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    return json.loads(_strip_fences(response.content[0].text))


def format_shock_message(shock: dict, assessment: dict) -> str:
    base = (
        f"SHOCK ALERT -- {shock['symbol']}\n\n"
        f"Direction: {shock['direction']} (z={shock['shock_z']:.2f}, a statistical extreme this "
        f"project's own methodology treats as excluded from routine conditions)\n\n"
        f"Assessment: {assessment['assessment']}\n"
        f"Recommended action: {assessment['recommended_action']}"
    )
    if assessment["recommended_action"] == "propose_novel_test" and assessment.get("novel_condition_spec"):
        spec = assessment["novel_condition_spec"]
        base += (
            f"\n\nProposed test: {spec['indicator']} {spec['op']} {spec['threshold']} -> {spec['direction']}\n"
            f"Reply 'test it' to run a real walk-forward backtest of this coin's own historical "
            f"shocks. If it validates, the result trades THIS live occurrence too (tagged "
            f"shock_reactive), so we can measure how the system actually did reacting in real time."
        )
    return base


def run_shock_scan(coins: list[str] | None = None) -> None:
    load_dotenv()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    scan_coins = coins or SHOCK_SCAN_COINS
    for shock in scan_for_shocks(scan_coins):
        assessment = sonnet_shock_response(shock, client)
        if assessment["recommended_action"] == "propose_novel_test" and assessment.get("novel_condition_spec"):
            s = assessment["novel_condition_spec"]
            spec = ConditionSpec(label=s["label"], indicator=s["indicator"], op=s["op"],
                                  threshold=s["threshold"], direction=s["direction"])
            push_pending_test(spec, scan_coins, live_coin=shock["symbol"], signal_class="shock_reactive")
        message = format_shock_message(shock, assessment)
        sent = send_telegram(message)
        status = "notified" if sent else "notify FAILED, see error above"
        print(f"Shock escalated + {status}: {shock['symbol']} {shock['direction']} z={shock['shock_z']:.2f}")


def send_telegram(message: str, reply_markup: dict | None = None) -> bool:
    load_dotenv()
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    if reply_markup is not None:
        payload["reply_markup"] = json.dumps(reply_markup)
    resp = requests.post(url, json=payload, timeout=15)
    if not resp.ok:
        print(f"Telegram send FAILED ({resp.status_code}): {resp.text[:300]}")
        return False
    return True


def execute_routine_trade(trade_proposal: dict) -> str:
    """Fires immediately, no human confirmation -- `trade_proposal` must
    reference a candidate CURRENTLY 'validated' in the battery (enforced
    here, not just requested in the prompt: Sonnet's output is untrusted
    input like any other model output, so the anchor lookup is the real
    gate, not the instruction asking it to behave). Returns a status
    string for the Telegram notification, never a request for approval --
    routine trades are not a human decision in this project (see module
    docstring for why)."""
    battery = load_battery_state()
    spec = battery.get("candidates", {}).get(trade_proposal.get("candidate", ""))
    if spec is None:
        return (f"REJECTED: '{trade_proposal.get('candidate')}' is not currently a validated "
                f"candidate -- Sonnet proposed a trade without a real anchor set behind it, refused.")
    push_manual_signal(
        coin=trade_proposal["coin"], direction=trade_proposal["direction"],
        tp_mult=spec["tp_mult"], sl_mult=spec["sl_mult"], anchors=spec["anchors"],
        reasoning=trade_proposal["reasoning"], approved_by="sonnet_autonomous",
        signal_class="sonnet_confirmed",
    )
    return f"Opened now: {trade_proposal['direction'].upper()} {trade_proposal['coin']} (candidate: {trade_proposal['candidate']})"


def format_sonnet_message(item: dict, assessment: dict, execution_status: str | None = None) -> str:
    base = (
        f"Sonnet Strategist Alert\n\n"
        f"Headline: {item['headline']}\n"
        f"Asset: {item['asset']} | Magnitude: {item['magnitude']}/5\n\n"
        f"Assessment: {assessment['assessment']}\n"
        f"Recommended action: {assessment['recommended_action']}"
    )
    if assessment["recommended_action"] == "propose_trade" and assessment.get("trade_proposal"):
        tp = assessment["trade_proposal"]
        base += (
            f"\n\nCandidate: {tp.get('candidate')}\n"
            f"{tp['direction'].upper()} {tp['coin']}\n"
            f"Reasoning: {tp['reasoning']}\n"
            f"{execution_status or 'Not executed.'}"
        )
    elif assessment["recommended_action"] == "propose_novel_test" and assessment.get("novel_condition_spec"):
        spec = assessment["novel_condition_spec"]
        base += (
            f"\n\nThis looks like a condition we haven't tested before.\n"
            f"Proposed test: {spec['indicator']} {spec['op']} {spec['threshold']} -> {spec['direction']}\n"
            f"Reply 'test it' to run a real walk-forward backtest of this condition before it "
            f"ever influences a trade."
        )
    else:
        base += f"\nWatch condition: {assessment.get('watch_condition') or 'n/a'}"
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
        assessment = sonnet_strategist(item, client)
        execution_status = None
        if assessment["recommended_action"] == "propose_trade" and assessment.get("trade_proposal"):
            execution_status = execute_routine_trade(assessment["trade_proposal"])
        elif assessment["recommended_action"] == "propose_novel_test" and assessment.get("novel_condition_spec"):
            s = assessment["novel_condition_spec"]
            spec = ConditionSpec(label=s["label"], indicator=s["indicator"], op=s["op"],
                                  threshold=s["threshold"], direction=s["direction"])
            push_pending_test(spec, SHOCK_SCAN_COINS, live_coin=_asset_to_coin(item.get("asset", "")), signal_class="manual")
        message = format_sonnet_message(item, assessment, execution_status)
        sent = send_telegram(message)
        status = "notified" if sent else "notify FAILED, see error above"
        print(f"Escalated + {status}: {item['headline'][:80]}")


if __name__ == "__main__":
    run_once()
    run_shock_scan()
