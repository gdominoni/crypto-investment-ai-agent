"""The two structurally separate interaction modes described in this
project's README: free-text conversation (routed to Sonnet, but every
number it cites still comes from a real computation, never invented) and
structured commands / inline keyboards (routed straight to
`kpi_queries.py`, no LLM involved at all). Also handles the automated
post-mortem fired when a trade closes on a stop-loss.
"""
from __future__ import annotations

import os

import requests
from anthropic import Anthropic
from dotenv import load_dotenv

from execution.signal_store import consume_manual_signal, load_manual_signals, push_manual_signal
from llm_pipeline.context_builder import build_context_summary, build_technical_snapshot
from llm_pipeline.novel_condition_tester import ConditionSpec, test_novel_condition
from telegram.kpi_queries import format_kpi_table, kpi_table

SONNET_MODEL = "claude-sonnet-5"
FREQTRADE_DB_PATH = os.environ.get("FREQTRADE_DB_PATH", "execution/tradesv3.sqlite")

MARKET_CHECK_SYSTEM_PROMPT = """You are a market-check assistant for a crypto trading system. \
Given real technical/portfolio state and the live candidate battery context, answer the user's \
question in 2-4 sentences. Cite only the numbers given to you in the state below -- never invent \
a price, a percentage, or a trade detail. If nothing in the given state answers the question, say \
so plainly rather than guessing."""

KPI_KEYBOARD = {
    "inline_keyboard": [
        [{"text": "By Coin", "callback_data": "kpi:coin"}, {"text": "By Signal", "callback_data": "kpi:signal"}],
        [{"text": "By Decision Type", "callback_data": "kpi:signal_class"}],
        [{"text": "Overall", "callback_data": "kpi:overall"}],
    ]
}


def _send(text: str, reply_markup: dict | None = None) -> bool:
    load_dotenv()
    token, chat_id = os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
    resp = requests.post(url, json=payload, timeout=15)
    return resp.ok


def handle_natural_language(text: str, client: Anthropic) -> str:
    """Market checks ("how's the market", "do we have open trades") and
    conversational follow-ups -- never the KPI path, which is command-
    driven and LLM-free by design (see `handle_command`)."""
    snapshot = build_technical_snapshot("MARKET", FREQTRADE_DB_PATH)
    context = build_context_summary()
    response = client.messages.create(
        model=SONNET_MODEL, max_tokens=400, system=MARKET_CHECK_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"USER QUESTION: {text}\n\nSTATE:\n{snapshot}\n\n{context}"}],
    )
    return response.content[0].text


def handle_command(command: str, args: list[str]) -> tuple[str, dict | None]:
    if command == "/results":
        return "What breakdown would you like?", KPI_KEYBOARD
    return f"Unknown command: {command}", None


def handle_kpi_callback(callback_data: str) -> str:
    _, group = callback_data.split(":")
    group_by = None if group == "overall" else group
    df = kpi_table(FREQTRADE_DB_PATH, group_by=group_by)
    title = f"Results by {group}:" if group_by else "Results (all-time, overall):"
    return format_kpi_table(df, title)


def handle_test_it_confirmation(pending_spec: ConditionSpec, coins: list[str], approved_by: str,
                                 live_coin: str | None = None, signal_class: str = "manual") -> str:
    """Fired when a human replies "test it" to a Sonnet novel-condition
    proposal -- runs the actual walk-forward test (never assumed, never
    faked). If it validates AND `live_coin` is given (the specific coin
    whose live occurrence prompted this test -- e.g. the coin a shock was
    just detected on), the resulting anchors are pushed immediately as a
    manual signal for THAT occurrence, tagged `signal_class` so it can be
    measured separately from routine trades. Validating in general and
    trading the specific live instance are two different things -- this
    does both, not just the first."""
    result = test_novel_condition(pending_spec, coins)
    status = result["status"]
    if status == "insufficient_data":
        return f"Tested '{pending_spec.label}': not enough historical occurrences to evaluate yet."
    lines = [
        f"Tested '{pending_spec.label}': status = {status}",
        f"N={result['n']}  win_rate={result['win_rate']:.1%}  strict_win_rate={result['strict_win_rate']:.1%}",
        f"Sortino={result['sortino']:.2f}  total_expectancy={result['total_expectancy']:+.1%}  "
        f"timeout_fraction={result['timeout_fraction']:.1%}",
    ]
    if status == "validated":
        lines.append("This condition is now eligible to be approved for a live trade the next time it fires.")
        if live_coin:
            push_manual_signal(
                coin=live_coin, direction=pending_spec.direction,
                tp_mult=result["live_tp_mult"], sl_mult=result["live_sl_mult"], anchors=result["live_anchors"],
                reasoning=f"Novel-condition test '{pending_spec.label}' validated on approval; "
                          f"trading the live occurrence that prompted it.",
                approved_by=approved_by, signal_class=signal_class,
            )
            lines.append(f"Pushed as a live signal for {live_coin} now -- tagged '{signal_class}'.")
    else:
        lines.append("This does not clear the bar for live trading -- logged as tested, will not be re-proposed without new evidence.")
    return "\n".join(lines)


def send_stoploss_postmortem(trade_pair: str, exit_reason: str, close_profit: float,
                              enter_tag: str | None, client: Anthropic) -> None:
    context = build_context_summary()
    prompt = (
        f"A trade just closed on {trade_pair}, reason='{exit_reason}', profit={close_profit:+.2%}, "
        f"opened by candidate/signal '{enter_tag or 'manual/sonnet'}'. Write a 2-3 sentence post-mortem: "
        f"was this an ordinary outcome for this signal's known behavior, or does it look like the "
        f"signal's underlying assumption broke? Cite only the numbers given here.\n\n{context}"
    )
    response = client.messages.create(model=SONNET_MODEL, max_tokens=300, messages=[{"role": "user", "content": prompt}])
    _send(f"{'SL' if close_profit < 0 else 'TP'} hit on {trade_pair} ({close_profit:+.2%}).\n\n{response.content[0].text}\n\nDocumented in the history report.")
