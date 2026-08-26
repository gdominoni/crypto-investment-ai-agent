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
from llm_pipeline.dynamic_candidates import record_test_result
from llm_pipeline.haiku_sonnet_pipeline import escape_html, extract_text
from llm_pipeline.novel_condition_tester import ConditionSpec, test_novel_condition
from llm_pipeline.pending_tests import pop_pending_test
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
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
    resp = requests.post(url, json=payload, timeout=15)
    return resp.ok


def handle_natural_language(text: str, client: Anthropic) -> str:
    """Market checks ("how's the market", "do we have open trades") and
    conversational follow-ups -- never the KPI path, which is command-
    driven and LLM-free by design (see `handle_command`). Escaped, not
    bolded: this is free-form Sonnet prose with no template around it, so
    only HTML-safety is applied, not formatting Sonnet didn't ask for."""
    snapshot = build_technical_snapshot("MARKET", FREQTRADE_DB_PATH)
    context = build_context_summary()
    response = client.messages.create(
        model=SONNET_MODEL, max_tokens=400, system=MARKET_CHECK_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"USER QUESTION: {text}\n\nSTATE:\n{snapshot}\n\n{context}"}],
    )
    return escape_html(extract_text(response))


def handle_command(command: str, args: list[str]) -> tuple[str, dict | None]:
    if command == "/results":
        return "What breakdown would you like?", KPI_KEYBOARD
    return f"Unknown command: {command}", None


def handle_kpi_callback(callback_data: str) -> str:
    _, group = callback_data.split(":")
    group_by = None if group == "overall" else group
    df = kpi_table(FREQTRADE_DB_PATH, group_by=group_by)
    title = f"Results by {group}:" if group_by else "Results (all-time, overall):"
    table = format_kpi_table(df, title)
    title_line, _, rest = table.partition("\n")
    return f"<b>{escape_html(title_line)}</b>\n<pre>{escape_html(rest)}</pre>"  # <pre> preserves the table's column alignment


def handle_test_it_confirmation(pending_spec: ConditionSpec, coins: list[str], approved_by: str,
                                 live_coin: str | None = None, signal_class: str = "manual") -> str:
    """Fired when a human replies "test it" -- this calls Phase 1's own
    methodology engine directly (test_novel_condition -> the same
    build_events/walk_forward/classify_status pipeline run_battery.py
    uses for the static candidates), never routing back through Sonnet:
    Sonnet's job ended when it proposed the spec, the actual validation
    is deterministic and Sonnet has no further say in the outcome.

    If it validates AND `live_coin` is given (the specific coin whose
    live occurrence prompted this test -- e.g. the coin a shock was just
    detected on), the resulting anchors are pushed immediately as a
    manual signal for THAT occurrence, tagged `signal_class`. Separately,
    and regardless of outcome, the result is recorded in the dynamic-
    candidate registry (llm_pipeline/dynamic_candidates.py) so it's
    re-tested weekly alongside the static battery from now on, and so a
    rejected condition isn't silently re-proposed later."""
    result = test_novel_condition(pending_spec, coins)
    status = result["status"]
    if status == "insufficient_data":
        return f"Tested '<b>{escape_html(pending_spec.label)}</b>': not enough historical occurrences to evaluate yet."
    record_test_result(pending_spec, status, source=signal_class)
    lines = [
        f"Tested '<b>{escape_html(pending_spec.label)}</b>': status = <b>{escape_html(status)}</b>",
        f"N={result['n']}  win_rate={result['win_rate']:.1%}  strict_win_rate={result['strict_win_rate']:.1%}",
        f"Sortino={result['sortino']:.2f}  total_expectancy={result['total_expectancy']:+.1%}  "
        f"timeout_fraction={result['timeout_fraction']:.1%}",
    ]
    if status == "validated":
        lines.append("Added to the weekly-refreshed battery going forward -- re-tested every Sunday alongside the static candidates, same as any of them.")
        if live_coin:
            push_manual_signal(
                coin=live_coin, direction=pending_spec.direction,
                tp_mult=result["live_tp_mult"], sl_mult=result["live_sl_mult"], anchors=result["live_anchors"],
                reasoning=f"Novel-condition test '{pending_spec.label}' validated on approval; "
                          f"trading the live occurrence that prompted it.",
                approved_by=approved_by, signal_class=signal_class,
            )
            lines.append(f"<b>{'━' * 4} Pushed as a live signal for {escape_html(live_coin)} now -- tagged '{escape_html(signal_class)}' {'━' * 4}</b>")
    else:
        lines.append("This does not clear the bar for live trading -- logged as tested, won't be re-proposed by Sonnet without new evidence, but will still be re-checked weekly in case conditions genuinely change.")
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
    kind = "SL" if close_profit < 0 else "TP"
    _send(f"<b>{kind} hit on {escape_html(trade_pair)} ({close_profit:+.2%})</b>\n\n"
          f"{escape_html(extract_text(response))}\n\nDocumented in the history report.")


def _answer_callback_query(callback_query_id: str) -> None:
    load_dotenv()
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    requests.post(f"https://api.telegram.org/bot{token}/answerCallbackQuery",
                  json={"callback_query_id": callback_query_id}, timeout=15)


def _get_updates(token: str, offset: int | None, timeout: int = 30) -> list[dict]:
    params = {"timeout": timeout}
    if offset is not None:
        params["offset"] = offset
    resp = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", params=params, timeout=timeout + 10)
    resp.raise_for_status()
    return resp.json().get("result", [])


def _dispatch_update(update: dict, client: Anthropic) -> None:
    if "callback_query" in update:
        cq = update["callback_query"]
        reply = handle_kpi_callback(cq["data"])
        _send(reply)
        _answer_callback_query(cq["id"])
        return

    message = update.get("message", {})
    text = (message.get("text") or "").strip()
    if not text:
        return

    if text.lower() in ("test it", "/test", "test"):
        pending = pop_pending_test()
        if pending is None:
            _send("Nothing pending to test right now.")
            return
        spec, coins, live_coin, signal_class = pending
        reply = handle_test_it_confirmation(spec, coins, approved_by="telegram_user", live_coin=live_coin, signal_class=signal_class)
        _send(reply)
        return

    if text.startswith("/"):
        command, *args = text.split()
        reply, keyboard = handle_command(command, args)
        _send(reply, keyboard)
        return

    reply = handle_natural_language(text, client)
    _send(reply)


def run_bot() -> None:
    """Long-polls Telegram for updates and dispatches them -- the live
    process behind every conversation in the README's Phase 4 mockups.
    Runs until interrupted; each update is processed and acknowledged
    (via the returned offset) before the next poll, so a crash mid-batch
    re-delivers rather than silently drops a message."""
    load_dotenv()
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    offset = None
    print("Telegram bot polling started.")
    while True:
        updates = _get_updates(token, offset)
        for update in updates:
            offset = update["update_id"] + 1
            try:
                _dispatch_update(update, client)
            except Exception as e:
                print(f"Error handling update {update.get('update_id')}: {e}")


if __name__ == "__main__":
    run_bot()
