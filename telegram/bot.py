"""The two structurally separate interaction modes described in this
project's README: free-text conversation (routed to Sonnet, but every
number it cites still comes from a real computation, never invented) and
structured commands / inline keyboards (routed straight to
no LLM involved at all).
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from anthropic import Anthropic
from dotenv import load_dotenv

from candidates.atomic_json import write_json
from candidates.definitions import TRIGGER_DESCRIPTIONS, TRIGGER_NUMERIC_DEFINITIONS
from candidates.methodology import format_candidate_details, format_trigger_summary
from candidates.status_history import drop_candidate, mark_asked, record_status
from llm_pipeline.context_builder import build_context_summary, build_live_test_summary, build_technical_snapshot
from llm_pipeline.dynamic_candidates import record_test_result, registered_specs
from llm_pipeline.haiku_sonnet_pipeline import escape_html, extract_text, format_spec_clauses
from llm_pipeline.novel_condition_tester import ConditionSpec, condition_desc, format_pattern_significance, test_novel_condition
from llm_pipeline.pending_tests import discard_pending_test_by_id, pop_pending_test_by_id
from llm_pipeline import usage as _usage

SONNET_MODEL = "claude-sonnet-5"
FREQTRADE_DB_PATH = os.environ.get("FREQTRADE_DB_PATH", "execution/tradesv3.sqlite")

MARKET_CHECK_SYSTEM_PROMPT = """You are a market-check assistant for a crypto pattern-discovery system \
(NOT a trading system -- no funded position is ever opened; "accepted" only means a candidate's own \
trigger opens an observational live test). Given real technical/portfolio state and the live candidate \
battery context, answer the user's question in 2-4 sentences. Cite only the numbers given to you in the \
state below -- never invent a price, a percentage, or a live-test detail. The reader may not know this \
project's internal vocabulary (status codes like "insufficient_data", "watch"; terms like "validated" \
vs. "accepted") -- briefly explain any such term you use in plain language rather than stating it bare, \
the way you'd explain a piece of jargon to someone unfamiliar with the system. If nothing in the given \
state answers the question, say so plainly rather than guessing."""

SENT_LOG_PATH = Path(__file__).resolve().parent / "sent_messages.json"


def _log_sent_message(message_id: int) -> None:
    """Telegram's Bot API has no way to list or bulk-delete a chat's
    history after the fact -- deleting a message later requires its
    exact id, known only at send time. Logging it here is what makes
    `delete_recent_messages` possible at all."""
    log = json.loads(SENT_LOG_PATH.read_text()) if SENT_LOG_PATH.exists() else []
    log.append({"message_id": message_id, "sent_at": datetime.now(timezone.utc).isoformat()})
    write_json(SENT_LOG_PATH, log, indent=None)


TELEGRAM_MAX_MESSAGE_LENGTH = 4096
_SAFE_CHUNK_LENGTH = 3500  # real margin below Telegram's hard 4096 limit


def _chunk_message(text: str, limit: int = _SAFE_CHUNK_LENGTH) -> list[str]:
    """Splits `text` into Telegram-safe pieces -- a real, observed
    failure this exists to prevent: `/replay_summary`'s own 'still under
    test' message reached 6,880 characters once the dynamic registry
    grew large enough (96 tracked candidates), Telegram's `sendMessage`
    rejected it outright (over its real 4,096-char limit), and because
    nothing checked `_send()`'s own return value at that call site, the
    failure was completely silent -- the human just never received that
    message, with no error anywhere. `format_trigger_summary()`'s own
    docstring already reasoned about the SAME risk for one giant combined
    message and split into two for exactly that reason -- this fixes the
    case that one split alone didn't cover: either half growing past the
    limit on its own as the registry keeps growing, which two fixed
    messages can't protect against no matter how the split is drawn.
    Prefers paragraph ("\\n\\n") boundaries, then line boundaries, so an
    HTML tag opened on one line is never split across two Telegram
    messages -- every multi-line tagged block in this project's own
    messages (e.g. a `<pre>...</pre>` table) stays well
    under the limit on its own. A single oversized line (no real
    precedent yet, but not impossible -- e.g. a very large
    'no historical occurrences yet' name list) falls back to a raw
    character split as a last resort, since at that point there's no
    safe boundary left to prefer."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(paragraph) <= limit:
            current = paragraph
            continue
        for line in paragraph.split("\n"):
            candidate = f"{current}\n{line}" if current else line
            if len(candidate) <= limit:
                current = candidate
                continue
            if current:
                chunks.append(current)
                current = ""
            while len(line) > limit:
                cut = _safe_cut(line, limit)
                chunks.append(line[:cut])
                line = line[cut:]
            current = line
    if current:
        chunks.append(current)
    return chunks


def _safe_cut(line: str, limit: int) -> int:
    """Where to cut an oversized single line so the split never lands
    inside an HTML tag or between a `<b>` and its `</b>`.

    A real, reachable bug this fixes -- and the SECOND time this exact
    silent-drop failure has been found. `_insufficient_data_block()`
    emits ONE line listing every zero-N candidate, each wrapped in
    `<b>...</b>`: at 90 tracked candidates that line is 4,884 characters,
    and the previous raw `line[:limit]` cut produced chunk 1 with 65
    `<b>` against 64 `</b>` and chunk 2 with the orphaned close. Telegram
    rejects malformed HTML with a 400, and since almost no caller checks
    `_send`'s return value the message simply never arrives -- exactly
    the failure `_chunk_message` was written to prevent.

    Prefers, in order: the last tag-boundary that leaves balanced markup,
    then the last space, then the hard limit (only if a single unbroken
    token really is longer than the limit, where no safe cut exists)."""
    window = line[:limit]
    # Walk back to a point where every opened tag is also closed.
    for end in range(len(window), max(len(window) - 400, 0), -1):
        head = window[:end]
        if head.count("<") != head.count(">"):
            continue  # mid-tag
        if head.count("<b>") == head.count("</b>") and head.count("<i>") == head.count("</i>"):
            # avoid cutting mid-word when a space is close by
            sp = head.rfind(" ")
            return sp + 1 if sp > end - 80 else end
    return limit  # no safe boundary found -- one unbroken token longer than the limit


def _send(text: str, reply_markup: dict | None = None, pin: bool = False) -> bool:
    """`pin=True` best-effort pins the message after sending (Telegram's
    own `pinChatMessage`, no special permission needed in a private 1:1
    chat like this bot's -- only in groups/channels would the bot need
    'can_pin_messages'). A pin failure never turns a successful send into
    a failed one -- it's a convenience on top, not part of what "sent"
    means, so this function's return value still reflects only the send.
    `text` longer than Telegram's real limit is split (`_chunk_message`)
    and sent as several messages -- `reply_markup` (if any) attaches
    only to the LAST chunk, so buttons appear once, where a human would
    expect to act on them, not repeated on every piece. Returns True
    only if every chunk sent successfully."""
    load_dotenv()
    token, chat_id = os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    chunks = _chunk_message(text)
    all_ok = True
    for i, chunk in enumerate(chunks):
        payload = {"chat_id": chat_id, "text": chunk, "parse_mode": "HTML"}
        if reply_markup is not None and i == len(chunks) - 1:
            payload["reply_markup"] = reply_markup
        try:
            resp = requests.post(url, json=payload, timeout=15)
        except Exception as e:
            # A network blip must not take down whatever was calling us --
            # notably live_daemon's poll loop or a mid-run scheduled job.
            print(f"SEND FAILED (network): {type(e).__name__}: {e}")
            return False
        if not resp.ok:
            # Loud, and self-diagnosing. 44 of 47 call sites ignore this
            # function's return value, so a quiet failure here is
            # indistinguishable from "nothing to report" -- which is
            # precisely how an entire /replay_summary section went missing
            # once already. Callers that cannot act on it should at least
            # leave a searchable trace in the log.
            detail = resp.text[:300]
            hint = ""
            if resp.status_code == 400 and "parse" in detail.lower():
                hint = ("  <-- malformed HTML, most likely a chunk boundary splitting a tag; "
                        "see _safe_cut")
            elif resp.status_code == 429:
                hint = "  <-- rate limited by Telegram"
            print(f"SEND FAILED ({resp.status_code}) on chunk {i + 1}/{len(chunks)}: {detail}{hint}")
            all_ok = False
            continue
        message_id = resp.json()["result"]["message_id"]
        _log_sent_message(message_id)
        if pin and i == 0:
            try:
                pin_resp = requests.post(
                    f"https://api.telegram.org/bot{token}/pinChatMessage",
                    json={"chat_id": chat_id, "message_id": message_id, "disable_notification": True}, timeout=15,
                )
                if not pin_resp.ok:
                    print(f"Pin failed ({pin_resp.status_code}): {pin_resp.text[:300]} -- message itself was still sent.")
            except Exception as e:
                print(f"Pin failed: {e} -- message itself was still sent.")
    return all_ok


def delete_recent_messages() -> dict:
    """Deletes every logged bot message still within Telegram's 48h
    deletion window. Anything older is left alone -- the platform
    refuses those regardless -- and reported separately so the caller
    knows to fall back to clearing the chat by hand on the client."""
    load_dotenv()
    token, chat_id = os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]
    if not SENT_LOG_PATH.exists():
        return {"deleted": 0, "too_old": 0, "failed": 0}
    log = json.loads(SENT_LOG_PATH.read_text())
    now = datetime.now(timezone.utc)
    deleted, too_old, failed = 0, 0, 0
    remaining = []
    for entry in log:
        sent_at = datetime.fromisoformat(entry["sent_at"])
        if (now - sent_at) >= timedelta(hours=48):
            too_old += 1
            remaining.append(entry)
            continue
        resp = requests.post(f"https://api.telegram.org/bot{token}/deleteMessage",
                              json={"chat_id": chat_id, "message_id": entry["message_id"]}, timeout=15)
        if resp.ok:
            deleted += 1
        else:
            failed += 1
            remaining.append(entry)
    write_json(SENT_LOG_PATH, remaining, indent=None)
    return {"deleted": deleted, "too_old": too_old, "failed": failed}


def handle_natural_language(text: str, client: Anthropic) -> str:
    """Market checks ("how's the market", "do we have open trades") and
    conversational follow-ups -- never the reporting path, which is
    command-driven and LLM-free by design (`/summary`, `/details`, and
    their replay twins compute every number they show). Escaped, not
    bolded: this is free-form Sonnet prose with no template around it, so
    only HTML-safety is applied, not formatting Sonnet didn't ask for."""
    snapshot = build_technical_snapshot("MARKET", FREQTRADE_DB_PATH)
    live_tests = build_live_test_summary()
    context = build_context_summary()
    response = client.messages.create(
        # 2000, not 800 -- 800 was STILL observed live (2026-08-28, replay's own
        # identical answer_market_question) to truncate mid-answer on a longer,
        # itemized question. The model emits a thinking block even without
        # `thinking` being requested, and it can consume over half the budget
        # on its own (see docs/case_study/methodology-decisions.md).
        model=SONNET_MODEL, max_tokens=2000, system=MARKET_CHECK_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"USER QUESTION: {text}\n\nSTATE:\n{snapshot}\n\n{live_tests}\n\n{context}"}],
    )
    _usage.record(response, "prod.market_check", SONNET_MODEL)
    return escape_html(extract_text(response))


def handle_command(command: str, args: list[str]) -> tuple[str, dict | None]:
    """Fallback for a slash command none of the specific handlers above
    claimed. Every real command is special-cased before this point; this
    only tells a human they mistyped one.

    `/results` used to live here, opening a keyboard that queried a
    Freqtrade trade database. Under the current no-funded-position model
    no order is ever placed, so that database never receives an entry and
    the command could only ever answer "no trades" -- removed rather than
    left as a button that looks like reporting and isn't. `/summary` and
    `/replay_summary` are the real reporting commands."""
    return f"Unknown command: {command} -- send /help for the full list.", None


def handle_prune_callback(callback_data: str) -> str:
    """Answers the "Keep Testing" / "Drop from Batch" buttons sent by
    scheduler/weekly_revalidation.py for a candidate tracked 2+ years
    without ever being accepted. The decision is the human's -- Sonnet's
    earlier message was advisory only -- so this just records it: "keep"
    resets the re-ask timer (mark_asked, same as sending the prompt
    already did) so it isn't re-proposed for another 6 months, "drop"
    removes the candidate from every future battery run via
    status_history.py."""
    _, action, candidate = callback_data.split(":", 2)
    if action == "drop":
        drop_candidate(candidate)
        return f"Dropped '<b>{escape_html(candidate)}</b>' from the batch. It will no longer be tested."
    mark_asked(candidate)
    return f"Keeping '<b>{escape_html(candidate)}</b>' in the batch. Will ask again in 6 months if it still hasn't been accepted."


def handle_replay_prune_callback(callback_data: str) -> str:
    """Same as handle_prune_callback, for the replay's own isolated
    status_history.py -- keyed off the replay's current simulated date,
    not real wall-clock time."""
    from replay import state as replay_state
    from replay import status_history as replay_sh
    _, action, candidate = callback_data.split(":", 2)
    as_of = replay_state.load_checkpoint().get("current_date")
    if action == "drop":
        replay_sh.drop_candidate(candidate)
        return f"Dropped '<b>{escape_html(candidate)}</b>' from the batch. It will no longer be tested."
    if as_of:
        replay_sh.mark_asked(candidate, as_of)
    return f"Keeping '<b>{escape_html(candidate)}</b>' in the batch. Will ask again in 6 months if it still hasn't been accepted."


def handle_propose_callback(callback_data: str) -> str:
    """Answers the "Test It" / "Don't Test It" buttons attached to every
    Sonnet/shock-scan proposal -- replaces the old free-text 'test it'
    matching, which silently did nothing for any phrasing that didn't
    match one of a few exact strings (see
    llm_pipeline/pending_tests.py's own module docstring). Each button
    is bound to its own proposal's id, so multiple proposals pending at
    once can never be confused with each other or resolved to the wrong
    one."""
    _, action, pending_id = callback_data.split(":", 2)
    if action == "skip":
        found = discard_pending_test_by_id(pending_id)
        return ("Dismissed -- this condition won't be tested. Sonnet may propose it again later if it comes up."
                if found else "This proposal already expired or was already answered.")
    pending = pop_pending_test_by_id(pending_id)
    if pending is None:
        return "This proposal already expired or was already answered."
    spec, coins, live_coin, signal_class = pending
    return handle_test_it_confirmation(spec, coins, approved_by="telegram_user", live_coin=live_coin, signal_class=signal_class)


def handle_replay_propose_callback(callback_data: str) -> str:
    """Same idea as handle_propose_callback, for the replay's own
    single-slot pending test (only one can ever be pending at a time --
    the replay halts until it's answered -- so no id is needed here)."""
    _, action = callback_data.split(":", 1)
    from replay.engine import discard_pending_test, resolve_pending_test
    result = discard_pending_test() if action == "skip" else resolve_pending_test()
    return "" if result is not None else "No replay proposal is currently pending."


def handle_test_it_confirmation(pending_spec: ConditionSpec, coins: list[str], approved_by: str,
                                 live_coin: str | None = None, signal_class: str = "manual") -> str:
    """Fired when a human presses the "Test It" button (never a free-text
    reply -- see handle_propose_callback's own docstring for why) -- this
    calls Phase 1's own methodology engine directly (test_novel_condition -> the same
    build_events/walk_forward/classify_status pipeline run_battery.py
    uses for the static candidates), never routing back through Sonnet:
    Sonnet's job ended when it proposed the spec, the actual acceptance
    check is deterministic and Sonnet has no further say in the outcome.

    This project never opens a funded position (see
    docs/case_study/methodology-decisions.md) -- if `live_coin` is given
    (the specific coin whose live occurrence prompted this test), a live
    TEST opens for that occurrence regardless of the verdict below
    (testing starts at identification, not only once accepted), backdated
    via `execution.live_testing.find_backdated_entry` to the real hour
    this condition first became true if that's discoverable within the
    lookback window, not the discovery moment. Separately, and
    regardless of outcome, the result is recorded in the dynamic-
    candidate registry (llm_pipeline/dynamic_candidates.py) so it's
    re-tested weekly alongside the static battery from now on, and so a
    rejected condition isn't silently re-proposed later."""
    condition_str = f"{condition_desc(pending_spec)} → {pending_spec.direction}"
    result = test_novel_condition(pending_spec, coins)
    status = result["status"]
    if status == "insufficient_data":
        return (f"<b>Tested -- {escape_html(pending_spec.label)}</b>\n\n"
                f"({escape_html(condition_str)})\n\n"
                f"Not enough historical occurrences to evaluate yet.")
    record_test_result(pending_spec, status, source=signal_class)
    # Recorded here, not deferred to the next weekly refresh -- without this, a
    # just-discovered candidate is already registered (and can already open live
    # tests via the mechanical scan) but stays completely invisible to
    # all_latest_statuses()/Sonnet's own context for up to a week, since that
    # function skips any candidate with no status_log entry at all (a real,
    # observed case: Sonnet correctly said "not listed... so I can't state its
    # status" for a candidate that WAS already live-testing).
    record_status(pending_spec.label, status)
    pattern = result.get("pattern_significance") or {}
    lines = [
        f"<b>Historical backtest -- {escape_html(pending_spec.label)}</b>",
        "",
        f"({escape_html(condition_str)})",
        "",
        format_pattern_significance(pattern),
        "",
        f"<b>Verdict:</b> {escape_html(status.upper())}",
        "",
        f"For reference, trading this with a TP/SL structure over the same history: "
        f"N={result['n']}  win_rate={result['win_rate']:.1%}  strict_win_rate={result['strict_win_rate']:.1%}  "
        f"Sortino={result['sortino']:.2f}  total_expectancy={result['total_expectancy']:+.1%}  "
        f"timeout_fraction={result['timeout_fraction']:.1%} (informational only, doesn't affect the verdict above).",
    ]

    if pattern.get("status") == "ok":
        from candidates import status_history as _sh
        from execution import live_test_state as _lts
        horizons = _lts.load_horizons()
        horizons[pending_spec.label] = pattern["horizon"]
        _lts.save_horizons(horizons)
        _sh.record_horizon(pending_spec.label, pattern["horizon"])

    if status == "accepted":
        lines.append("")
        lines.append("This is a historical screening result, not a live track record -- the real test is ongoing: "
                      "added to the battery now, re-tested every Sunday alongside the static candidates, same as any of them.")
    if live_coin:
        from execution.live_testing import _open_live_test, find_backdated_entry
        decision_date = find_backdated_entry(pending_spec, live_coin) if status not in ("insufficient_data",) else None
        execution = _open_live_test(pending_spec.label, live_coin, pending_spec.direction, decision_date)
        if execution.get("opened"):
            when = f"backdated to {execution['entry_date'].date()}" if decision_date is not None else "opened now"
            lines.append("")
            lines.append(f"<b>Live test {when} -- {pending_spec.direction.upper()} {escape_html(live_coin)}</b>\n\n"
                         f"Held for <b>{execution['horizon']}d</b>, then resolved -- no TP/SL, tagged '{escape_html(signal_class)}'.")
    if status != "accepted":
        lines.append("")
        lines.append("No significant pattern found (or the risk profile doesn't clear the bar). Logged as "
                      "tested, won't be re-proposed by Sonnet without new evidence, but will still be re-checked "
                      "weekly in case a better TP/SL fit emerges or conditions genuinely change.")
    return "\n".join(lines)


def _trigger_description(candidate: str) -> str:
    """Same lookup as replay/engine.py's and scheduler/weekly_revalidation.py's
    own copies of this function -- duplicated rather than shared because
    importing scheduler.weekly_revalidation here would be circular (it
    already imports _send from this module)."""
    base = candidate.rsplit("_", 1)[0]
    if base in TRIGGER_DESCRIPTIONS:
        return TRIGGER_DESCRIPTIONS[base]
    for spec in registered_specs():
        if spec.label == candidate:
            return f"{condition_desc(spec)} → {spec.direction}"
    return "trigger definition not found -- treat this as missing information, do not guess at it"


def _trigger_numeric_description(candidate: str) -> str:
    """Same lookup as _trigger_description(), but the trigger's exact
    numeric definition instead of prose -- what /details and
    /replay_details need to answer "how elevated, exactly?" for a static
    candidate's own description, which is deliberately kept short and
    number-free everywhere else (proposal messages, /summary) to stay
    readable. Dynamic (Sonnet-proposed) conditions already carry their
    own numeric threshold in condition_desc(), so this is identical to
    _trigger_description() for those."""
    base = candidate.rsplit("_", 1)[0]
    if base in TRIGGER_NUMERIC_DEFINITIONS:
        return TRIGGER_NUMERIC_DEFINITIONS[base]
    for spec in registered_specs():
        if spec.label == candidate:
            return f"{condition_desc(spec)} → {spec.direction}"
    return "trigger definition not found -- treat this as missing information, do not guess at it"


def _replay_trigger_numeric_description(candidate: str) -> str:
    """Same as _trigger_numeric_description(), but against the replay's
    own isolated dynamic-candidate registry (replay/state.py) instead of
    production's (llm_pipeline/dynamic_candidates.py, via
    registered_specs()) -- these are two separate registries (see
    PROJECT_MAP.md's "Historical Replay" section). A real, live-caught
    bug: /replay_details on a candidate that only exists in the replay's
    own registry showed "trigger definition not found" because the
    lookup this function replaces only ever checked production's,
    mirrors replay/engine.py::_trigger_description's own lookup exactly."""
    base = candidate.rsplit("_", 1)[0]
    if base in TRIGGER_NUMERIC_DEFINITIONS:
        return TRIGGER_NUMERIC_DEFINITIONS[base]
    from replay import state as replay_state
    spec_dict = replay_state.load_dynamic_candidates().get(candidate)
    if spec_dict:
        return f"{format_spec_clauses(spec_dict)} → {spec_dict['direction']}"
    return "trigger definition not found -- treat this as missing information, do not guess at it"


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
        data = cq["data"]
        if data.startswith("replay_prune:"):
            reply = handle_replay_prune_callback(data)
        elif data.startswith("replay_propose:"):
            reply = handle_replay_propose_callback(data)
        elif data.startswith("prune:"):
            reply = handle_prune_callback(data)
        elif data.startswith("propose:"):
            reply = handle_propose_callback(data)
        else:
            # Every button this bot sends carries one of the four prefixes
            # above. Anything else is a stale button from an older message
            # format, or malformed data -- answer it plainly instead of
            # parsing it. The previous fallback here unpacked `data` on ":"
            # into exactly two parts, so an unrecognised callback raised
            # ValueError BEFORE _answer_callback_query() below, leaving
            # Telegram's loading spinner turning on the user's button with
            # no reply and no visible error.
            reply = "That button is from an older message and no longer does anything."
        if reply:
            # handle_replay_propose_callback's underlying functions
            # (resolve_pending_test/discard_pending_test) already send
            # their own message(s) on success and return "" here -- only
            # a non-empty reply (an error/no-op case, or every other
            # callback) still needs sending.
            _send(reply)
        _answer_callback_query(cq["id"])
        return

    message = update.get("message", {})
    text = (message.get("text") or "").strip()
    if not text:
        return

    if text.lower() in ("replay continue", "continue replay"):
        from replay.engine import advance
        result = advance()
        if result.get("stopped") == "waiting_for_human":
            _send(f"<b>[Historical Replay]</b> Paused at {result['current_date']} -- awaiting a Test It / Don't Test It decision (see the proposal above).")
        # advance() already sends its own checkpoint digest on a normal stop
        return

    if text.lower() == "/replay_summary":
        from replay.battery import run_replay_battery
        from replay import state as replay_state
        from replay import status_history as replay_sh
        checkpoint = replay_state.load_checkpoint()
        if checkpoint.get("current_date") is None:
            _send("Replay hasn't started yet -- nothing to summarize.")
            return
        as_of = pd.Timestamp(checkpoint["current_date"])
        status_summary = run_replay_battery(as_of)
        under_test, discarded = format_trigger_summary(status_summary, replay_sh.all_latest_statuses())
        _send(f"<b>{as_of.date()}</b> (as of the replay's current simulated date)\n\n{under_test}")
        _send(discarded)
        return

    if text.lower().startswith("/replay_details"):
        from replay.battery import run_replay_battery
        from replay import state as replay_state
        from replay import status_history as replay_sh
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            _send("Usage: /replay_details &lt;trigger_name&gt;  (e.g. /replay_details c2_long -- see /replay_summary for the exact names currently tracked)")
            return
        candidate = parts[1].strip()
        checkpoint = replay_state.load_checkpoint()
        if checkpoint.get("current_date") is None:
            _send("Replay hasn't started yet -- nothing to show details for.")
            return
        as_of = pd.Timestamp(checkpoint["current_date"])
        status_summary = run_replay_battery(as_of)
        row = status_summary.get(candidate)
        if row is None:
            from candidates.status_history import all_latest_statuses as production_statuses
            hint = (" This name IS tracked in production, though -- try /details instead."
                    if candidate in production_statuses() else "")
            _send(f"No candidate named '<b>{escape_html(candidate)}</b>' found in the replay's current battery. Check /replay_summary for exact names.{hint}")
            return
        horizon = replay_state.load_horizons().get(candidate)
        milestone = replay_sh.all_latest_statuses().get(candidate, {})
        live_entry = replay_state.load_battery_status().get("candidates", {}).get(candidate, {})
        _send(format_candidate_details(candidate, row, definition=_replay_trigger_numeric_description(candidate), horizon=horizon,
                                        milestone=milestone, tp_mult=live_entry.get("tp_mult"), sl_mult=live_entry.get("sl_mult")))
        return

    if text.lower() == "/replay_status":
        from replay import state
        checkpoint = state.load_checkpoint()
        battery = state.load_battery_status()
        log = state.load_trade_log()
        lines = [
            f"<b>Historical Replay status</b>",
            f"Simulated date: {checkpoint['current_date'] or 'not started'}  (status: {checkpoint['status']})",
            f"Accepted candidates as of that date: {', '.join(battery.get('candidates', {}).keys()) or 'none'}",
            f"Live tests so far: {len(log)}" + (
                f", mean forward return (resolved): {sum(t['forward_return'] for t in log if t['status'] == 'closed') / max(1, sum(1 for t in log if t['status'] == 'closed')):+.2%}"
                if any(t["status"] == "closed" for t in log) else ""
            ),
        ]
        _send("\n".join(lines))
        return

    if text.lower().startswith("/details"):
        from candidates.run_battery import run_all
        from candidates.status_history import all_latest_statuses
        from execution.live_test_state import load_horizons
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            _send("Usage: /details &lt;trigger_name&gt;  (e.g. /details c2_long -- see /summary for the exact names currently tracked)")
            return
        candidate = parts[1].strip()
        result, live_state, _meta = run_all()
        status_summary = result.set_index("candidate").to_dict(orient="index") if len(result) else {}
        row = status_summary.get(candidate)
        if row is None:
            from replay import status_history as replay_sh
            hint = (" This name IS tracked in the historical replay, though -- try /replay_details instead."
                    if candidate in replay_sh.all_latest_statuses() else "")
            _send(f"No candidate named '<b>{escape_html(candidate)}</b>' found in the current battery. Check /summary for exact names.{hint}")
            return
        horizon = load_horizons().get(candidate)
        milestone = all_latest_statuses().get(candidate, {})
        live_entry = live_state.get("candidates", {}).get(candidate, {})
        _send(format_candidate_details(candidate, row, definition=_trigger_numeric_description(candidate), horizon=horizon,
                                        milestone=milestone, tp_mult=live_entry.get("tp_mult"), sl_mult=live_entry.get("sl_mult")))
        return

    if text.lower() == "/summary":
        from candidates.run_battery import run_all
        from candidates.status_history import all_latest_statuses
        result, _live_state, _meta = run_all()
        status_summary = result.set_index("candidate").to_dict(orient="index") if len(result) else {}
        under_test, discarded = format_trigger_summary(status_summary, all_latest_statuses())
        _send(under_test)
        _send(discarded)
        return

    if text.lower() in ("/help", "/start"):
        # Pinned so it stays at the top of the chat as a standing reference --
        # this message never changes based on live state (unlike every other
        # command's reply), so it's the one thing worth keeping in view rather
        # than scrolled past.
        lines = [
            "<b>Standard commands</b>",
            "",
            "/summary -- current status of every tracked trigger (production): accepted, watch, rejected, insufficient data. Recomputed fresh, no LLM call.",
            "/details &lt;trigger_name&gt; -- full numeric breakdown for one trigger from /summary: the exact threshold that defines it (e.g. \"funding z-score below -2.0\", not just \"elevated funding\"), N, p-value, concentration percentages, risk-path ratio, and why it isn't accepted if it isn't. Example: /details c2_long. No LLM call.",
            "/replay_summary -- same as /summary, for the historical replay (a walk-forward simulation through real past data, used to validate the system and build an initial live-test track record before going live).",
            "/replay_details &lt;trigger_name&gt; -- same as /details, for the historical replay.",
            "/replay_status -- quick snapshot of the replay's simulated date, accepted candidates, live-test count.",
            "\"replay continue\" -- advances the historical replay.",
            "/usage -- real recorded Anthropic token usage and cost so far, by call site. Measured, never estimated. No LLM call.",
            "",
            "Anything else you type is answered in plain language (Sonnet), grounded only in the real numbers above -- ask about the market, a specific candidate, recent results, anything.",
            "",
            "<b>Buttons you'll see on their own</b>",
            "\"Test It\" / \"Don't Test It\" -- on a new condition Sonnet proposes.",
            "\"Keep Testing\" / \"Drop from Batch\" -- on a keep-or-drop or milestone checkpoint.",
            "",
            "<b>Freqtrade hyperopt cross-check -- run from your own terminal, never from here</b>",
            "Deliberately local-only (keeps the live host cheap to run, see PROJECT_MAP.md's \"Cost Optimization\" Part 3). From the project root:",
            "",
            "<code>python3 -m execution.hyperopt_runner</code>",
            "  -- runs every tracked candidate, 50 epochs, 2018-01-01 to now (the defaults).",
            "<code>python3 -m execution.hyperopt_runner NAME1 NAME2</code>",
            "  -- only those candidates.",
            "<code>python3 -m execution.hyperopt_runner --epochs 30</code>",
            "  -- fewer epochs, faster.",
            "<code>python3 -m execution.hyperopt_runner --timerange 20220101-</code>",
            "  -- restrict the date range.",
            "<code>python3 -m execution.hyperopt_runner --help</code>",
            "  -- full option reference.",
            "",
            "Writes execution/hyperopt_results.json -- copy or push that one file to wherever this bot runs when you're done; nothing else needs to move.",
        ]
        _send("\n".join(lines), pin=True)
        return

    if text.lower().startswith("/usage"):
        # Real recorded token counts, never an estimate -- see llm_pipeline/usage.py
        _send(f"<b>Anthropic usage so far</b>\n\n<pre>{escape_html(_usage.summary())}</pre>")
        return

    if text.startswith("/"):
        command, *args = text.split()
        reply, keyboard = handle_command(command, args)
        _send(reply, keyboard)
        return

    from replay import state as replay_state
    if replay_state.load_checkpoint().get("current_date") is not None:
        # A replay is in progress -- a market question almost certainly
        # means "as of where the replay currently stands," not
        # production's own (likely empty) real state.
        from replay.judgment import answer_market_question
        _send(answer_market_question(text, client))
        return

    reply = handle_natural_language(text, client)
    _send(reply)


def run_bot() -> None:
    """Long-polls Telegram for updates and dispatches them -- the live
    process behind every conversation in the README's Phase 4 mockups.
    Runs until interrupted; each update is processed and acknowledged
    (via the returned offset) before the next poll, so a crash mid-batch
    re-delivers rather than silently drops a message. `_get_updates()`
    itself is wrapped too -- a real, observed case: a stray extra
    `getUpdates` call from outside this loop (Telegram allows only one
    active long-poll per bot token) returned an HTTP 409 Conflict on the
    very next poll and crashed this whole process, previously undetected
    since this exact call was never protected (see
    docs/case_study/methodology-decisions.md). Mirrors
    scheduler/live_daemon.py's own polling resilience, which already had
    this -- `run_bot()` didn't, and this is the gap that actually
    surfaced it."""
    load_dotenv()
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    offset = None
    print("Telegram bot polling started.")
    while True:
        try:
            updates = _get_updates(token, offset)
        except Exception as e:
            print(f"Telegram poll failed, will retry: {e}")
            time.sleep(10)
            continue
        for update in updates:
            offset = update["update_id"] + 1
            try:
                _dispatch_update(update, client)
            except Exception as e:
                print(f"Error handling update {update.get('update_id')}: {e}")


if __name__ == "__main__":
    run_bot()
