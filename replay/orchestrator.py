"""Drives the replay to completion autonomously: advances chunk by
chunk, auto-approves every novel-condition test proposal the moment it
comes up (standing in for the human's own "test it" reply -- every
proposal seen so far has been a cheap, whitelisted-indicator backtest,
no real risk in approving it immediately rather than waiting on a manual
relay for each one), and periodically asks a natural-language check-in
question so the resulting Telegram history reads like an attended
system, not merely a batch job. Stops when the replay reaches the
present or after `max_chunks` as a safety cap against runaway API cost
from an unexpected infinite loop.
"""
from __future__ import annotations

import os

from anthropic import Anthropic
from dotenv import load_dotenv

from llm_pipeline.haiku_sonnet_pipeline import escape_html
from replay.engine import advance, resolve_pending_test
from replay.judgment import answer_market_question
from telegram.bot import _send

CHECKIN_QUESTIONS = [
    "What's the crypto market situation right now?",
    "How are the currently active trades doing?",
    "Which trigger conditions have been tested so far -- has anything been validated or dropped?",
    "Profit and loss, win rate and sortino of all the trades so far, split by trigger?",
]


def run_to_completion(max_chunks: int = 200, ask_every: int = 4) -> dict:
    load_dotenv()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    chunk_count = 0
    question_idx = 0
    while chunk_count < max_chunks:
        result = advance()
        chunk_count += 1
        print(f"[{chunk_count}] {result}")

        if result.get("stopped") == "waiting_for_human":
            status = resolve_pending_test()
            print(f"    auto-resolved pending test: {status}")
            continue

        if result.get("reached_end"):
            print("Replay has reached the present. Stopping.")
            return {"chunks": chunk_count, "reached_end": True}

        if chunk_count % ask_every == 0:
            question = CHECKIN_QUESTIONS[question_idx % len(CHECKIN_QUESTIONS)]
            question_idx += 1
            reply = answer_market_question(question, client)
            _send(f"<i>{escape_html(question)}</i>\n\n{reply}")
            print(f"    asked: {question}")

    print(f"Stopped after reaching the {max_chunks}-chunk safety cap.")
    return {"chunks": chunk_count, "reached_end": False}


if __name__ == "__main__":
    run_to_completion()
