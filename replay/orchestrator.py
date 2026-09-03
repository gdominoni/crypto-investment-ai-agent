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
from replay import state
from replay.engine import advance, resolve_pending_test
from replay.judgment import answer_market_question
from telegram.bot import _send

CHECKIN_QUESTIONS = [
    "What's the crypto market situation right now?",
    "How are the currently active trades doing?",
    "Which trigger conditions have been tested so far -- has anything been validated or dropped?",
    "Profit and loss, win rate and sortino of all the trades so far, split by trigger?",
]


def run_to_completion(max_chunks: int = 1500, ask_every: int = 4) -> dict:
    # Exclusive lock on this replay's own state, for the whole run -- kept as a
    # precaution, not because it's what fixed a real incident. An overnight run
    # was once corrupted by a deterministic single-process checkpoint-rollback
    # loop (a resolved test's pending "as_of" being written back as the clock,
    # not two processes racing); this lock would not have prevented that. See
    # state.acquire_replay_lock's docstring and
    # docs/case_study/methodology-decisions.md ("The replay clock and the
    # backtest's data cutoff are two different dates") for the actual cause and
    # fix.
    state.acquire_replay_lock()
    try:
        return _run_to_completion_locked(max_chunks, ask_every)
    finally:
        state.release_replay_lock()


def _run_to_completion_locked(max_chunks: int, ask_every: int) -> dict:
    load_dotenv()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # 200 was too low and stopped a real run at 2023-03 with budget still
    # available. A chunk ends early on every proposal, so the cap is consumed by
    # DISCOVERY, not by elapsed time: 197 proposals plus ~110 thirty-day chunks
    # exhausted it well before the timeline did. The cap exists only to bound a
    # runaway loop, and the api_failure halt now covers the failure mode that
    # actually needed bounding, so it can be generous.
    chunk_count = 0
    question_idx = 0
    while chunk_count < max_chunks:
        result = advance()
        chunk_count += 1
        print(f"[{chunk_count}] {result}")

        # A systemic API failure (out of credit, rejected key, unreachable API)
        # already checkpointed a clean resume point and alerted on Telegram.
        # Stop here rather than spending the remaining chunks retrying a call
        # that cannot succeed -- and, more importantly, rather than falling
        # through to the check-in question below, whose own failure was not
        # caught and would end the run in a traceback instead of a resume point.
        if result.get("stopped") == "api_failure":
            print(f"Replay halted: {result.get('reason')}. "
                  f"Resume from {result.get('current_date')} with `replay continue`.")
            return {"chunks": chunk_count, "reached_end": False,
                    "halted": result.get("reason"), "resume_from": result.get("current_date")}

        if result.get("stopped") == "waiting_for_human":
            try:
                status = resolve_pending_test()
            except Exception as e:
                from replay.engine import _is_systemic_api_failure
                if _is_systemic_api_failure(e):
                    print(f"Replay halted while resolving a pending test: {e}")
                    return {"chunks": chunk_count, "reached_end": False, "halted": str(e)}
                raise
            print(f"    auto-resolved pending test: {status}")
            continue

        if result.get("reached_end"):
            print("Replay has reached the present. Stopping.")
            # The hyperopt cross-check is deliberately NOT run inline -- it
            # optimises over the whole history to the present, so a message dated
            # 2020 quoting it would be showing the future, and at several minutes
            # a candidate it would add 10-16 hours to the run. See
            # replay/post_replay_hyperopt.py.
            print("\nNext, if you want the independent Freqtrade cross-check on what survived:")
            print("    python3 -m replay.post_replay_hyperopt --dry-run   # see what it would run")
            print("    python3 -m replay.post_replay_hyperopt             # run it (local, hours)")
            return {"chunks": chunk_count, "reached_end": True}

        if chunk_count % ask_every == 0:
            question = CHECKIN_QUESTIONS[question_idx % len(CHECKIN_QUESTIONS)]
            question_idx += 1
            # Cosmetic: this exists so the Telegram history reads like an
            # attended system. It must never be the thing that ends a run whose
            # actual work is succeeding -- and if the API is genuinely gone, the
            # replay's own halt path handles it on the next chunk with a proper
            # resume point.
            try:
                reply = answer_market_question(question, client)
                _send(f"<i>{escape_html(question)}</i>\n\n{reply}")
                print(f"    asked: {question}")
            except Exception as e:
                print(f"    check-in question failed, continuing anyway: {e}")

    print(f"Stopped after reaching the {max_chunks}-chunk safety cap.")
    return {"chunks": chunk_count, "reached_end": False}


if __name__ == "__main__":
    run_to_completion()
