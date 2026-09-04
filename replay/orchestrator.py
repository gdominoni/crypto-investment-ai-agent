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
from telegram.bot import _send, drain_outbox, flush_outbox, outbox_pending

CHECKIN_QUESTIONS = [
    "What's the crypto market situation right now?",
    "How are the currently active trades doing?",
    "Which trigger conditions have been tested so far -- has anything been confirmed or dropped?",
    "Profit and loss, win rate and sortino of all the trades so far, split by trigger?",
]

# Asked only once the replay is inside its last stretch of simulated time, and
# repeated, so the answers land against a full registry rather than a half-built
# one. These two are the pair a reader of the case study actually needs: the
# list, then one trigger in depth -- and they are what the README's screenshots
# are taken from, which is why they fire late and more than once rather than
# being sampled at whatever date the rotation happened to reach.
FINAL_QUESTIONS = [
    "Send me the list of all the triggers that have been repeating so far, with how many "
    "confirmations each has and how many would be needed for statistical power.",
    # Deliberately "furthest along", NOT "best performing". Asking for the best
    # by outcome would select on the result in the one message the case study
    # puts on display -- the same error as ranking the digest by success rate,
    # which put n=6 candidates with a 100% hit rate and a `rejected` status on
    # top. Furthest along is an ordering the data cannot flatter.
    "Give me the full detail of the trigger that is furthest along in its confirmation count: "
    "what the condition actually tests, and its last several dated occurrences with their outcomes.",
]
# How far from the end the final questions start, and how many times each is
# asked. Two rounds gives a screenshot a second chance if one answer comes back
# poorly worded, without spending much: these are Q&A-path calls.
FINAL_QUESTION_WINDOW_DAYS = 210
FINAL_QUESTION_ROUNDS = 2


# How long the end-of-run flush will sit waiting for a Telegram ban to lift.
# Set above the longest `retry_after` this project has actually been given
# (63,364s / 17.6h): the run's own compute is ~10 hours, so a ban that starts
# early can easily outlast it, and a queue nobody drains is the evidence lost.
WAIT_OUT_BAN_S = 20 * 3600


def _finish(max_wait_s: float) -> None:
    """Deliver whatever is still queued before the process exits.

    Without this the outbox is a leak dressed as a safety net: `drain_outbox`
    only runs from inside `_send`, so a run that ends with messages queued has
    nothing left to push them out. Says plainly what could not be delivered
    rather than letting a silent gap look like a complete record."""
    pending = outbox_pending()
    if not pending:
        return
    print(f"Flushing {pending} queued Telegram message(s) -- waiting out a rate limit if needed...")
    left = flush_outbox(max_wait_s=max_wait_s)
    if left:
        print(f"WARNING: {left} message(s) still undelivered. The replay's record is INCOMPLETE. "
              f"Run `python3 -m telegram.flush` once Telegram is reachable to finish sending them.")
    else:
        print("Outbox empty -- the Telegram record is complete.")


def _final_questions_due(current_date: str | None, already_asked: list[str]) -> str | None:
    """The next final question owed at this simulated date, or None.

    Gated on how close the simulated clock is to the present rather than on a
    chunk count: the number of chunks is not known in advance (a proposal ends
    one early, so it varies with discovery), and what these questions need is a
    FULL registry, which is a property of the date. Each is asked
    FINAL_QUESTION_ROUNDS times, alternating, so the two interleave rather than
    firing one after the other in the same minute."""
    if current_date is None:
        return None
    import pandas as pd

    days_left = (pd.Timestamp.today().normalize() - pd.Timestamp(current_date)).days
    if days_left > FINAL_QUESTION_WINDOW_DAYS:
        return None
    for round_i in range(FINAL_QUESTION_ROUNDS):
        for q in FINAL_QUESTIONS:
            if already_asked.count(q) <= round_i:
                return q
    return None


def run_to_completion(max_chunks: int = 1500, ask_every: int = 8) -> dict:
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
    final_asked: list[str] = []
    while chunk_count < max_chunks:
        result = advance()
        chunk_count += 1
        # Opportunistic, and a no-op while a ban is known to be active: costs
        # nothing per chunk and means a lifted ban starts clearing the backlog
        # immediately rather than at the next message the replay happens to send.
        drain_outbox()
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
            _finish(max_wait_s=WAIT_OUT_BAN_S)
            return {"chunks": chunk_count, "reached_end": True}

        # The two case-study questions, once the run is inside its final
        # stretch. Checked before the rotation below so they cannot be crowded
        # out by it, and driven by the SIMULATED date rather than a chunk count,
        # which is not known in advance and varies with how often a proposal
        # ends a chunk early.
        due_final = _final_questions_due(result.get("current_date"), final_asked)
        if due_final is not None:
            question = due_final
            final_asked.append(due_final)
        elif chunk_count % ask_every == 0:
            question = CHECKIN_QUESTIONS[question_idx % len(CHECKIN_QUESTIONS)]
            question_idx += 1
        else:
            question = None
        if question is not None:
            # Cosmetic for the rotation, load-bearing for the final two: this
            # exists so the Telegram history reads like an
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
    _finish(max_wait_s=WAIT_OUT_BAN_S)
    return {"chunks": chunk_count, "reached_end": False}


if __name__ == "__main__":
    run_to_completion()
