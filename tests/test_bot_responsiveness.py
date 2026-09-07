"""The bot has to answer while the daemon is doing other work.

This project's premise is that a human interacts with it only through
Telegram. The daemon polls Telegram and runs its scheduled jobs in ONE
sequential loop, so anything slow inside a command handler silences the bot
for its whole duration -- there is no second thread to answer with.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _handler_source() -> str:
    import telegram.bot as B

    return inspect.getsource(B._dispatch_update)


class TestReadCommandsNeverRecomputeTheBattery:
    """/summary called run_all() -- the whole battery -- synchronously inside
    the dispatch loop. Measured on the deployed host (one shared ARM core):
    103 minutes, during which the bot answered nothing at all and looked dead.

    Nothing was gained by it. The weekly re-validation already runs the battery
    and persists every row (run_battery's live_state["summary"]); the answer was
    on disk the whole time. The laptop hid this: 21 minutes is bad and still
    returns, so it read as slow rather than broken."""

    def test_the_dispatcher_does_not_call_run_all(self):
        src = _handler_source()
        assert "run_all()" not in src, (
            "a Telegram command runs the full battery again. On the deployed host that is "
            "~103 minutes with the bot unresponsive throughout -- read the stored result "
            "via _last_battery_result() instead.")

    def test_summary_and_details_read_the_stored_result(self):
        src = _handler_source()
        assert src.count("_last_battery_result()") >= 2, (
            "/summary and /details should both read the persisted battery result")

    def test_the_stored_result_carries_everything_the_summary_needs(self):
        """If run_battery ever stops persisting per-candidate rows, the cheap
        path silently degrades to an empty summary rather than an error."""
        import candidates.run_battery as rb

        src = inspect.getsource(rb.run_all)
        assert '"summary"' in src or "'summary'" in src, (
            "run_all no longer stores the per-candidate summary the bot reads")

    def test_a_missing_battery_result_explains_itself(self):
        """A freshly deployed host has no stored battery until its first weekly
        cycle. That must read as 'not computed yet', not as an empty answer."""
        import telegram.bot as B

        assert "weekly" in B._NO_BATTERY_YET.lower()
        assert "run_all" not in B._NO_BATTERY_YET

    def test_cached_answers_say_when_they_were_computed(self):
        import telegram.bot as B

        note = B._as_of_note("2026-09-06T14:27:19.444143+00:00")
        assert "2026-09-06 14:27" in note, "a cached answer must carry its age"
        assert B._as_of_note(None) == "", "no timestamp, no misleading claim of freshness"


def test_the_retired_word_stays_retired_in_user_facing_strings():
    """'validated' was retired in favour of 'accepted'/'CONFIRMED': no reachable
    occurrence count demonstrates an effect of interesting size, so a checkpoint
    asserts persistence and nothing stronger.

    format_candidate_details was fixed when the word was retired;
    _trigger_summary_line was missed and went on printing VALIDATED in
    /summary, the most-read message the bot sends.

    Scans the WHOLE module, not a hand-picked list of functions. The first
    version of this test inspected format_trigger_summary -- which is not where
    the string lived; it lived in the helper that formats each line -- so it
    passed while the bug sat untouched. Comments and docstrings may still
    discuss the word; only literals that can reach a human count."""
    import candidates.methodology as M

    tree = ast.parse(Path(M.__file__).read_text())
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module))
                  and getattr(n, "body", None) and isinstance(n.body[0], ast.Expr)
                  and isinstance(n.body[0].value, ast.Constant)
                  and isinstance(n.body[0].value.value, str)}
    offenders = [n.value for n in ast.walk(tree)
                 if isinstance(n, ast.Constant) and isinstance(n.value, str)
                 and id(n) not in docstrings
                 and "validated" in n.value.lower()
                 and "revalidation" not in n.value.lower()]
    assert not offenders, (
        f"a user-facing string in {Path(M.__file__).name} still uses the retired word: {offenders}")
