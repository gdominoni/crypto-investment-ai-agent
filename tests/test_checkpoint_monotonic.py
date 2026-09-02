"""The replay clock must never move backward.

The invariant that would have caught an overnight run being destroyed. The
compression trigger asks at point C about point B, five days earlier, so a
pending test carries two different dates: `as_of` (the backtest's data cutoff,
B) and `resume_from` (the replay's own clock, C). They were one field, and
`resolve_pending_test` wrote it straight back as the checkpoint -- rolling the
clock back five days, walking forward into the same compression exit, proposing
again, and rolling back again.

Deterministic, single-process, and invisible in every existing test: the loop
produced ~300 near-duplicate proposals for one episode, a trade log with 218
entries dated AFTER the checkpoint that was supposedly ahead of them, and
roughly 3x the expected spend.
"""
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _isolated(isolated_replay_state):
    """Shared fixture -- patching STATE_DIR alone leaks, see tests/conftest.py."""


def test_resolving_a_pending_test_never_rewinds_the_clock(monkeypatch):
    """The exact failure. A compression proposal is pending: the clock is at
    point C, the data cutoff is point B five days earlier. Resolving it must
    resume at C."""
    from replay import engine, state

    point_b, point_c = "2019-11-21", "2019-11-26"
    state.save_checkpoint(point_c, status="waiting_for_human")
    state.save_pending_test({
        "specs": [{"label": "x", "direction": "long", "clauses": [
            {"indicator": "cpi_surprise", "op": ">=", "threshold": 1.0},
            {"indicator": "rsi_14d", "op": "<=", "threshold": 30}]}],
        "coins": ["BTCUSDT"], "live_coin": None,
        "as_of": point_b, "resume_from": point_c,
    })

    # Stub the expensive parts -- this test is about the checkpoint, not the stats.
    monkeypatch.setattr(engine, "test_novel_condition",
                        lambda *a, **k: {"status": "insufficient_data", "n_raw_triggers": 3})
    monkeypatch.setattr(engine, "_send", lambda *a, **k: True)

    engine.resolve_pending_test()

    resumed = state.load_checkpoint()["current_date"]
    assert resumed == point_c, (
        f"resumed at {resumed}, the backtest's data cutoff, instead of {point_c}, "
        "the replay's own clock -- this is the rollback loop"
    )


def test_discarding_a_pending_test_never_rewinds_either(monkeypatch):
    from replay import engine, state

    point_b, point_c = "2019-11-21", "2019-11-26"
    state.save_checkpoint(point_c, status="waiting_for_human")
    state.save_pending_test({"specs": [], "coins": ["BTCUSDT"], "live_coin": None,
                             "as_of": point_b, "resume_from": point_c})
    monkeypatch.setattr(engine, "_send", lambda *a, **k: True)

    engine.discard_pending_test()
    assert state.load_checkpoint()["current_date"] == point_c


def test_a_pending_entry_written_before_the_two_dates_split_still_resumes(monkeypatch):
    """Backward compatibility: an entry with only `as_of` must still resolve
    rather than crash on a missing key."""
    from replay import engine, state

    state.save_checkpoint("2019-11-26", status="waiting_for_human")
    state.save_pending_test({"specs": [], "coins": ["BTCUSDT"], "live_coin": None,
                             "as_of": "2019-11-21"})
    monkeypatch.setattr(engine, "_send", lambda *a, **k: True)

    engine.discard_pending_test()
    assert state.load_checkpoint()["current_date"] == "2019-11-21"


def test_the_compression_path_passes_the_clock_not_just_the_cutoff():
    """Structural: the call site must hand `advance()`'s own day to
    `_handle_assessment`, or the two dates collapse back into one."""
    import inspect

    from replay import engine
    src = inspect.getsource(engine.advance)
    assert "resume_from=d" in src, "the compression trigger must pass the replay's own day"
    assert 'as_of_b = episode["b_date"]' in src


def test_a_stored_pending_test_carries_both_dates():
    import inspect

    from replay import engine
    src = inspect.getsource(engine._handle_assessment)
    assert '"as_of"' in src and '"resume_from"' in src
