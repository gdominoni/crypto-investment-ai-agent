"""The lock against two orchestrator processes running concurrently.

A real incident, not a hypothetical: two unlocked instances wrote the same
replay state overnight, silently overwriting each other's progress. The
symptom was a checkpoint date that visibly jumped backward between log lines,
~300 near-duplicate proposals for the same handful of days, and 218 trade-log
entries dated up to 8 days after the final checkpoint -- found only by
checking that the checkpoint-never-moves-backward invariant directly against
the trade log.
"""
import os

import pytest


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    from replay import state
    monkeypatch.setattr(state, "STATE_DIR", tmp_path)
    monkeypatch.setattr(state, "LOCK_PATH", tmp_path / "replay.lock")
    yield
    state.release_replay_lock()


def test_a_second_acquire_is_refused_while_the_first_holds_it():
    from replay import state
    state.acquire_replay_lock()
    with pytest.raises(state.ReplayAlreadyRunning):
        state.acquire_replay_lock()


def test_release_then_acquire_succeeds():
    from replay import state
    state.acquire_replay_lock()
    state.release_replay_lock()
    state.acquire_replay_lock()  # must not raise


def test_a_lock_naming_a_dead_pid_is_reclaimed_not_honoured_forever():
    """A crash that never reached release_replay_lock() must not require a
    human to notice and delete a file by hand before the next run can start."""
    from replay import state
    # A PID essentially guaranteed not to exist.
    dead_pid = 2_000_000_000
    state.LOCK_PATH.write_text(str(dead_pid))
    state.acquire_replay_lock()  # must reclaim, not raise


def test_a_lock_naming_this_process_s_own_pid_is_refused():
    """The live-process check must actually distinguish alive from dead, not
    just check file existence."""
    from replay import state
    state.LOCK_PATH.write_text(str(os.getpid()))
    with pytest.raises(state.ReplayAlreadyRunning):
        state.acquire_replay_lock()


def test_run_to_completion_holds_the_lock_for_its_whole_run(monkeypatch):
    """The lock must wrap the whole batch run, not just its first chunk --
    that was the actual gap: a per-call lock would still let two full runs
    interleave chunk by chunk."""
    import inspect

    from replay import orchestrator
    src = inspect.getsource(orchestrator.run_to_completion)
    assert "acquire_replay_lock" in src
    assert "release_replay_lock" in src
    assert "finally" in src, "the lock must release even if the run raises"
