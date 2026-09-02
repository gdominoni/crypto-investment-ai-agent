"""The lock against two orchestrator processes running concurrently.

Precautionary. None of the replay's state files has any concurrency
protection -- every writer does read-modify-write on whole-file JSON -- so two
orchestrators against the same directory would silently overwrite each other.
This makes that impossible rather than merely unlikely.

Not to be confused with the overnight corruption that actually happened, which
was a single process rolling its own checkpoint backward and which this lock
would not have prevented -- see tests/test_checkpoint_monotonic.py.
"""
import os

import pytest


@pytest.fixture(autouse=True)
def _isolated(isolated_replay_state):
    from replay import state
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
    """The lock must wrap the whole batch run, not just its first chunk: a
    per-call lock would still let two full runs interleave chunk by chunk."""
    import inspect

    from replay import orchestrator
    src = inspect.getsource(orchestrator.run_to_completion)
    assert "acquire_replay_lock" in src
    assert "release_replay_lock" in src
    assert "finally" in src, "the lock must release even if the run raises"
