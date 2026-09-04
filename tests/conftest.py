import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Every module-level path constant that points into a real state directory, with
# the module that owns it. Redirected wholesale by `isolated_replay_state` below.
#
# WHY A LIST AND NOT JUST STATE_DIR. `replay/state.py` computes each path once at
# import time (`CHECKPOINT_PATH = STATE_DIR / "checkpoint.json"`), so patching
# `STATE_DIR` afterwards changes nothing for any of them -- the constants already
# hold absolute paths into the real directory. A test that patched only
# `STATE_DIR` looked isolated, passed, and wrote a dummy candidate called "x"
# into the live replay's own `dynamic_candidates.json`. That is not merely untidy:
# a test run during a live replay would corrupt a nine-year run in progress.
#
# `replay/status_history.py` is worse, holding its own `HISTORY_PATH` that never
# referenced `STATE_DIR` at all, so no amount of patching that one would have
# helped.
_STATE_PATH_CONSTANTS = [
    ("replay.state", "STATE_DIR", None),          # None -> the directory itself
    ("replay.state", "CHECKPOINT_PATH", "checkpoint.json"),
    ("replay.state", "BATTERY_STATUS_PATH", "battery_status.json"),
    ("replay.state", "TRADE_LOG_PATH", "trade_log.json"),
    ("replay.state", "PENDING_TEST_PATH", "pending_test.json"),
    ("replay.state", "DYNAMIC_CANDIDATES_PATH", "dynamic_candidates.json"),
    ("replay.state", "PENDING_REVEALS_PATH", "pending_reveals.json"),
    ("replay.state", "HORIZONS_PATH", "horizons.json"),
    ("replay.state", "LOCK_PATH", "replay.lock"),
    ("replay.status_history", "HISTORY_PATH", "status_history.json"),
    # The Telegram outbox is replay state in every sense that matters here: it
    # holds messages a run produced and could not yet deliver. Leaving it
    # unpatched meant `run_to_completion`'s end-of-run flush picked up the LIVE
    # queue -- 678 real messages from a nine-hour run in progress -- and sat
    # waiting up to WAIT_OUT_BAN_S (20h) to send them. The test hung, and had
    # the rate limit lapsed it would have sent that queue for real, from a test.
    ("telegram.bot", "_OUTBOX_PATH", "telegram_outbox.json"),
]


@pytest.fixture
def isolated_replay_state(tmp_path, monkeypatch):
    """Redirect every replay state path into a temp directory.

    Use this instead of patching `STATE_DIR` by hand -- see the note above for
    why that is not enough. Returns the temp directory, for tests that want to
    inspect what was written."""
    import importlib

    for module_name, attr, filename in _STATE_PATH_CONSTANTS:
        module = importlib.import_module(module_name)
        if not hasattr(module, attr):
            continue
        target = tmp_path if filename is None else tmp_path / filename
        monkeypatch.setattr(module, attr, target)
    return tmp_path


@pytest.fixture(autouse=True)
def _fail_if_a_test_touches_real_replay_state():
    """Backstop: fail loudly if any test writes into the real state directory.

    The isolation fixture above is opt-in, and an opt-in guard protects only the
    tests that remember to ask for it. This one runs everywhere and turns a
    silent corruption of live state into a failing test naming the file.

    SKIPPED while a replay is actually running. The check is an mtime diff, so
    it cannot tell a test's write from the replay's own -- and a replay writes
    its checkpoint after EVERY simulated day, which made the whole suite fail
    with "test wrote into the REAL replay state" whenever a run was in progress.
    That is a false accusation, and one that trains a reader to ignore this
    assertion, which is the only thing it must never do. Detected via the lock
    file the orchestrator holds for the duration of a run."""
    root = Path(__file__).resolve().parent.parent
    real_dir = root / "replay" / "state"
    if _a_replay_is_running(real_dir / "replay.lock"):
        yield
        return

    def snapshot():
        # The Telegram outbox is watched alongside the state directory: it holds
        # undelivered evidence from a real run, and a test that drained or
        # overwrote it would destroy exactly what it exists to protect.
        files = list(real_dir.glob("*")) if real_dir.exists() else []
        outbox = root / "telegram" / "outbox.json"
        if outbox.exists():
            files.append(outbox)
        return {p: p.stat().st_mtime_ns for p in files}

    before = snapshot()
    yield
    after = snapshot()
    touched = [p.name for p in set(before) | set(after) if before.get(p) != after.get(p)]
    assert not touched, (
        f"test wrote into the REAL replay state: {sorted(touched)}. "
        f"Use the `isolated_replay_state` fixture -- patching STATE_DIR alone is "
        f"not enough, the per-file path constants are computed at import."
    )


def _a_replay_is_running(lock_path: Path) -> bool:
    """Whether a live replay holds the lock. A stale lock (the process is gone)
    does not count -- otherwise a crashed run would disable this guard
    permanently, which is the failure mode worth avoiding here."""
    import os

    try:
        pid = int(lock_path.read_text().strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True
