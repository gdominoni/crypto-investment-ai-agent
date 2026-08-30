"""Integration test for candidates/run_battery.py's per-candidate failure
isolation -- runs the REAL battery against the real historical data (a
few seconds, unlike the fast synthetic unit tests elsewhere in tests/),
with one candidate deliberately broken, to prove one bad candidate can't
take down every other candidate's already-computed result.
"""
from __future__ import annotations

import pytest

import candidates.run_battery as rb
import candidates.status_history as sh


@pytest.fixture(autouse=True)
def _isolated_status_history(tmp_path, monkeypatch):
    monkeypatch.setattr(sh, "HISTORY_PATH", tmp_path / "status_history.json")
    monkeypatch.setattr(sh, "SHUTDOWN_FLAG_PATH", tmp_path / "SHUTDOWN")


def test_one_broken_candidate_does_not_cost_the_others_their_result(monkeypatch):
    broken = dict(rb.CANDIDATE_DIRECTIONS)
    broken["zzz_test_crash"] = "long"  # no such column in build_triggers()'s output -> KeyError inside the loop
    monkeypatch.setattr(rb, "CANDIDATE_DIRECTIONS", broken)

    result, live_state, meta = rb.run_all()

    assert "zzz_test_crash" in meta["failed_candidates"]
    broken_row = result[result["candidate"] == "zzz_test_crash"].iloc[0]
    assert broken_row["status"] == "error"

    # every real candidate must still have produced a real, non-error result
    real_candidates = {"c1_long", "c1_short", "c2_long", "c2_short", "c6_long", "c6_short"}
    assert real_candidates <= set(result["candidate"])
    for name in real_candidates:
        assert result[result["candidate"] == name].iloc[0]["status"] != "error"


class TestTriggerDefinitionsMatchTheCode:
    """The prose `/details` shows a human must not drift from the thresholds
    the code actually uses. It used to be hand-copied in both places."""

    def test_every_threshold_in_the_description_comes_from_the_constant(self):
        from candidates.definitions import (C1_FUNDING_Z, C2_RANGE_MULT, C6_EFFICIENCY_RATIO,
                                             C6_VOLUME_MULT, TRIGGER_NUMERIC_DEFINITIONS)
        assert f"-{C1_FUNDING_Z}" in TRIGGER_NUMERIC_DEFINITIONS["c1"]
        assert f"{C2_RANGE_MULT}x" in TRIGGER_NUMERIC_DEFINITIONS["c2"]
        assert f"{C6_EFFICIENCY_RATIO:.2f}" in TRIGGER_NUMERIC_DEFINITIONS["c6"]
        assert f"{C6_VOLUME_MULT}x" in TRIGGER_NUMERIC_DEFINITIONS["c6"]

    def test_the_constants_are_what_compute_triggers_actually_applies(self):
        """Guards the other direction: a constant could be renamed in the
        description while compute_triggers kept a literal."""
        import inspect

        from candidates import definitions as D
        src = inspect.getsource(D.compute_triggers)
        for name in ("C1_FUNDING_Z", "C2_RANGE_MULT", "C6_EFFICIENCY_RATIO", "C6_VOLUME_MULT"):
            assert name in src, f"{name} is defined but compute_triggers uses a literal instead"
