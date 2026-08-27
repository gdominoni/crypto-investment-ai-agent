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
