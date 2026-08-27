"""Unit tests for candidates/status_history.py's prune/shutdown logic.
Every test redirects HISTORY_PATH/SHUTDOWN_FLAG_PATH to a temp file
(_isolated_state, autouse) so these never touch the real, committed
candidates/status_history.json.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from candidates import status_history as sh


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(sh, "HISTORY_PATH", tmp_path / "status_history.json")
    monkeypatch.setattr(sh, "SHUTDOWN_FLAG_PATH", tmp_path / "SHUTDOWN")


def _age_first_tracked(candidate: str, days_ago: int) -> None:
    history = sh.load_history()
    history[candidate]["first_tracked_at"] = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    sh._save(history)


def test_record_status_sets_first_tracked_at_only_once():
    sh.record_status("c1_long", "rejected")
    first = sh.load_history()["c1_long"]["first_tracked_at"]
    sh.record_status("c1_long", "watch")
    assert sh.load_history()["c1_long"]["first_tracked_at"] == first


def test_drop_candidate_marks_it_dropped():
    sh.record_status("c1_long", "rejected")
    assert not sh.is_dropped("c1_long")
    sh.drop_candidate("c1_long")
    assert sh.is_dropped("c1_long")


def test_old_never_validated_candidate_is_due_for_a_prune_decision():
    sh.record_status("c1_long", "rejected")
    _age_first_tracked("c1_long", 800)
    assert "c1_long" in sh.candidates_due_for_prune_decision()


def test_recently_tracked_candidate_is_not_yet_due():
    sh.record_status("c1_long", "rejected")  # first_tracked_at defaults to now
    assert "c1_long" not in sh.candidates_due_for_prune_decision()


def test_candidate_validated_at_least_once_is_never_due_even_if_it_later_degrades():
    sh.record_status("c1_long", "watch")
    sh.record_status("c1_long", "validated")
    _age_first_tracked("c1_long", 800)
    sh.record_status("c1_long", "rejected")
    assert "c1_long" not in sh.candidates_due_for_prune_decision()


def test_recently_asked_candidate_is_not_re_asked():
    sh.record_status("c1_long", "rejected")
    _age_first_tracked("c1_long", 800)
    sh.mark_asked("c1_long")
    assert "c1_long" not in sh.candidates_due_for_prune_decision()


def test_dropped_candidate_is_never_due_for_a_prune_decision():
    sh.record_status("c1_long", "rejected")
    _age_first_tracked("c1_long", 800)
    sh.drop_candidate("c1_long")
    assert "c1_long" not in sh.candidates_due_for_prune_decision()


def test_shutdown_flag_round_trip():
    assert not sh.is_shut_down()
    sh.trigger_shutdown("no candidates left")
    assert sh.is_shut_down()
