"""Tests for safety/execution_mode.py -- the live-trading gate."""

import pytest

from safety import execution_mode


@pytest.fixture(autouse=True)
def _isolated_state_file(tmp_path, monkeypatch):
    monkeypatch.setattr(execution_mode, "STATE_FILE", tmp_path / "execution_mode.json")
    yield


def test_defaults_to_dry_run_when_no_state_file():
    assert execution_mode.get_mode() == "dry_run"


def test_wrong_confirmation_phrase_is_rejected():
    assert execution_mode.request_live_mode("please go live") is False
    assert execution_mode.get_mode() == "dry_run"


def test_exact_confirmation_phrase_switches_to_live():
    assert execution_mode.request_live_mode("CONFIRM LIVE TRADING") is True
    assert execution_mode.get_mode() == "live"


def test_revert_to_dry_run_always_allowed():
    execution_mode.request_live_mode("CONFIRM LIVE TRADING")
    execution_mode.revert_to_dry_run()
    assert execution_mode.get_mode() == "dry_run"
