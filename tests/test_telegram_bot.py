"""Tests for orchestrator/telegram_bot.py -- specifically that the
live-mode confirmation check happens against the raw message text,
unmodified, before anything else runs. This is the concrete regression
guard for the Phase 3 decision that Haiku never decides whether a
confirmation phrase "counts."
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from orchestrator import telegram_bot
from safety import execution_mode


@pytest.fixture(autouse=True)
def _isolated_state_file(tmp_path, monkeypatch):
    monkeypatch.setattr(execution_mode, "STATE_FILE", tmp_path / "execution_mode.json")
    yield


def _fake_update(text: str) -> SimpleNamespace:
    message = SimpleNamespace(text=text, reply_text=AsyncMock())
    return SimpleNamespace(message=message)


def test_exact_phrase_switches_to_live():
    update = _fake_update("CONFIRM LIVE TRADING")
    asyncio.run(telegram_bot.handle_text(update, context=None))
    assert execution_mode.get_mode() == "live"
    reply = update.message.reply_text.await_args.args[0]
    assert "activated" in reply.lower()


def test_wrong_case_does_not_switch():
    update = _fake_update("confirm live trading")
    asyncio.run(telegram_bot.handle_text(update, context=None))
    assert execution_mode.get_mode() == "dry_run"


def test_phrase_embedded_in_other_text_does_not_switch():
    update = _fake_update("please CONFIRM LIVE TRADING now")
    asyncio.run(telegram_bot.handle_text(update, context=None))
    assert execution_mode.get_mode() == "dry_run"


def test_unrecognized_text_gets_static_instructions():
    update = _fake_update("hey what's up")
    asyncio.run(telegram_bot.handle_text(update, context=None))
    reply = update.message.reply_text.await_args.args[0]
    assert "Unrecognized" in reply


def test_dry_run_command_always_succeeds():
    execution_mode.request_live_mode("CONFIRM LIVE TRADING")
    update = _fake_update("/dry_run")
    asyncio.run(telegram_bot.dry_run(update, context=None))
    assert execution_mode.get_mode() == "dry_run"
