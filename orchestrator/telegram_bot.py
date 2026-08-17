"""Telegram bot: the human-facing control surface for the orchestrator.

Command handlers are thin wrappers around orchestrator/status.py and
safety/execution_mode.py. The live-mode confirmation check happens
against the raw message text, before any Haiku involvement -- exactly as
decided in Phase 3 (safety/execution_mode.py's docstring): Haiku's role
is to relay and explain, never to decide whether a phrase was "close
enough." There is deliberately no open-ended Haiku chat fallback here --
Haiku is used only to *format* already-computed, already-safe data
(status_formatter.py), never to interpret free text into a decision. That
keeps the LLM's role strictly read-only with respect to system state.

Run locally with:
    python -m orchestrator.telegram_bot
"""

import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from orchestrator.status import gather_system_status
from orchestrator.status_formatter import format_status_message
from safety.execution_mode import LIVE_CONFIRMATION_PHRASE, get_mode, request_live_mode, revert_to_dry_run

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Crypto Investment AI Agent online.\n"
        f"Current mode: {get_mode().upper()}\n"
        "Send /status for a full report, or /dry_run to revert to paper trading."
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    system_status = gather_system_status()
    message = format_status_message(system_status)
    await update.message.reply_text(message)


async def dry_run(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    revert_to_dry_run()
    await update.message.reply_text("Reverted to DRY-RUN mode. No confirmation needed to go safer.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text or ""

    if text.strip() == LIVE_CONFIRMATION_PHRASE:
        switched = request_live_mode(text)
        if switched:
            await update.message.reply_text(
                "LIVE mode confirmed and activated. Hardcoded safety limits "
                "(safety/limits.py) remain in force regardless of mode -- "
                "this bot cannot raise leverage or disable stop-losses."
            )
        else:
            await update.message.reply_text("Confirmation phrase did not match. Still in DRY-RUN.")
        return

    await update.message.reply_text(
        "Unrecognized message. Use /status for a report, or send the exact "
        "live-trading confirmation phrase to switch modes."
    )


def build_application() -> Application:
    load_dotenv()
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("dry_run", dry_run))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    return app


if __name__ == "__main__":
    application = build_application()
    application.run_polling()
