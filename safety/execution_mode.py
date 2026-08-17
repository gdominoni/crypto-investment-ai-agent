"""Dry-run/live execution mode guard.

The system always boots in dry-run: get_mode() returns "dry_run" whenever
no state file exists yet, so a fresh deploy is never accidentally live.
Moving to live requires request_live_mode() with the exact hardcoded
confirmation phrase -- a literal string match, not an LLM's interpretation
of intent. The orchestrator (Phase 8) relays a human's exact Telegram
message text into this function; Haiku is never given authority to decide
*whether* to go live, only to pass the human's words through unmodified.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

STATE_FILE = Path(__file__).resolve().parent / "state" / "execution_mode.json"
LIVE_CONFIRMATION_PHRASE = "CONFIRM LIVE TRADING"


@dataclass
class ExecutionModeState:
    mode: str = "dry_run"  # "dry_run" | "live"


def get_mode() -> str:
    if not STATE_FILE.exists():
        return "dry_run"
    return json.loads(STATE_FILE.read_text())["mode"]


def request_live_mode(confirmation_text: str) -> bool:
    """Switches to live mode only on an exact phrase match. Returns whether
    the switch happened, so the caller can report success/failure."""
    if confirmation_text.strip() != LIVE_CONFIRMATION_PHRASE:
        return False
    _write_mode("live")
    return True


def revert_to_dry_run() -> None:
    """Always allowed, no confirmation needed -- moving to the safer state
    never requires a gate."""
    _write_mode("dry_run")


def _write_mode(mode: str) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(asdict(ExecutionModeState(mode=mode))))


if __name__ == "__main__":
    print(f"current mode: {get_mode()}")
