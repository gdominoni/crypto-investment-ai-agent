"""Formats SystemStatus into a human-readable Telegram message using
Claude Haiku -- per the project's cost model (README Section 2), the
server daemon uses Haiku exclusively for this, never Sonnet/Opus. Enforced
concretely by tests/test_orchestrator_cost_model.py, not just documented.
"""

import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv

from orchestrator.status import SystemStatus

MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = """You are a status-reporting assistant for a crypto trading system. \
Given a JSON blob describing the system's current state, write a short, clear Telegram \
message (plain text, no markdown headers, a few emoji are fine) summarizing:
- Whether the system is in DRY-RUN or LIVE mode
- The circuit breaker status (safe / triggered, and why if triggered)
- Capital allocation across modules, and why any module was excluded
Keep it under 800 characters. Do not invent numbers not present in the JSON."""


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def status_to_json(status: SystemStatus) -> dict:
    return {
        "execution_mode": status.execution_mode,
        "generated_at": status.generated_at.isoformat(),
        "modules": [
            {
                "name": k.module_name,
                "win_rate": round(k.win_rate, 3),
                "sortino_ratio": round(k.sortino_ratio, 2),
                "net_profit_pct": round(k.net_profit_pct, 2),
                "sample_size": k.sample_size,
                "significant": k.meets_significance_threshold,
            }
            for k in status.module_kpis
        ],
        "allocation": {
            "weights": {name: round(w, 3) for name, w in status.allocation.weights.items()},
            "cash_reserve_pct": round(status.allocation.cash_reserve_pct, 3),
            "excluded": status.allocation.excluded_modules,
        },
        "circuit_breaker": (
            {"should_liquidate": status.circuit_breaker.should_liquidate, "reasons": status.circuit_breaker.reasons}
            if status.circuit_breaker is not None
            else {"status": "unavailable -- no local OHLC data to check"}
        ),
    }


def format_status_message(status: SystemStatus) -> str:
    load_dotenv()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(status_to_json(status))}],
    )
    return _strip_markdown_fences(response.content[0].text)


if __name__ == "__main__":
    from orchestrator.status import gather_system_status

    print(format_status_message(gather_system_status()))
