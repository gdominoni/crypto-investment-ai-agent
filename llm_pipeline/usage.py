"""Token-usage and cost accounting for every Anthropic call this project
makes.

Why this exists. Asked "what will a full replay cost", this project could
only offer an estimate built on assumptions, and one of those assumptions
was wrong by roughly 4x: `max_tokens=2000` was read as the expected output
length when it is only a CAP -- most assessments return a short
`no_action` JSON. Nothing anywhere recorded a single real token count,
so the error was invisible and the disagreement unresolvable.

Every Anthropic response already carries `usage.input_tokens` and
`usage.output_tokens`. Recording them costs nothing and turns the next
run into its own answer, per-call-site, instead of a forecast.

Prices are USD per million tokens and are a LOCAL ASSUMPTION -- update
`PRICES` if Anthropic's published pricing changes. Token counts are real
and remain correct regardless; only the dollar column depends on this
table, which is why both are stored.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from candidates.atomic_json import read_json, write_json

USAGE_PATH = Path(__file__).resolve().parent.parent / "llm_usage.json"

# USD per 1M tokens (input, output).
PRICES = {
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def _price(model: str) -> tuple[float, float]:
    for known, p in PRICES.items():
        if model.startswith(known):
            return p
    return (0.0, 0.0)  # unknown model: count tokens, don't invent a price


def record(response, label: str, model: str) -> None:
    """Accumulate one call's real token usage under `label` (the call
    site, e.g. "replay.shock", "replay.macro").

    Never raises: a metering failure must not take down a replay that is
    otherwise working. `getattr` chains rather than attribute access
    because a stubbed client in tests has no `usage` at all.
    """
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        tin = int(getattr(usage, "input_tokens", 0) or 0)
        tout = int(getattr(usage, "output_tokens", 0) or 0)
        pin, pout = _price(model)
        data = read_json(USAGE_PATH, {})
        e = data.setdefault(label, {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                     "model": model, "usd": 0.0})
        e["calls"] += 1
        e["input_tokens"] += tin
        e["output_tokens"] += tout
        e["usd"] = round(e["input_tokens"] / 1e6 * pin + e["output_tokens"] / 1e6 * pout, 4)
        e["model"] = model
        data["_updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(USAGE_PATH, data)
    except Exception:
        pass


def summary() -> str:
    """Human-readable table of everything spent so far -- what `/usage`
    in Telegram reports, and what a cost forecast should be built on
    instead of an assumption."""
    data = read_json(USAGE_PATH, {})
    rows = [(k, v) for k, v in data.items() if not k.startswith("_")]
    if not rows:
        return "No Anthropic calls recorded yet."
    rows.sort(key=lambda kv: -kv[1]["usd"])
    lines = [f"{'call site':<22}{'calls':>7}{'in tok':>11}{'out tok':>10}{'avg out':>9}{'USD':>9}"]
    tc = ti = to = 0
    tu = 0.0
    for label, v in rows:
        avg = v["output_tokens"] / v["calls"] if v["calls"] else 0
        lines.append(f"{label:<22}{v['calls']:>7}{v['input_tokens']:>11,}{v['output_tokens']:>10,}"
                     f"{avg:>9.0f}{v['usd']:>9.2f}")
        tc += v["calls"]; ti += v["input_tokens"]; to += v["output_tokens"]; tu += v["usd"]
    lines.append(f"{'TOTAL':<22}{tc:>7}{ti:>11,}{to:>10,}{to/max(tc,1):>9.0f}{tu:>9.2f}")
    if tc:
        lines.append(f"\nAverage cost per call: ${tu/tc:.5f}")
    lines.append(f"\nPrices are a local assumption ({USAGE_PATH.name} stores real token counts "
                 f"either way) -- see llm_pipeline/usage.py::PRICES.")
    return "\n".join(lines)
