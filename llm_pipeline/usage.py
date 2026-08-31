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

# USD per 1M tokens (input, output), list prices.
PRICES = {
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Empirical correction, from the only check that can settle this: comparing the
# figure below against a real invoice. Over one measured window the list-price
# arithmetic produced $12.25 where the account was actually charged $9.00 -- an
# overestimate of 36%.
#
# Deliberately a single scalar rather than re-derived per-token prices. Two
# different price sets reconcile that same observation (output at $8.81/M with
# input unchanged, or everything scaled by 0.735), and one data point cannot
# distinguish them. Inventing a specific price would be overfitting a noisy
# measurement and would read as authoritative when it is not.
#
# TOKEN COUNTS ARE EXACT regardless -- they come straight off each response.
# Only the dollar column depends on this. Re-derive the factor by comparing a
# fresh /usage total against the console's billing page over the same period.
COST_CALIBRATION = 0.735


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
        # Cache tokens are reported SEPARATELY from input_tokens, and are priced
        # differently: a write costs 1.25x base input, a read 0.1x. Recording them
        # is what makes the caching verifiable instead of assumed -- a breakpoint
        # placed on a varying block still "works", it just writes every time and
        # never reads, and the only way to see that is these two counters.
        tcw = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        tcr = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        pin, pout = _price(model)
        data = read_json(USAGE_PATH, {})
        e = data.setdefault(label, {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                     "cache_write_tokens": 0, "cache_read_tokens": 0,
                                     "model": model, "usd": 0.0})
        e["calls"] += 1
        e["input_tokens"] += tin
        e["output_tokens"] += tout
        e["cache_write_tokens"] = e.get("cache_write_tokens", 0) + tcw
        e["cache_read_tokens"] = e.get("cache_read_tokens", 0) + tcr
        e["usd_list"] = round(e["input_tokens"] / 1e6 * pin
                              + e["cache_write_tokens"] / 1e6 * pin * 1.25
                              + e["cache_read_tokens"] / 1e6 * pin * 0.10
                              + e["output_tokens"] / 1e6 * pout, 4)
        e["usd"] = round(e["usd_list"] * COST_CALIBRATION, 4)
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
    lines = [f"{'call site':<22}{'calls':>7}{'in tok':>11}{'cache wr':>10}{'cache rd':>10}"
             f"{'out tok':>10}{'USD':>9}"]
    tc = ti = to = tw = tr = 0
    tu = 0.0
    for label, v in rows:
        w, r = v.get("cache_write_tokens", 0), v.get("cache_read_tokens", 0)
        # Recomputed at display time rather than read from the stored figure, so
        # that changing COST_CALIBRATION corrects the whole history at once
        # instead of applying only to rows written afterwards.
        pin, pout = _price(v.get("model", ""))
        v = {**v, "usd": round((v["input_tokens"] / 1e6 * pin + w / 1e6 * pin * 1.25
                                + r / 1e6 * pin * 0.10 + v["output_tokens"] / 1e6 * pout)
                               * COST_CALIBRATION, 4)}
        lines.append(f"{label:<22}{v['calls']:>7}{v['input_tokens']:>11,}{w:>10,}{r:>10,}"
                     f"{v['output_tokens']:>10,}{v['usd']:>9.2f}")
        tc += v["calls"]; ti += v["input_tokens"]; to += v["output_tokens"]
        tw += w; tr += r; tu += v["usd"]
    lines.append(f"{'TOTAL':<22}{tc:>7}{ti:>11,}{tw:>10,}{tr:>10,}{to:>10,}{tu:>9.2f}")
    if tw or tr:
        # The number that says whether the breakpoint is in the right place. A
        # breakpoint on varying content writes every call and reads nothing.
        lines.append(f"\ncache reads / (reads + writes) = {tr/(tr+tw)*100:.0f}%  "
                     f"-- near 0% means the breakpoint is on content that changes")
    if tc:
        lines.append(f"\nAverage cost per call: ${tu/tc:.5f}")
    lines.append(f"\nToken counts are exact. Dollars apply a {COST_CALIBRATION:.3f} calibration to "
                 f"list prices, fitted against one real invoice -- treat them as indicative and "
                 f"check the console for what was actually billed.")
    return "\n".join(lines)
