"""Persistent registry of candidates discovered through the novel-
condition / shock-detection pathway (novel_condition_tester.py), so a
condition accepted once via "test it" doesn't just fire one live trade
and vanish -- it's re-tested every week alongside the static battery
(candidates/definitions.py) by run_battery.py, exactly like a rejected
static candidate is cheaply re-checked rather than assumed permanent.
Committed to git, like execution/live_battery_state.json: a durable
record of what's been discovered and tried, not ephemeral runtime state.

This is also what keeps Sonnet from re-proposing a condition a human
already tested: `rejected_labels()` feeds directly into the context
summary Sonnet reads before deciding whether something is "genuinely
new."
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from llm_pipeline.novel_condition_tester import ConditionSpec, clause_from_dict, clause_to_dict

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "candidates" / "dynamic_candidates.json"


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    return json.loads(REGISTRY_PATH.read_text())


def _spec_to_dict(spec: ConditionSpec) -> dict:
    return {"label": spec.label,
            "clauses": [clause_to_dict(c) for c in spec.clauses],
            "direction": spec.direction, "horizons": list(spec.horizons)}


def _dict_to_spec(d: dict) -> ConditionSpec:
    return ConditionSpec(label=d["label"], clauses=tuple(clause_from_dict(c) for c in d["clauses"]),
                          direction=d["direction"], horizons=tuple(d["horizons"]))


def record_test_result(spec: ConditionSpec, status: str, source: str) -> None:
    """Called after EVERY ad-hoc test -- accepted, watch, or rejected,
    never only the successes. The label is this condition's permanent
    identity in the registry; re-testing the same label overwrites its
    prior status rather than accumulating duplicate entries, since only
    the most recent test result should govern whether it's proposed or
    traded again."""
    registry = load_registry()
    registry[spec.label] = {
        "spec": _spec_to_dict(spec), "status": status, "source": source,
        "last_tested_at": datetime.now(timezone.utc).isoformat(),
    }
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


def registered_specs() -> list[ConditionSpec]:
    """Every dynamically-discovered condition, regardless of its last
    known status -- run_battery.py re-tests all of them weekly, the same
    "cheap re-check, don't assume permanent" treatment already applied to
    rejected static candidates.

    Entries that no longer satisfy `ConditionSpec`'s own validation are
    SKIPPED -- not crashed on, and not silently coerced into passing. In
    practice these are conditions recorded before a news/macro event
    clause became a NECESSARY condition: measured on the replay's own
    registry, 80 of 92 were pure chart patterns or shock-only and never
    tested this project's actual question. They stay on disk as a record
    of what was tried (deleting history would be its own dishonesty) but
    are no longer re-tested, re-reported, or able to open live tests.
    `off_thesis_labels()` names them, so the drop in tracked count is
    visible rather than looking like they quietly vanished."""
    specs = []
    for entry in load_registry().values():
        try:
            specs.append(_dict_to_spec(entry["spec"]))
        except ValueError:
            continue
    return specs


def off_thesis_labels() -> dict[str, str]:
    """Registry entries excluded by `ConditionSpec` validation, mapped to
    the reason -- so a human can see WHY the tracked count dropped rather
    than discovering it later as an unexplained gap."""
    out = {}
    for label, entry in load_registry().items():
        try:
            _dict_to_spec(entry["spec"])
        except ValueError as e:
            out[label] = str(e)
    return out


def rejected_labels() -> set[str]:
    """For Sonnet's context (context_builder.py): conditions already
    tested and not currently accepted, so they aren't silently
    re-proposed without new evidence."""
    return {label for label, entry in load_registry().items() if entry["status"] != "accepted"}
