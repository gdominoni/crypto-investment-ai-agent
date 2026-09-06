"""Migrate the completed replay's state into production's own state files.

WHY THIS IS A SCRIPT AND NOT `cp`. The two sides were built to mirror each
other and mostly do, but three of the six files are NOT copy-compatible, and
each would fail in a different way -- one of them silently:

  * `dynamic_candidates.json`. The replay stores the spec FLAT
    (`{label, clauses, direction, horizons}`); production wraps it
    (`{"spec": {...}, "status", "source", "last_tested_at"}`) and reads
    `entry["spec"]` / `entry["status"]` directly. `registered_specs()` catches
    ValueError, not KeyError, so a straight copy raises an UNCAUGHT KeyError
    inside the hourly scan.

  * `status_history.json`. The replay writes date-only stamps
    ("2018-07-06"); production writes tz-aware ISO. `years_tracked()` does
    `datetime.now(timezone.utc) - fromisoformat(first_tracked_at)`, and a
    date-only string parses NAIVE -- "can't subtract offset-naive and
    offset-aware datetimes", raised inside the weekly re-validation.

  * `battery_status.json`. Deliberately NOT migrated. Production's
    `live_battery_state.json` has a different top-level shape
    (`generated_at`/`horizons_days`/`candidates` vs `as_of`/`candidates`/
    `summary`) and is a DERIVED cache: `run_battery.run_all()` rebuilds it
    from the registry on every run. Translating stale numbers across two
    schemas would be strictly worse than recomputing them on real data, which
    is what the post-migration step does.

The other three (`trade_log` -> `live_tests`, `horizons`,
`confirmation_priors`, `parked_proposals`) are copy-compatible, verified
field by field rather than assumed.

Run:
    python3 -m scripts.migrate_replay_state_to_production            # dry run
    python3 -m scripts.migrate_replay_state_to_production --apply    # writes

Writing is opt-in because this overwrites real production state. Whatever is
already there is backed up first, into a timestamped folder.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPLAY = ROOT / "replay" / "state"

# source -> destination, for the files that migrate unchanged.
COPY_AS_IS = {
    REPLAY / "trade_log.json": ROOT / "execution" / "live_tests.json",
    REPLAY / "horizons.json": ROOT / "execution" / "horizons.json",
    REPLAY / "confirmation_priors.json": ROOT / "execution" / "confirmation_priors.json",
    REPLAY / "parked_proposals.json": ROOT / "execution" / "parked_proposals.json",
}
REGISTRY_DST = ROOT / "candidates" / "dynamic_candidates.json"
HISTORY_DST = ROOT / "candidates" / "status_history.json"


def _read(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _iso(stamp):
    """Replay date-only -> tz-aware ISO. Anything already tz-aware is left
    alone, and None stays None (`last_asked_at` is legitimately null)."""
    if not stamp:
        return stamp
    parsed = datetime.fromisoformat(str(stamp))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat()


def build_registry(replay_registry: dict, replay_history: dict) -> dict:
    """Wrap each flat replay spec in production's envelope.

    `status` is taken from the candidate's own last recorded status rather
    than from the battery cache: status_history covers every tracked
    candidate, the cache only those the final refresh happened to score.
    Production's `rejected_labels()` indexes `entry["status"]` with no
    default, so every entry must carry one."""
    out = {}
    for label, spec in replay_registry.items():
        log = (replay_history.get(label) or {}).get("status_log") or []
        status = log[-1]["status"] if log else "insufficient_data"
        out[label] = {
            "spec": spec,
            "status": status,
            "source": "replay_migration",
            "last_tested_at": _iso((log[-1]["at"] if log else None) or "2026-09-05"),
        }
    return out


def build_history(replay_history: dict) -> tuple[dict, list[str]]:
    """Same structure, every date stamp normalised to tz-aware ISO.

    Entries with no usable `first_tracked_at` are LEFT OUT, and returned
    separately so the count is reported rather than silently absorbed.
    Production's `_years_tracked` indexes that field with no default and
    `fromisoformat(None)` raises -- inside the weekly prune check, which is
    exactly where a crash is least visible. The replay produced one such
    entry (a horizon recorded before any status ever was, so an empty
    `status_log` and nothing to carry); production cannot produce them,
    because both of its own writers stamp `first_tracked_at` on creation.
    Dropping it loses nothing: the candidate stays in the registry, and the
    first battery run that scores it recreates the entry properly."""
    out, skipped = {}, []
    for label, entry in replay_history.items():
        if not isinstance(entry.get("first_tracked_at"), str):
            skipped.append(label)
            continue
        migrated = dict(entry)
        migrated["first_tracked_at"] = _iso(entry.get("first_tracked_at"))
        migrated["last_asked_at"] = _iso(entry.get("last_asked_at"))
        migrated["status_log"] = [{**e, "at": _iso(e.get("at"))} for e in entry.get("status_log", [])]
        out[label] = migrated
    return out, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually write (default is a dry run)")
    args = ap.parse_args()

    replay_registry = _read(REPLAY / "dynamic_candidates.json")
    replay_history = _read(REPLAY / "status_history.json")
    if replay_registry is None or replay_history is None:
        print("No completed replay state found -- nothing to migrate.")
        return 1

    registry = build_registry(replay_registry, replay_history)
    history, skipped = build_history(replay_history)

    print(f"{'APPLYING' if args.apply else 'DRY RUN'} -- replay state -> production\n")
    print(f"  candidates/dynamic_candidates.json   {len(registry):>6} entries (spec wrapped, status attached)")
    print(f"  candidates/status_history.json       {len(history):>6} entries (dates -> tz-aware ISO)")
    if skipped:
        print(f"       ... {len(skipped)} skipped, no usable first_tracked_at: {', '.join(skipped)}")
    for src, dst in COPY_AS_IS.items():
        data = _read(src)
        n = len(data) if isinstance(data, (list, dict)) else 0
        print(f"  {dst.relative_to(ROOT).as_posix():<36} {n:>6} entries (copied unchanged)")
    print("\n  execution/live_battery_state.json    NOT migrated -- rebuilt by run_battery.run_all()")

    dropped = [k for k, v in history.items() if v.get("dropped")]
    confirmed = [k for k, v in history.items() if v.get("milestone_cleared")]
    print(f"\n  carried forward: {len(dropped)} dropped candidate(s), {len(confirmed)} that reached CONFIRMED")

    if not args.apply:
        print("\nNothing written. Re-run with --apply to migrate.")
        return 0

    backup = ROOT / "migration_backup" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup.mkdir(parents=True, exist_ok=True)
    for dst in [REGISTRY_DST, HISTORY_DST, *COPY_AS_IS.values(), ROOT / "execution" / "live_battery_state.json"]:
        if dst.exists():
            shutil.copy2(dst, backup / dst.name)
    print(f"\nExisting production state backed up to {backup.relative_to(ROOT)}/")

    REGISTRY_DST.write_text(json.dumps(registry, indent=2))
    HISTORY_DST.write_text(json.dumps(history, indent=2))
    for src, dst in COPY_AS_IS.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print("Written.\n")
    print("Next: `python3 -m candidates.run_battery` to rebuild live_battery_state.json "
          "on real current data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
