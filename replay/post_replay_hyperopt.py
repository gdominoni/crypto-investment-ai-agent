"""The Freqtrade hyperopt cross-check, run ONCE after a replay finishes.

WHY IT IS NOT IN THE PER-OCCURRENCE MESSAGES, which is a deliberate choice and
not an oversight. `hyperopt_runner.run_all` optimises over `timerange="20180101-"`
-- the whole history, to the present. A replay message dated 2020 that quoted a
TP/SL optimised on data through 2026 would be showing information from the
future.

That misalignment is an artefact of compressing nine years into a short run, not
a property of the system: live, "the whole history to the present" IS the
present, and the two coincide. The cross-check never gates anything and its
output goes to a human rather than to the model, so nothing downstream is
contaminated either way. It is moved here rather than removed because the
independent second opinion is worth having -- just not stamped with a date it
could not have been computed on.

The other reason is cost. A single candidate at 50 epochs takes several real
minutes; a replay that discovers ~190 candidates would spend 10-16 hours on this
if it ran inline, for a figure that never needs to be fresh.

Run after the replay reaches the present:

    python3 -m replay.post_replay_hyperopt            # every surviving candidate
    python3 -m replay.post_replay_hyperopt --accepted # only those still accepted
    python3 -m replay.post_replay_hyperopt --dry-run  # list what it would run

It writes into the same `execution/hyperopt_results.json` everything else reads,
so the summary below and any later message pick it up automatically.
"""
from __future__ import annotations

import argparse
import sys

from execution import hyperopt_runner
from replay import state
from replay import status_history as sh


def candidates_to_check(accepted_only: bool = False) -> list[str]:
    """Every candidate the replay ended up tracking, dropped ones excluded.

    Dropped candidates are skipped because the keep-or-drop decision has already
    been made about them; re-optimising something the system stopped tracking
    spends minutes to produce a number nobody will read."""
    registry = state.load_dynamic_candidates()
    statuses = sh.all_latest_statuses()
    out = []
    for label in registry:
        if sh.is_dropped(label):
            continue
        if accepted_only and statuses.get(label, {}).get("status") != "accepted":
            continue
        out.append(label)
    return sorted(out)


def format_summary(labels: list[str]) -> str:
    """A plain-text block for the replay's final write-up.

    Reports the cross-check beside nothing else deliberately: it is a SECOND
    opinion from a different optimiser on a different engine, and printing it
    next to this project's own walk-forward numbers invites the two to be read
    as agreeing or disagreeing when they answer slightly different questions.
    What it is good for is the opposite check -- a candidate this project likes
    that an independent Bayesian search over a continuous space cannot make
    money on is worth a second look."""
    results = hyperopt_runner.load_results()
    lines = [f"Freqtrade hyperopt cross-check -- {len(labels)} candidate(s), run after the replay",
             "Independent optimiser, informational only: it never gated any verdict above.", ""]
    ran = [l for l in labels if isinstance(results.get(l), dict)]
    if not ran:
        lines.append("No results recorded yet -- run `python3 -m replay.post_replay_hyperopt` first.")
        return "\n".join(lines)
    for label in ran:
        lines.append(f"  {label}")
        lines.append(f"    {hyperopt_runner.format_result(label)}")
    missing = [l for l in labels if l not in ran]
    if missing:
        lines.append("")
        lines.append(f"  {len(missing)} candidate(s) have no result: "
                     f"{', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted", action="store_true",
                    help="only candidates still accepted at the end of the replay")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true", help="list candidates, run nothing")
    args = ap.parse_args()

    labels = candidates_to_check(accepted_only=args.accepted)
    if not labels:
        print("No tracked candidates in the replay state -- nothing to cross-check.")
        return 0
    # Several real minutes each, so the count is stated before anything starts
    # rather than discovered halfway through.
    print(f"{len(labels)} candidate(s) to cross-check at {args.epochs} epochs each.")
    print(f"Expect roughly {len(labels) * 3}-{len(labels) * 5} minutes. Local only; nothing is sent anywhere.")
    if args.dry_run:
        for label in labels:
            print(f"  {label}")
        return 0

    hyperopt_runner.run_all(labels, epochs=args.epochs)
    print()
    print(format_summary(labels))
    return 0


if __name__ == "__main__":
    sys.exit(main())
