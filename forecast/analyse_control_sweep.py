"""Score `forecast/control_sweep.py` -- per sentiment arm, so the ceiling
(planted), the floor (random) and the real-news arm can be read against
each other and against the real grammar sweep.

The comparison that matters:

    planted arms  = what a PERFECT news feed looks like through this
                    architecture. The upper bound on discoverability.
    random arm    = the false-positive floor.
    real_news arm = actual crypto news, same machinery.
    grammar sweep = the real macro calendar (forecast/analyse_sweep.py).

If the planted arms are barely discoverable, then better news data cannot
rescue the project and the constraint is the METHOD. If the planted arms
are comfortably discovered while real news is not, the method is sound
and the constraint is the DATA -- which is the case where a GDELT
backfill is worth building.

Each arm is FDR-corrected within its own family, which is the honest unit
here: each arm is a separate experiment with its own hypothesis set.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_PATH = Path(__file__).resolve().parent / "control_sweep.json"

ARM_ORDER = ["planted20", "planted35", "planted50", "real_news", "random"]
ARM_DESC = {
    "planted20": "PLANTED, very strong (a perfect news feed)",
    "planted35": "PLANTED, strong",
    "planted50": "PLANTED, marginal (barely better than a coin flip)",
    "real_news": "REAL crypto news dates",
    "random": "RANDOM noise (the false-positive floor)",
}


def main() -> None:
    from candidates.methodology import benjamini_hochberg

    if not RESULTS_PATH.exists():
        print("No results -- run `python3 -m forecast.control_sweep` first.")
        return
    rows = json.loads(RESULTS_PATH.read_text())
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r.get("arm")].append(r)

    print(f"Control sweep: {len(rows)} conditions across {len(by_arm)} sentiment arms\n")
    print(f"{'arm':<50}{'tested':>7}{'family':>7}{'raw<.05':>9}{'BH':>5}{'accepted':>10}{'med excess':>12}")
    print("-" * 100)

    summary = {}
    for arm in ARM_ORDER:
        rs = by_arm.get(arm, [])
        if not rs:
            continue
        scored = [r for r in rs if r.get("p") is not None]
        # Same rule as analyse_sweep: the family is the ACTIONABLE set. A
        # p-value on a handful of events is not a test and must not consume
        # the family's alpha (see candidates/methodology.py::apply_fdr_demotion).
        family = [r for r in scored if r.get("status") != "insufficient_data"]
        raw = [r for r in family if r["p"] < 0.05]
        bh = []
        if family:
            keep = benjamini_hochberg([r["p"] for r in family], alpha=0.05)
            bh = [r for r, k in zip(family, keep) if k]
        acc = [r for r in family if r.get("status") == "accepted"]
        ex = [r["excess"] for r in family if r.get("excess") is not None]
        med = np.median(ex) * 100 if ex else float("nan")
        summary[arm] = {"family": len(family), "raw": len(raw), "bh": len(bh), "acc": len(acc)}
        print(f"{ARM_DESC[arm]:<50}{len(rs):>7}{len(family):>7}{len(raw):>9}{len(bh):>5}"
              f"{len(acc):>10}{med:>11.2f}%")

    print("\n" + "=" * 100)
    p20 = summary.get("planted20", {})
    rnd = summary.get("random", {})
    real = summary.get("real_news", {})

    print("VERDICT\n")
    if p20.get("acc", 0) == 0:
        print("  Even a PERFECT sentiment signal is not discoverable through this grammar.")
        print("  The constraint is the METHOD (statistical power at these sample sizes),")
        print("  not the quality of the news data. A GDELT backfill would not change the")
        print("  outcome -- the same conjunction test would swallow a perfect signal too.")
    elif rnd.get("acc", 0) >= p20.get("acc", 0):
        print("  The random arm scores as well as the planted arm -- the discovery layer is")
        print("  not discriminating between signal and noise. Treat every acceptance as")
        print("  suspect until this is explained.")
    else:
        print(f"  A perfect sentiment signal IS discoverable: {p20.get('acc')} accepted vs "
              f"{rnd.get('acc')} for noise.")
        print("  The architecture can do the job it was built for. Since real macro/news data")
        print("  produced nothing, the constraint is the DATA, not the method -- which is")
        print("  exactly the case where better news input (GDELT) is worth building.")
        print(f"\n  Real crypto news dates landed at {real.get('acc', 0)} accepted, "
              f"{real.get('bh', 0)} surviving FDR.")

    print("\n  Compare with the real macro grammar sweep (python3 -m forecast.analyse_sweep):")
    print("  296 actionable conditions, 2 raw p<0.05, 0 surviving FDR, 0 accepted.")


if __name__ == "__main__":
    main()
