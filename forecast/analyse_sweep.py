"""Score `forecast/grammar_sweep.py`'s output against the decision rule
that was written down BEFORE the sweep ran.

The rule, pre-registered 2026-08-29 (see the printed output below):

    Under the null "there are no real news+market-state patterns", a
    family of 672 conditions should produce ~34 raw p<0.05 (95% interval
    23-45) and ~0 surviving Benjamini-Hochberg FDR, because that is
    exactly what FDR controls.

        0 BH survivors  -> the replay will find nothing; don't spend on it
        1-2 survivors   -> borderline; run the replay to see if it finds
                           them PROSPECTIVELY, which is the stronger claim
        >=3 survivors   -> real signal in the grammar; run the replay

Recording the rule in code, rather than deciding what counts as success
after seeing the numbers, is the whole point -- a family of 672 tests
will always contain something that looks good if you go looking for it
afterwards.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS_PATH = Path(__file__).resolve().parent / "grammar_sweep.json"
FAMILY_TARGET = 672


def main() -> None:
    from candidates.methodology import benjamini_hochberg

    if not RESULTS_PATH.exists():
        print("No sweep results yet -- run `python3 -m forecast.grammar_sweep` first.")
        return
    rows = json.loads(RESULTS_PATH.read_text())
    done = len(rows)
    partial = done < FAMILY_TARGET

    errors = [r for r in rows if r.get("status") == "error"]
    scored = [r for r in rows if r.get("p") is not None]
    print(f"Sweep results: {done}/{FAMILY_TARGET} conditions"
          f"{'  *** PARTIAL -- interpret with care ***' if partial else ''}")
    print(f"  scored (p-value computed): {len(scored)}")
    print(f"  unscorable (insufficient data / no valid folds): {done - len(scored) - len(errors)}")
    print(f"  errors: {len(errors)}\n")
    if not scored:
        print("Nothing scored yet.")
        return

    # THE FAMILY IS THE ACTIONABLE SET. pattern_significance returns a
    # p-value even for a condition with ONE out-of-sample event; that is not
    # a test, and classify_status already refuses to call it anything but
    # insufficient_data. Scoring those as "discoveries" is how the first run
    # of this analysis reported two BH survivors with n=1 and n=5. Excluding
    # them makes the verdict MORE negative, not less -- it discards a
    # favourable-looking artifact rather than rescuing one.
    family = [r for r in scored if r.get("status") != "insufficient_data"]
    excluded = len(scored) - len(family)
    print(f"Excluded from the family (n too small to ever be classified): {excluded}")
    print(f"Actionable family: {len(family)}\n")

    degenerate_hits = [r for r in scored if r["p"] < 0.05 and r.get("status") == "insufficient_data"]
    if degenerate_hits:
        print(f"  ({len(degenerate_hits)} of the raw p<0.05 hits are degenerate-n artifacts, "
              f"n = {sorted(r['n'] for r in degenerate_hits)} -- excluded)\n")

    ps = np.array([r["p"] for r in family])
    raw_hits = [r for r in family if r["p"] < 0.05]
    expected = 0.05 * len(family)
    print(f"Raw p<0.05 within the family: {len(raw_hits)}   (expected under the null: {expected:.0f})")

    rejected = benjamini_hochberg(ps, alpha=0.05)
    survivors = [r for r, keep in zip(family, rejected) if keep]
    print(f"Surviving Benjamini-Hochberg FDR at 0.05: {len(survivors)}\n")

    accepted = [r for r in family if r.get("status") == "accepted"]
    print(f"Reaching status 'accepted' (every gate: significance, direction, "
          f"concentration, risk path): {len(accepted)}\n")

    if survivors:
        print("BH survivors, strongest first:")
        for r in sorted(survivors, key=lambda r: r["p"]):
            print(f"  p={r['p']:.4f}  excess={r['excess']*100:+6.2f}%  "
                  f"MFE/MAE={r['mfe_mae'] if r['mfe_mae'] is not None else float('nan'):.2f}  "
                  f"n={r['n']}  status={r['status']}\n    {r['label']}")
        print()

    # Positive-excess share is a second, independent read: with no real effect
    # anywhere it should sit at ~50%, a coin flip, regardless of p-values.
    ex = [r["excess"] for r in family if r.get("excess") is not None]
    if ex:
        pos = sum(1 for e in ex if e > 0)
        print(f"Positive excess return: {pos}/{len(ex)} ({pos/len(ex)*100:.0f}%) "
              f"-- ~50% is what no-real-effect looks like\n")

    print("PRE-REGISTERED VERDICT")
    n = len(survivors)
    if n == 0:
        print("  0 BH survivors -> the grammar contains no detectable pattern on this")
        print("  data. Since the replay tests with LESS data than this sweep does, it")
        print("  cannot find one either. Predicted replay outcome: 0 accepted,")
        print("  0 validated. Recommendation: do not spend on the replay; put the")
        print("  budget toward the GDELT news backfill, which adds a genuinely new")
        print("  information source rather than re-testing the same one.")
    elif n <= 2:
        print(f"  {n} BH survivor(s) -> borderline. Worth running the replay: a condition")
        print("  found PROSPECTIVELY, by a model that never saw the backtest, is a")
        print("  much stronger claim than one found by exhaustive search here.")
    else:
        print(f"  {n} BH survivors -> real signal in the grammar. Run the replay.")
    if partial:
        print("\n  NOTE: partial sweep. BH depends on the whole family, so this verdict")
        print("  is provisional until all 672 are in.")


if __name__ == "__main__":
    main()
