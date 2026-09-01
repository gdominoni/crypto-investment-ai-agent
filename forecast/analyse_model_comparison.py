"""Scores the Haiku-vs-Sonnet substitutability test and states whether
switching is justified.

Everything is read against the CEILING, never against 1.0. Sonnet is not
deterministic here (`temperature=0` is rejected by the API), so Sonnet asked
the same question twice already disagrees with itself. That self-agreement is
the most any second model could achieve, and it is the only honest baseline:
scoring Haiku against perfect agreement would condemn it for variance the
reference model has too.

The verdict rule is fixed before the numbers are seen, because a threshold
chosen afterwards is not a threshold:

  * Haiku must hold the output contract. A judgment that does not parse costs
    a call and returns nothing, and no agreement rate redeems that.
  * Haiku's agreement with Sonnet must not be MEASURABLY BELOW Sonnet's
    agreement with itself. Tested paired, on the events where both produced a
    condition, with Wilcoxon signed-rank -- agreement scores are bounded in
    [0,1] and nowhere near normal, so a t-test would be the wrong instrument.

What a pass does NOT establish: at these sample sizes, failing to detect a
difference is not evidence of equivalence. It is grounds for a cost decision,
not for a claim that the two models are the same.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS_PATH = Path(__file__).resolve().parent / "model_comparison.json"

SONNET_PER_CALL = 0.0139   # measured, at the corrected $2/$10 pricing
HAIKU_PER_CALL = SONNET_PER_CALL / 3
FULL_REPLAY_CALLS = 1200



def _quality_report(rows) -> None:
    """The second question: which model is the better analyst.

    Agreement says whether one model can stand in for another. It says nothing
    about whether either is any good -- and this project's whole subject is
    whether an LLB can find a real pattern, not whether two of them can be
    swapped. Reported on the p-value distribution because acceptance is far too
    rare to separate anything at this sample size."""
    import numpy as np

    print("\n" + "=" * 62)
    print("Which model is the better ANALYST -- independent of who agrees with whom\n")
    print(f"{'':<22}{'specs':>7}{'testable':>10}{'accepted':>10}{'median p':>11}{'p<0.10':>8}")
    print("-" * 68)
    for tag, label in (("sonnet_a", "Sonnet"), ("sonnet_b", "Sonnet (2nd run)"), ("haiku", "Haiku")):
        n = sum(r.get(f"{tag}_quality", {}).get("n_specs", 0) for r in rows)
        te = sum(r.get(f"{tag}_quality", {}).get("testable", 0) for r in rows)
        ac = sum(r.get(f"{tag}_quality", {}).get("accepted", 0) for r in rows)
        ps = [p for r in rows for p in r.get(f"{tag}_quality", {}).get("p_values", [])]
        med = f"{np.median(ps):.3f}" if ps else "--"
        lo = sum(1 for p in ps if p < 0.10)
        print(f"{label:<22}{n:>7}{te:>10}{ac:>10}{med:>11}{lo:>8}")
    sp = [p for r in rows for p in r.get("sonnet_a_quality", {}).get("p_values", [])]
    hp = [p for r in rows for p in r.get("haiku_quality", {}).get("p_values", [])]
    if len(sp) >= 5 and len(hp) >= 5:
        from scipy.stats import mannwhitneyu
        pv = mannwhitneyu(sp, hp, alternative="less").pvalue
        print(f"\nAre Sonnet's p-values lower than Haiku's? Mann-Whitney p = {pv:.3f}")
        print("(one-sided: lower p-values mean the analyst is finding more that holds up)")
    else:
        print(f"\nToo few tested proposals to compare p-value distributions "
              f"(Sonnet {len(sp)}, Haiku {len(hp)}).")


def main() -> None:
    if not RESULTS_PATH.exists():
        print("No results -- run `python3 -m forecast.model_comparison` first.")
        return
    rows = json.loads(RESULTS_PATH.read_text())
    n = len(rows)
    print(f"Does Haiku propose what Sonnet proposes?  --  {n} events, 3 judgments each\n")

    # 1. The contract. Measured first because it is disqualifying on its own.
    errs = {t: sum(1 for r in rows if f"{t}_error" in r) for t in ("sonnet_a", "sonnet_b", "haiku")}
    print(f"Unparseable / failed responses:  Sonnet {errs['sonnet_a'] + errs['sonnet_b']}/{2*n}   "
          f"Haiku {errs['haiku']}/{n}")

    # 2. Do they even agree on whether there is anything to propose?
    for label, key in (("Sonnet vs Sonnet (ceiling)", "ceiling"),
                        ("Haiku  vs Sonnet", "haiku_vs_sonnet")):
        same = sum(1 for r in rows if r[key]["same_action"])
        both = sum(1 for r in rows if r[key]["both_proposed"])
        silent = sum(1 for r in rows if r[key]["both_silent"])
        print(f"{label:<28} agreed to act or not act: {same}/{n}   "
              f"(both proposed {both}, both silent {silent})")

    # 3. The measurement that matters: when both proposed, is it the same
    #    hypothesis? Paired -- only events where BOTH comparisons are defined,
    #    otherwise the two distributions are over different events.
    paired = [(r["ceiling"]["overlap"], r["haiku_vs_sonnet"]["overlap"]) for r in rows
              if r["ceiling"]["overlap"] is not None
              and r["haiku_vs_sonnet"]["overlap"] is not None]
    paired = [(c, h) for c, h in paired if not (np.isnan(c) or np.isnan(h))]

    print(f"\nBehavioural agreement (Jaccard overlap of the days each condition fires on)")
    print(f"{len(paired)} events where both comparisons are defined\n")
    if not paired:
        print("No comparable events -- too few cases where every model proposed something.")
        print("Rerun with a larger --n before drawing any conclusion.")
        return

    ceil = np.array([c for c, _ in paired])
    haik = np.array([h for _, h in paired])
    print(f"{'':<30}{'mean':>8}{'median':>9}{'>=0.5':>8}{'=0.0':>7}")
    print("-" * 62)
    for label, arr in (("Sonnet vs Sonnet (ceiling)", ceil), ("Haiku  vs Sonnet", haik)):
        print(f"{label:<30}{arr.mean():>8.3f}{np.median(arr):>9.3f}"
              f"{(arr >= 0.5).sum():>8}{(arr == 0).sum():>7}")

    from scipy.stats import wilcoxon

    diff = ceil - haik
    if np.all(diff == 0):
        pval, stat_note = 1.0, "identical on every event"
    elif len(paired) < 6:
        pval, stat_note = float("nan"), f"only {len(paired)} pairs -- too few to test"
    else:
        pval = wilcoxon(ceil, haik, alternative="greater").pvalue
        stat_note = f"Wilcoxon signed-rank, one-sided (ceiling > Haiku): p = {pval:.4f}"
    print(f"\n{stat_note}")
    print(f"Mean shortfall vs the ceiling: {diff.mean():+.3f}")

    _quality_report(rows)

    saving = FULL_REPLAY_CALLS * (SONNET_PER_CALL - HAIKU_PER_CALL)
    print("\n" + "=" * 62)
    print(f"Cost: ${SONNET_PER_CALL:.4f} vs ${HAIKU_PER_CALL:.4f} per call. Over a full "
          f"~{FULL_REPLAY_CALLS}-call replay,\nswitching saves about ${saving:.2f} "
          f"(${FULL_REPLAY_CALLS*SONNET_PER_CALL:.2f} -> "
          f"${FULL_REPLAY_CALLS*HAIKU_PER_CALL:.2f}).\n")

    if errs["haiku"] > errs["sonnet_a"]:
        print(f"VERDICT: NO. Haiku failed to hold the output contract on {errs['haiku']}/{n}")
        print("events. A call that returns nothing is not a cheaper call.")
    elif not np.isnan(pval) and pval < 0.05:
        print("VERDICT: NO. Haiku's proposals overlap Sonnet's measurably less than")
        print("Sonnet's own re-runs do. It is not proposing the same conditions -- the")
        print("gap is larger than Sonnet's own run-to-run variance.")
    elif np.isnan(pval):
        print("VERDICT: UNDECIDED -- too few comparable events. Rerun with a larger --n.")
    else:
        print("VERDICT: YES, on this evidence. Haiku's agreement with Sonnet is not")
        print(f"measurably below Sonnet's agreement with itself, and switching saves")
        print(f"about ${saving:.2f} per full replay.")
        print("\nRead the ceiling before acting on this. If Sonnet self-agrees only")
        print("weakly, both models are largely sampling from a wide space of defensible")
        print("proposals -- which makes Haiku an adequate substitute AND means the")
        print("proposal step is less determinate than a single run suggests.")


if __name__ == "__main__":
    main()
