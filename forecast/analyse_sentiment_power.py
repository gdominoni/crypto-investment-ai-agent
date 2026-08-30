"""What quality of news-sentiment feed would this system actually need?

Scores `forecast/sentiment_power.py` by rho -- the correlation between the
sentiment score and the realised forward return -- so the output is a
single, checkable number: the minimum feed quality at which anything is
discoverable.

Why that number is the one worth having. Building a real sentiment feed
(GDELT ingestion, entity extraction to attribute a story to a coin,
scoring, backfill) is the largest remaining item in this project. It is
worth doing only if a feed of ACHIEVABLE quality would be detected. The
published literature on news sentiment and next-period returns generally
reports single-digit correlations, so a realistic feed plausibly lands
near rho = 0.05 -- which is why the sweep brackets 0.04 and 0.08 rather
than only testing an oracle.

rho = 0.00 is the noise floor. It is NOT expected to be empty: at
SIGNIFICANCE_ALPHA = 0.10 a null arm should produce acceptances, and
demanding zero would condemn a correctly-behaving test. Every arm is
therefore compared AGAINST that floor with Fisher's exact test, rather
than against zero or against a margin chosen by hand.

Reading the verdict:
  separable at rho <= 0.08  -> a realistic feed is worth building
  separable only at rho >= 0.15 -> only an exceptional feed would pay,
      and GDELT-style general news is unlikely to reach it
  separable nowhere -> the conjunction test cannot use sentiment at these
      sample sizes, and the constraint is the method, not the data
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_PATH = Path(__file__).resolve().parent / "sentiment_power.json"


def main() -> None:
    from candidates.methodology import benjamini_hochberg

    if not RESULTS_PATH.exists():
        print("No results -- run `python3 -m forecast.sentiment_power` first.")
        return
    rows = json.loads(RESULTS_PATH.read_text())
    gate = 20

    by_rho = defaultdict(list)
    for r in rows:
        by_rho[r.get("rho")].append(r)

    print(f"Sentiment-feed quality vs discoverability  ({len(rows)} conditions)\n")
    print(f"{'rho':>6}{'meaning':<34}{'family':>8}{'raw<a':>7}{'BH':>5}{'accepted':>10}")
    print("-" * 70)
    MEANING = {0.30: "oracle -- not achievable",
               0.15: "exceptional feed",
               0.08: "very good feed",
               0.04: "realistic news sentiment",
               0.00: "no signal (false-positive floor)"}
    summary = {}
    for rho in sorted(by_rho, key=lambda x: -(x if x is not None else -1)):
        rs = by_rho[rho]
        fam = [r for r in rs if r.get("p") is not None and (r.get("n") or 0) > gate]
        raw = [r for r in fam if r["p"] < 0.10]
        bh = []
        if fam:
            keep = benjamini_hochberg([r["p"] for r in fam], alpha=0.05)
            bh = [r for r, k in zip(fam, keep) if k]
        acc = [r for r in fam if r.get("status") == "accepted"]
        summary[rho] = len(acc)
        print(f"{rho:>6.2f}{MEANING.get(rho,''):<34}{len(fam):>8}{len(raw):>7}{len(bh):>5}{len(acc):>10}")

    # Which trigger threshold works best -- the event-rate/sample-size trade-off.
    print(f"\nBy trigger threshold (sigma of the score), accepted counts:")
    print(f"{'rho':>6}" + "".join(f"{f'>={t}s':>10}" for t in (1.0, 1.5, 2.0)))
    for rho in sorted(by_rho, key=lambda x: -(x if x is not None else -1)):
        cells = []
        for t in (1.0, 1.5, 2.0):
            n = sum(1 for r in by_rho[rho]
                    if r.get("threshold") == t and r.get("status") == "accepted")
            cells.append(f"{n:>10}")
        print(f"{rho:>6.2f}" + "".join(cells))

    print("\n" + "=" * 70)
    # The floor is a BASELINE, not a tripwire at zero. At SIGNIFICANCE_ALPHA=0.10
    # a null arm is SUPPOSED to produce some acceptances -- that is what the alpha
    # means, and an earlier version of this check demanded exactly zero and
    # declared a correctly-behaving test broken.
    #
    # But "beats the floor by a margin I picked" is no better. The margin has to
    # be a TEST: Fisher's exact, one-sided, each arm's accepted count against the
    # noise arm's. Without it, 5/41 vs 2/41 reads as a win when it is p=0.22 --
    # and that mistake pointed the GDELT recommendation the wrong way for exactly
    # one commit.
    from scipy.stats import fisher_exact

    f0 = [r for r in by_rho.get(0.00, []) if r.get("p") is not None and (r.get("n") or 0) > gate]
    a0 = sum(1 for r in f0 if r.get("status") == "accepted")
    print(f"Noise floor (rho=0.00): {a0}/{len(f0)} accepted. Every arm is tested AGAINST this,")
    print("not against zero, and by Fisher's exact test rather than by eye.\n")
    print(f"{'rho':>6}{'accepted':>11}{'vs noise p':>13}   verdict")
    print("-" * 60)
    distinguishable = []
    for rho in sorted((r for r in by_rho if r), reverse=True):
        f = [r for r in by_rho[rho] if r.get("p") is not None and (r.get("n") or 0) > gate]
        a = sum(1 for r in f if r.get("status") == "accepted")
        if not f0 or not f:
            continue
        _, pv = fisher_exact([[a, len(f) - a], [a0, len(f0) - a0]], alternative="greater")
        ok = pv < 0.05
        if ok:
            distinguishable.append(rho)
        print(f"{rho:>6.2f}{a:>8}/{len(f):<3}{pv:>13.4f}   "
              f"{'DISTINGUISHABLE' if ok else 'indistinguishable from noise'}")

    print("\n" + "=" * 70)
    if not distinguishable:
        print("VERDICT: no feed quality tested, including the oracle, is separable from")
        print("noise. The constraint is the METHOD at these sample sizes.")
        return
    best = min(distinguishable)
    print(f"VERDICT: the weakest feed separable from noise is rho = {best:.2f} "
          f"({MEANING.get(best,'')}).")
    if best <= 0.08:
        print("A feed of realistic quality clears the bar -- building real sentiment")
        print("ingestion (GDELT + entity extraction) is justified on these numbers.")
    else:
        print("Realistic news sentiment (rho ~ 0.04-0.08) is NOT separable from noise:")
        print("it produces the same number of acceptances as a feed containing no")
        print("information at all. Broad general-news ingestion (GDELT) would not pay")
        print("for itself. The effort belongs elsewhere -- more events per candidate,")
        print("or a narrow high-signal source (exchange listings, regulatory filings,")
        print("protocol incidents) whose correlation with returns is far above what")
        print("broad news sentiment achieves.")


if __name__ == "__main__":
    main()
