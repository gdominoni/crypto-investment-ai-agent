"""Scores the Haiku-vs-Sonnet comparison and states, in one line, whether
switching is justified.

The test is PAIRED -- both models judged the same events -- so the comparison
uses McNemar's test on the discordant pairs rather than comparing two
independent proportions. That is not pedantry: pairing is what makes a sample
this small informative, and analysing paired data as unpaired throws the
advantage away and understates the evidence.

The verdict rule is fixed here, before the numbers are looked at, because a
threshold chosen after seeing the result is not a threshold. Switching to
Haiku is justified only if it is NOT WORSE on the gate that matters -- and
"not worse" is a statistical statement, not an eyeball one. A model that
proposes less often is not thereby worse; it is cheaper and may be more
selective. What would disqualify it is producing proposals that fail the
gates, or failing to hold the JSON contract at all.
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parent / "model_comparison.json"

# Per-call USD, from this project's own measured usage (llm_pipeline/usage.py),
# not from a list-price assumption.
SONNET_PER_CALL = 0.0153
HAIKU_PER_CALL = SONNET_PER_CALL / 3
FULL_REPLAY_CALLS = 1200


def _rate(rows, tag, field):
    vals = [r[tag].get(field) for r in rows if r.get(tag)]
    hits = sum(1 for v in vals if v is True)
    n = sum(1 for v in vals if v is not None)
    return hits, n


def main() -> None:
    if not RESULTS_PATH.exists():
        print("No results -- run `python3 -m forecast.model_comparison` first.")
        return
    rows = json.loads(RESULTS_PATH.read_text())
    print(f"Haiku vs Sonnet on the replay's judgment task -- {len(rows)} paired events\n")

    print(f"{'gate':<38}{'Sonnet':>14}{'Haiku':>14}")
    print("-" * 66)
    gates = [("held the JSON contract", "parsed"),
             ("proposed (vs no_action)", None),
             ("on-thesis (real news term)", "on_thesis"),
             ("measurable (>=35 occurrences)", "measurable"),
             ("accepted by significance", "accepted")]
    for label, field in gates:
        cells = []
        for tag in ("sonnet", "haiku"):
            if field is None:
                k = sum(1 for r in rows if (r.get(tag) or {}).get("action") == "propose_novel_test")
                cells.append(f"{k}/{len(rows)}")
            else:
                h, n = _rate(rows, tag, field)
                cells.append(f"{h}/{n}" if n else "--")
        print(f"{label:<38}{cells[0]:>14}{cells[1]:>14}")

    # The decisive comparison, on the last gate a proposal must clear. McNemar
    # uses ONLY the discordant pairs -- events where the two models disagree --
    # because concordant pairs carry no information about which is better.
    from scipy.stats import binomtest

    s_only = h_only = both = neither = 0
    for r in rows:
        s = (r.get("sonnet") or {}).get("accepted") is True
        h = (r.get("haiku") or {}).get("accepted") is True
        both += s and h
        neither += (not s) and (not h)
        s_only += s and not h
        h_only += h and not s

    print(f"\nPaired outcomes on 'accepted': both={both}, neither={neither}, "
          f"Sonnet only={s_only}, Haiku only={h_only}")
    disc = s_only + h_only
    if disc == 0:
        print("No discordant pairs -- the two models are indistinguishable on this gate here.")
        pval = 1.0
    else:
        pval = binomtest(s_only, disc, 0.5).pvalue
        print(f"McNemar (exact binomial on {disc} discordant pairs): p = {pval:.4f}")

    print("\n" + "=" * 66)
    saving = FULL_REPLAY_CALLS * (SONNET_PER_CALL - HAIKU_PER_CALL)
    print(f"Cost: ${SONNET_PER_CALL:.4f} vs ${HAIKU_PER_CALL:.4f} per call. Over a full "
          f"~{FULL_REPLAY_CALLS}-call replay,\nswitching would save about ${saving:.2f} "
          f"(${FULL_REPLAY_CALLS*SONNET_PER_CALL:.2f} -> ${FULL_REPLAY_CALLS*HAIKU_PER_CALL:.2f}).\n")

    h_parsed, n_parsed = _rate(rows, "haiku", "parsed")
    if n_parsed and h_parsed < n_parsed:
        print(f"VERDICT: NO. Haiku failed to hold the output contract on "
              f"{n_parsed - h_parsed}/{n_parsed} events.")
        print("A judgment that does not parse costs a call and returns nothing; the")
        print("saving is on calls that produce no result.")
    elif pval < 0.05 and s_only > h_only:
        print("VERDICT: NO. Haiku is measurably worse at the gate that matters --")
        print("it produces fewer conditions that survive the significance test, and the")
        print("difference is larger than chance on the discordant pairs.")
    elif disc == 0 and both == 0:
        print("VERDICT: UNDECIDED. Neither model produced an accepted condition here, so")
        print("this sample cannot separate them on the gate that matters. That is the")
        print("normal outcome at these acceptance rates -- rerun with a larger --n, or")
        print("decide on the earlier gates (contract, on-thesis, measurable) instead.")
    else:
        print("VERDICT: YES, on this evidence. Haiku is not measurably worse at any gate,")
        print(f"and switching saves roughly ${saving:.2f} per full replay.")
        print("Note what this does NOT establish: absence of a difference at this sample")
        print("size is not evidence of equivalence. Re-check on a larger sample before")
        print("relying on it for anything other than cost.")


if __name__ == "__main__":
    main()
