"""Does Haiku propose the same conditions Sonnet does, at a third of the price?

Sonnet judgment is this project's largest running cost: 212 measured calls at
$0.0153 each, roughly 1,200 over a full replay. Haiku 4.5 is priced at exactly
one third on both input and output. Whether that is a saving or a false economy
is an empirical question, and a cheap one.

TWO QUESTIONS, NOT ONE. Substitutability -- does Haiku propose what Sonnet
proposes -- and QUALITY: which model is the better analyst. They are different,
and agreement cannot answer the second, since two models can agree perfectly
while both being useless. Quality is scored by `_quality()` on the p-value
distribution of each model's own proposals, which costs nothing beyond local
compute.

ON SUBSTITUTABILITY. Sonnet is taken as the reference:
in a given situation it proposes some set of conditions, and what matters is
whether Haiku proposes the same ones. This is deliberately not a test of which
model is *better*.

An earlier version of this file scored both models against the downstream gates
instead -- does the proposal parse, is it on-thesis, is it measurable, does it
survive `pattern_significance`. That answers a different question and answers it
badly here, for a reason worth recording: acceptance is rare for BOTH models, so
the comparison would almost always come back with neither model producing an
accepted condition and nothing to separate them. It would have cost real money
to return "undecided". Agreement is far denser per event, because every paired
event yields a measurement rather than only the rare accepted one.

MEASURING "THE SAME CONDITIONS" -- not textually. Two models never emit the same
JSON: different thresholds, different clause order, sometimes a different
indicator expressing the same idea. Matching on text would score formatting.

The honest question is whether the two conditions FIRE ON THE SAME DAYS.
`behavioural_agreement` returns the Jaccard overlap of the (coin, day) pairs
each spec selects, as of the simulated date: 1.0 identical behaviour, 0.0
disjoint. Two specs with different thresholds that fire on 90% of the same days
are the same hypothesis; two with similar-looking clauses that never coincide
are not, whatever their labels say.

THE CONTROL THAT MAKES THE NUMBER MEAN ANYTHING. `temperature=0` is rejected by
the API, so Sonnet is not deterministic: asked the same question twice it does
not return the same proposal. Without knowing that self-agreement, "Haiku agrees
with Sonnet 0.6 of the time" is uninterpretable -- if Sonnet agrees with ITSELF
0.65 of the time, Haiku is very nearly a drop-in substitute; if Sonnet
self-agrees at 0.95, Haiku is proposing something else.

So every event is judged THREE times: Sonnet twice and Haiku once. Sonnet-vs-
Sonnet is the ceiling any second model could reach; Haiku-vs-Sonnet is measured
against that ceiling, not against 1.0. Same reasoning as the noise floor in
forecast/sentiment_power.py -- an arm is compared to its control, never to
perfection.

Cost: 2 Sonnet + 1 Haiku per event ~= $0.036. Nothing here is evidence about
markets; historical data is used only as realistic input.

SAMPLE SIZE IS COUNTED IN PAIRS, NOT EVENTS -- see `--pairs`. An event yields a
comparable pair only when all three judgments produce a condition, and barely
half of real calls do, so a fixed event count fixes the wrong quantity.

Run:  python3 -m forecast.model_comparison --dry-run   # sample + cost, free
      python3 -m forecast.model_comparison --pairs 12 --max-spend 2.50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

RESULTS_PATH = Path(__file__).resolve().parent / "model_comparison.json"

SAMPLE_START = pd.Timestamp("2019-01-01")   # past the indicators' warm-up
SAMPLE_END = pd.Timestamp("2025-12-31")
SEED = 20260831

# Measured, not assumed: llm_pipeline/usage.py over 212 real judgment calls.
SONNET_PER_CALL = 0.0153
HAIKU_PER_CALL = SONNET_PER_CALL / 3
# Share of judgment calls that yield a condition the validator accepts: 118
# usable specs out of 212 real calls. The rest are no_action or refused, and
# neither produces anything to compare.
USABLE_SPEC_RATE = 118 / 212


def _sample_events(n: int) -> list[dict]:
    """Real trigger days, drawn deterministically and rebuilt exactly as
    `replay/engine.py::advance` builds them -- the same `compression_exit`, the
    same three-phase event text. If this drifted from the engine the comparison
    would measure the models on a task neither is actually asked to do, which is
    why this file was deferred until the trigger design settled."""
    import numpy as np

    from candidates.data_loading import load_daily
    from candidates.methodology import compression_exit
    from candidates.run_battery import COINS
    from replay import judgment

    rng = np.random.default_rng(SEED)
    pool = []
    for coin in COINS:
        ohlc = load_daily(coin)
        idx = ohlc.index[(ohlc.index >= SAMPLE_START) & (ohlc.index <= SAMPLE_END)]
        for d in idx:
            episode = compression_exit(ohlc, d)
            if episode is not None:
                pool.append({"coin": coin, "as_of": episode["b_date"], "episode": {"symbol": coin, **episode}})
    if not pool:
        return []
    chosen = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    picks = [pool[i] for i in sorted(chosen)]
    for p in picks:
        p["desc"] = judgment.format_compression_event(p["coin"], p["episode"])
    return picks


def _specs_of(raw: dict | None) -> list:
    """Every usable condition in one judgment. A call now returns up to two, and
    a proposal the validator refuses counts as no condition, which is what it is
    downstream."""
    from llm_pipeline.novel_condition_tester import proposals_from_assessment, spec_from_proposal

    out = []
    for d in proposals_from_assessment(raw or {}):
        spec, _ = spec_from_proposal(d)
        if spec is not None:
            out.append(spec)
    return out


def _compare(a: list, b: list, as_of) -> dict:
    """Agreement between two judgments, each of which may hold one or two
    conditions.

    A SET's behaviour is the union of the days its conditions fire on, and two
    sets agree to the extent those unions coincide. Comparing sets pairwise would
    need an arbitrary matching rule and would break on a one-versus-two
    comparison; the union has neither problem and is the honest question anyway
    -- "would these two judgments have selected the same moments".

    Agreeing to propose NOTHING is real agreement and is recorded as such.
    Dropping those events would restrict the comparison to cases where the
    reference model happened to be talkative, and a model that correctly stays
    quiet is exactly what a cheaper substitute should do."""
    from candidates.run_battery import COINS
    from llm_pipeline.novel_condition_tester import occurrence_set

    out = {"same_action": (len(a) == 0) == (len(b) == 0),
           "both_proposed": bool(a) and bool(b),
           "both_silent": not a and not b,
           "n_a": len(a), "n_b": len(b), "overlap": None}
    if not out["both_proposed"]:
        return out
    sa = set().union(*(occurrence_set(s, COINS, as_of=as_of) for s in a))
    sb = set().union(*(occurrence_set(s, COINS, as_of=as_of) for s in b))
    union = sa | sb
    out["overlap"] = (len(sa & sb) / len(union)) if union else float("nan")
    out["labels_a"] = [s.label for s in a]
    out["labels_b"] = [s.label for s in b]
    return out



def _quality(specs: list, as_of) -> dict:
    """How good the proposals ARE, independent of who else agrees with them.

    Agreement answers substitutability; it does not answer which model is the
    better analyst, and two models can agree perfectly while both being useless.
    This is the second question, and it is free -- `test_novel_condition` is
    local compute, no API call.

    Scored on the P-VALUE distribution rather than on acceptances, deliberately.
    Acceptance is a rare binary outcome (measured at 8% across this grammar), so
    counting acceptances over a few dozen proposals yields two or three events
    and separates nothing -- that is why an earlier version of this file, which
    scored exactly that way, was discarded. A p-value is continuous, defined for
    every testable proposal, and is what the acceptance is a threshold on: an
    analyst whose hypotheses land at p=0.2 is finding something an analyst whose
    hypotheses land at p=0.6 is not, long before either clears a gate.

    Also records the cheaper gates, which are denser still: how many proposals
    were on-thesis at all, and how many were testable without being loosened."""
    from candidates.run_battery import COINS
    from llm_pipeline.novel_condition_tester import is_testable, test_novel_condition

    out = {"n_specs": len(specs), "testable": 0, "tested": 0,
           "accepted": 0, "p_values": [], "excess": []}
    for spec in specs:
        if is_testable(spec, COINS, as_of=as_of) is not None:
            continue
        out["testable"] += 1
        try:
            res = test_novel_condition(spec, COINS, as_of=as_of)
        except Exception:
            continue
        pat = res.get("pattern_significance") or {}
        out["tested"] += 1
        if res.get("status") == "accepted":
            out["accepted"] += 1
        if pat.get("p_value") is not None:
            out["p_values"].append(float(pat["p_value"]))
        if pat.get("excess_return") is not None:
            out["excess"].append(float(pat["excess_return"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    # The target is PAIRS, not events, because pairs are what carry the power and
    # an event yields one only when all three judgments produce a condition.
    # Measured propose-and-validate rate: 118 usable specs from 212 real calls
    # (55.7%), so between 17% (if the three calls decide independently) and 56%
    # (if they move together) of events yield a pair -- a factor of three that
    # cannot be known before running. Fixing the event count fixes the wrong
    # quantity: 25 events can land anywhere between 4 usable pairs and 14, and at
    # 4 the one-sided Wilcoxon CANNOT return p < 0.05 at all (its smallest
    # attainable p is 1/2^4 = 0.0625), so the run would be incapable of a result
    # regardless of how different the models are.
    #
    # 12 pairs is the default because that is where power against a 0.20 overlap
    # gap reaches 93%; 6 pairs gives 63%, 4 gives nothing.
    ap.add_argument("--pairs", type=int, default=12, help="usable pairs to collect (default 12)")
    ap.add_argument("--max-spend", type=float, default=2.50, help="USD cap (default 2.50)")
    ap.add_argument("--dry-run", action="store_true", help="sample and cost only")
    args = ap.parse_args()

    # HAIKU_MODEL is defined here rather than imported: the Haiku HEADLINE path
    # was removed from the pipeline on 2026-09-02, and this experiment -- which
    # asks the separate question of whether Haiku could replace Sonnet as the
    # JUDGE -- is its only remaining caller. Kept so the committed result set
    # (forecast/model_comparison.json) stays reproducible.
    from llm_pipeline.haiku_sonnet_pipeline import SONNET_MODEL
    from replay import judgment

    HAIKU_MODEL = "claude-haiku-4-5"

    per_event = 2 * SONNET_PER_CALL + HAIKU_PER_CALL
    max_events = int(args.max_spend / per_event)
    events = _sample_events(max_events)
    print(f"target: {args.pairs} usable pairs, spending at most ${args.max_spend:.2f} "
          f"({max_events} events available, ${per_event:.4f} each)")

    if args.dry_run:
        print(f"\nMeasured usable-spec rate: {USABLE_SPEC_RATE:.1%} per call, so an event")
        print(f"yields a pair with probability between {USABLE_SPEC_RATE**3:.0%} (independent) "
              f"and {USABLE_SPEC_RATE:.0%} (correlated).")
        lo = args.pairs / USABLE_SPEC_RATE
        hi = args.pairs / USABLE_SPEC_RATE ** 3
        print(f"Reaching {args.pairs} pairs therefore needs roughly {lo:.0f}-{hi:.0f} events: "
              f"${lo*per_event:.2f}-${hi*per_event:.2f}.")
        print(f"The ${args.max_spend:.2f} cap stops it either way, reporting how far it got.")
        print("\nThe second Sonnet call is the control, not overhead: without it the")
        print("Haiku number has no ceiling to be read against.")
        print("\nSampling only -- no API calls made. Drop --dry-run to run it.")
        return 0

    load_dotenv()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    done = {r["key"]: r for r in json.loads(RESULTS_PATH.read_text())} if RESULTS_PATH.exists() else {}

    def _pairs_so_far() -> int:
        return sum(1 for r in done.values()
                   if (r.get("ceiling") or {}).get("overlap") is not None
                   and (r.get("haiku_vs_sonnet") or {}).get("overlap") is not None)

    t0 = time.time()
    for ev in events:
        if _pairs_so_far() >= args.pairs:
            print(f"\nReached {args.pairs} usable pairs after {len(done)} events "
                  f"(~${len(done)*per_event:.2f}).")
            break
        if len(done) * per_event >= args.max_spend:
            print(f"\nSPEND CAP: stopped at {len(done)} events (~${len(done)*per_event:.2f}) "
                  f"with {_pairs_so_far()}/{args.pairs} usable pairs.")
            print("Raise --max-spend, or read the same-action agreement, which is")
            print("defined on every event and does not need pairs.")
            break
        key = f"compression_{ev['as_of'].date()}_{ev['coin']}"
        if key in done:
            continue
        row = {"key": key, "as_of": str(ev["as_of"].date()),
               "coin": ev["coin"], "desc": ev["desc"][:200]}
        specs = {}
        for tag, model in (("sonnet_a", SONNET_MODEL), ("sonnet_b", SONNET_MODEL),
                           ("haiku", HAIKU_MODEL)):
            try:
                raw = judgment.judge_event(ev["desc"], client, as_of=ev["as_of"],
                                            coin=ev.get("coin"), model=model)
                specs[tag] = _specs_of(raw)
                row[f"{tag}_action"] = (raw or {}).get("recommended_action")
                row[f"{tag}_labels"] = [s.label for s in specs[tag]]
            except Exception as e:
                # A response that cannot be parsed IS a result: it means the model
                # does not hold the output contract, which disqualifies it as a
                # substitute regardless of how well it agrees when it does parse.
                specs[tag] = []
                row[f"{tag}_error"] = str(e)[:160]
        row["ceiling"] = _compare(specs["sonnet_a"], specs["sonnet_b"], ev["as_of"])
        row["haiku_vs_sonnet"] = _compare(specs["sonnet_a"], specs["haiku"], ev["as_of"])
        # The second question, and the one agreement cannot answer: which model
        # proposes hypotheses that hold up. Free -- local compute, no API call.
        for tag in ("sonnet_a", "sonnet_b", "haiku"):
            row[f"{tag}_quality"] = _quality(specs[tag], ev["as_of"])
        done[key] = row
        RESULTS_PATH.write_text(json.dumps(list(done.values()), indent=1))
        c, h = row["ceiling"]["overlap"], row["haiku_vs_sonnet"]["overlap"]
        print(f"  {len(done)}ev {_pairs_so_far()}/{args.pairs}pr  {key:<34} "
              f"S/S={'--' if c is None else f'{c:.2f}'}  "
              f"S/H={'--' if h is None else f'{h:.2f}'}  ({(time.time()-t0)/60:.1f}m)",
              flush=True)

    print("\nDONE -- run: python3 -m forecast.analyse_model_comparison")
    return 0


if __name__ == "__main__":
    sys.exit(main())
