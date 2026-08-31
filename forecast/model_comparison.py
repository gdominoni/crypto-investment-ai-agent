"""Does Haiku propose the same conditions Sonnet does, at a third of the price?

Sonnet judgment is this project's largest running cost: 212 measured calls at
$0.0153 each, roughly 1,200 over a full replay. Haiku 4.5 is priced at exactly
one third on both input and output. Whether that is a saving or a false economy
is an empirical question, and a cheap one.

THE QUESTION IS SUBSTITUTABILITY, NOT QUALITY. Sonnet is taken as the reference:
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

Run:  python3 -m forecast.model_comparison --dry-run   # sample + cost, free
      python3 -m forecast.model_comparison --n 25
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


def _sample_events(n: int) -> list[dict]:
    """Real trigger days, drawn deterministically and rebuilt exactly as
    `replay/engine.py::advance` builds them -- same `release_dates` filter
    (`new_periods_only`), same `_shock_transition` rule, same event text. If
    this drifted from the engine the comparison would measure the models on a
    task neither is actually asked to do."""
    import numpy as np

    from candidates.data_loading import load_daily
    from candidates.macro_vintage import MACRO_SERIES
    from candidates.run_battery import COINS
    from replay import judgment
    from replay.engine import _shock_transition
    from replay.time_sandbox import latest_release_with_prior, release_dates

    rng = np.random.default_rng(SEED)

    macro = [{"kind": "macro", "series": k, "label": lbl, "as_of": d}
             for k, lbl in MACRO_SERIES.items()
             for d in release_dates(k, SAMPLE_START, SAMPLE_END, new_periods_only=True)]

    shock, ohlc = [], {c: load_daily(c) for c in COINS}
    for coin in COINS:
        idx = ohlc[coin].index
        for d in idx[(idx >= SAMPLE_START) & (idx <= SAMPLE_END)]:
            t = _shock_transition(ohlc[coin], d)
            if t is not None:
                shock.append({"kind": "shock", "coin": coin, "as_of": d,
                              "z": t[0], "direction": t[1]})

    half = n // 2
    picks = []
    for pool, k in ((macro, half), (shock, n - half)):
        if pool:
            chosen = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
            picks += [pool[i] for i in sorted(chosen)]

    for p in picks:
        if p["kind"] == "macro":
            rel = latest_release_with_prior(p["series"], p["as_of"])
            p["desc"] = judgment.format_macro_event(p["label"], rel) if rel else None
        else:
            p["desc"] = judgment.format_shock_event(p["coin"], p["z"], p["direction"])
    return [p for p in picks if p.get("desc")]


def _spec_of(raw: dict | None):
    """The proposed condition, or None when the model proposed nothing (or
    proposed something the validator refuses). A refused proposal is treated as
    'no usable condition', which is what it is downstream."""
    from llm_pipeline.novel_condition_tester import spec_from_proposal

    if not raw or raw.get("recommended_action") != "propose_novel_test":
        return None
    if not raw.get("novel_condition_spec"):
        return None
    spec, _ = spec_from_proposal(raw["novel_condition_spec"])
    return spec


def _compare(a, b, as_of) -> dict:
    """Agreement between two judgments, at both levels that matter.

    Agreeing to propose NOTHING is real agreement and is recorded as such --
    dropping those events would silently restrict the comparison to the cases
    where the reference model happened to be talkative, and a model that
    correctly stays quiet is exactly what a cheaper substitute should do."""
    from candidates.run_battery import COINS
    from llm_pipeline.novel_condition_tester import behavioural_agreement

    out = {"same_action": (a is None) == (b is None), "overlap": None,
           "both_proposed": a is not None and b is not None,
           "both_silent": a is None and b is None}
    if out["both_proposed"]:
        out["overlap"] = behavioural_agreement(a, b, COINS, as_of=as_of)
        out["same_direction"] = a.direction == b.direction
        out["indicators_a"] = sorted({c.indicator for c in a.clauses})
        out["indicators_b"] = sorted({c.indicator for c in b.clauses})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25, help="events (default 25)")
    ap.add_argument("--dry-run", action="store_true", help="sample and cost only")
    args = ap.parse_args()

    from llm_pipeline.haiku_sonnet_pipeline import HAIKU_MODEL, SONNET_MODEL
    from replay import judgment

    events = _sample_events(args.n)
    print(f"{len(events)} events ({sum(1 for e in events if e['kind']=='macro')} macro, "
          f"{sum(1 for e in events if e['kind']=='shock')} shock), judged 3x each "
          f"(Sonnet, Sonnet again, Haiku)")

    if args.dry_run:
        cost = len(events) * (2 * SONNET_PER_CALL + HAIKU_PER_CALL)
        print(f"\nEstimated cost: ${cost:.2f} "
              f"(${len(events)*2*SONNET_PER_CALL:.2f} Sonnet x2 + "
              f"${len(events)*HAIKU_PER_CALL:.2f} Haiku)")
        print("The second Sonnet call is the control, not overhead: without it the")
        print("Haiku number has no ceiling to be read against.")
        print("\nSampling only -- no API calls made. Drop --dry-run to run it.")
        return 0

    load_dotenv()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    done = {r["key"]: r for r in json.loads(RESULTS_PATH.read_text())} if RESULTS_PATH.exists() else {}

    t0 = time.time()
    for ev in events:
        key = f"{ev['kind']}_{ev['as_of'].date()}_{ev.get('coin') or ev.get('series')}"
        if key in done:
            continue
        row = {"key": key, "kind": ev["kind"], "as_of": str(ev["as_of"].date()),
               "desc": ev["desc"][:200]}
        specs = {}
        for tag, model in (("sonnet_a", SONNET_MODEL), ("sonnet_b", SONNET_MODEL),
                           ("haiku", HAIKU_MODEL)):
            try:
                raw = judgment.judge_event(ev["desc"], client, as_of=ev["as_of"],
                                            coin=ev.get("coin"), model=model)
                specs[tag] = _spec_of(raw)
                row[f"{tag}_action"] = (raw or {}).get("recommended_action")
                row[f"{tag}_label"] = ((raw or {}).get("novel_condition_spec") or {}).get("label")
            except Exception as e:
                # A response that cannot be parsed IS a result: it means the model
                # does not hold the output contract, which disqualifies it as a
                # substitute regardless of how well it agrees when it does parse.
                specs[tag] = None
                row[f"{tag}_error"] = str(e)[:160]
        row["ceiling"] = _compare(specs["sonnet_a"], specs["sonnet_b"], ev["as_of"])
        row["haiku_vs_sonnet"] = _compare(specs["sonnet_a"], specs["haiku"], ev["as_of"])
        done[key] = row
        RESULTS_PATH.write_text(json.dumps(list(done.values()), indent=1))
        c, h = row["ceiling"]["overlap"], row["haiku_vs_sonnet"]["overlap"]
        print(f"  {len(done)}/{len(events)}  {key:<38} "
              f"S/S={'--' if c is None else f'{c:.2f}'}  "
              f"S/H={'--' if h is None else f'{h:.2f}'}  ({(time.time()-t0)/60:.1f}m)",
              flush=True)

    print("\nDONE -- run: python3 -m forecast.analyse_model_comparison")
    return 0


if __name__ == "__main__":
    sys.exit(main())
