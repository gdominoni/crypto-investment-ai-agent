"""Could Haiku do the replay's judgment job, at a third of the price?

Sonnet is the single largest running cost in this project: measured over a
partial replay, 212 judgment calls cost $3.25 at $0.0153 each, and a full
5.5-year run makes roughly 1,200 of them. Haiku 4.5 is priced at exactly one
third of Sonnet on both input and output, so the same workload would cost
about a third as much. Whether that is a saving or a false economy is an
empirical question, and a cheap one to answer.

WHY NOT "SEND TEN QUESTIONS AND COMPARE THE ANSWERS". Reading two prose
assessments side by side and judging which is better is exactly the kind of
unfalsifiable comparison this project avoids everywhere else. It has no
ground truth, it scores fluency rather than usefulness, and it would be
graded by the same class of model being tested.

Every proposal in this pipeline already passes through deterministic gates,
and those gates ARE the criteria. A judgment is useful here if it produces a
condition that is well-formed, on-thesis, measurable, and survives the
statistical test. All four are checkable in code, for free, with no second
opinion required:

  1. PARSES        -- valid JSON in the expected schema at all
  2. ON-THESIS     -- survives spec_from_proposal: carries a real news/macro
                      term, uses no banned indicator
  3. MEASURABLE    -- has occurred >= MIN_HISTORICAL_OCCURRENCES times as of
                      the simulated date, or can be relaxed until it has
  4. SURVIVES      -- pattern_significance accepts it (the only one that
                      costs real compute, and still no API call)

Plus two descriptive measures that decide whether a model is usable at all:
the rate at which it proposes rather than returning no_action (a model that
never proposes is cheap and useless; one that always proposes is expensive
and noisy), and its real measured cost per call.

DESIGN. Paired: both models see the SAME events, the same time-sandboxed
context, the same system prompt. Pairing removes between-event variance,
which is large here -- some days genuinely have nothing to propose -- and it
is what makes a modest sample informative.

SAMPLE SIZE. Ten events, the number that prompted this, cannot distinguish
anything short of total failure: at n=10 a 60%-vs-30% difference in pass rate
is well inside noise. The binding constraint is not money -- a paired event
costs about $0.02 -- so the default is larger. At n=40 the comparison can
actually resolve a difference worth acting on, for well under a dollar.

Stratified across both trigger types, because they are different tasks: a
macro release comes with a snapshot of every coin, a shock with one coin's
snapshot plus its lead-up.

NOTHING HERE IS EVIDENCE ABOUT MARKETS. It is a measurement of two models on
one task, using historical data purely as realistic input.

Run:  python3 -m forecast.model_comparison            # default n=40
      python3 -m forecast.model_comparison --n 10     # the cheap version
      python3 -m forecast.model_comparison --dry-run  # cost estimate, no calls
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

# Both trigger types, sampled evenly. The replay window, avoiding the first
# months where the trailing windows every indicator depends on are still warming
# up and would fail for reasons that have nothing to do with the model.
SAMPLE_START = pd.Timestamp("2019-01-01")
SAMPLE_END = pd.Timestamp("2025-12-31")
SEED = 20260831


def _sample_events(n: int) -> list[dict]:
    """Real trigger days, drawn deterministically, reconstructed exactly as
    `replay/engine.py::advance` would build them -- same `release_dates`
    filter, same `_shock_transition` rule, same event description. If this
    drifted from the engine the comparison would measure the models on a task
    neither is asked to do."""
    import numpy as np

    from candidates.data_loading import load_daily
    from candidates.macro_vintage import MACRO_SERIES
    from candidates.run_battery import COINS
    from replay import judgment
    from replay.engine import _shock_transition
    from replay.time_sandbox import latest_release_with_prior, release_dates

    rng = np.random.default_rng(SEED)

    macro = []
    for key, label in MACRO_SERIES.items():
        for d in release_dates(key, SAMPLE_START, SAMPLE_END, new_periods_only=True):
            macro.append({"kind": "macro", "series": key, "label": label, "as_of": d})

    shock = []
    ohlc = {c: load_daily(c) for c in COINS}
    for coin in COINS:
        idx = ohlc[coin].index
        for d in idx[(idx >= SAMPLE_START) & (idx <= SAMPLE_END)]:
            t = _shock_transition(ohlc[coin], d)
            if t is not None:
                shock.append({"kind": "shock", "coin": coin, "as_of": d, "z": t[0], "direction": t[1]})

    half = n // 2
    picks = []
    for pool, k in ((macro, half), (shock, n - half)):
        if not pool:
            continue
        chosen = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        picks += [pool[i] for i in sorted(chosen)]

    for p in picks:
        if p["kind"] == "macro":
            rel = latest_release_with_prior(p["series"], p["as_of"])
            p["desc"] = judgment.format_macro_event(p["label"], rel) if rel else None
        else:
            p["desc"] = judgment.format_shock_event(p["coin"], p["z"], p["direction"])
    return [p for p in picks if p.get("desc")]


def _score(raw: dict, as_of: pd.Timestamp) -> dict:
    """Grade one assessment against the four gates it would really face.

    Deliberately identical to the production path rather than a
    reimplementation of it: the same `spec_from_proposal`, the same
    `count_occurrences` with the same `as_of`, the same `relax_to_testable`,
    the same `test_novel_condition`. A model is being judged on whether the
    system it feeds can use its output, not on a proxy for that."""
    from candidates.run_battery import COINS
    from llm_pipeline.novel_condition_tester import (MIN_HISTORICAL_OCCURRENCES, count_occurrences,
                                                     relax_to_testable, spec_from_proposal,
                                                     test_novel_condition)

    out = {"parsed": raw is not None, "action": None, "on_thesis": None,
           "measurable": None, "relaxed": None, "n_occurrences": None,
           "accepted": None, "p_value": None, "reject_reason": None}
    if raw is None:
        return out
    out["action"] = raw.get("recommended_action")
    if out["action"] != "propose_novel_test" or not raw.get("novel_condition_spec"):
        return out  # a considered no_action is a legitimate answer, not a failure

    spec, err = spec_from_proposal(raw["novel_condition_spec"])
    out["on_thesis"] = spec is not None
    if spec is None:
        out["reject_reason"] = (err or "")[:160]
        return out

    n = count_occurrences(spec, COINS, as_of=as_of)
    out["n_occurrences"] = n
    if n < MIN_HISTORICAL_OCCURRENCES:
        rescued = relax_to_testable(spec, COINS, as_of=as_of)
        out["relaxed"] = rescued is not None
        if rescued is None:
            out["measurable"] = False
            out["reject_reason"] = f"too rare ({n}), unrescuable"
            return out
        spec = rescued[0]
    out["measurable"] = True

    try:
        res = test_novel_condition(spec, COINS)
        pat = res.get("pattern_significance") or {}
        out["accepted"] = res.get("status") == "accepted"
        out["p_value"] = pat.get("p_value")
    except Exception as e:
        out["reject_reason"] = f"test failed: {str(e)[:120]}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="paired events (default 40)")
    ap.add_argument("--dry-run", action="store_true", help="sample and cost, no API calls")
    args = ap.parse_args()

    from llm_pipeline.haiku_sonnet_pipeline import HAIKU_MODEL, SONNET_MODEL
    from replay import judgment

    events = _sample_events(args.n)
    print(f"{len(events)} paired events "
          f"({sum(1 for e in events if e['kind']=='macro')} macro, "
          f"{sum(1 for e in events if e['kind']=='shock')} shock)")

    if args.dry_run:
        # From this project's own measured usage, not an assumption: $0.0153 per
        # Sonnet judgment call over 212 real calls, and Haiku is priced at exactly
        # one third on both input and output.
        per_sonnet = 0.0153
        cost = len(events) * (per_sonnet + per_sonnet / 3)
        print(f"\nEstimated cost: ${cost:.2f} "
              f"(${len(events)*per_sonnet:.2f} Sonnet + ${len(events)*per_sonnet/3:.2f} Haiku)")
        print("Sampling only -- no API calls made. Drop --dry-run to run it.")
        return 0

    load_dotenv()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    done = {r["key"]: r for r in json.loads(RESULTS_PATH.read_text())} if RESULTS_PATH.exists() else {}

    t0 = time.time()
    for i, ev in enumerate(events):
        key = f"{ev['kind']}_{ev['as_of'].date()}_{ev.get('coin') or ev.get('series')}"
        if key in done:
            continue
        row = {"key": key, "kind": ev["kind"], "as_of": str(ev["as_of"].date()),
               "desc": ev["desc"][:200]}
        for tag, model in (("sonnet", SONNET_MODEL), ("haiku", HAIKU_MODEL)):
            try:
                raw = judgment.judge_event(ev["desc"], client, as_of=ev["as_of"],
                                            coin=ev.get("coin"), model=model)
            except Exception as e:
                # A malformed response IS a result -- it means the model cannot
                # hold the output contract, which is the thing being measured.
                raw = None
                row[f"{tag}_error"] = str(e)[:160]
            row[tag] = _score(raw, ev["as_of"])
        done[key] = row
        RESULTS_PATH.write_text(json.dumps(list(done.values()), indent=1))
        el = time.time() - t0
        print(f"  {len(done)}/{len(events)}  {key:<40} "
              f"S={row['sonnet']['action']}  H={row['haiku']['action']}  "
              f"({el/60:.1f}m)", flush=True)

    print("\nDONE -- run: python3 -m forecast.analyse_model_comparison")
    return 0


if __name__ == "__main__":
    sys.exit(main())
