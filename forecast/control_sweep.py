"""POSITIVE CONTROL, full grammar: if a genuinely informative sentiment
signal EXISTED, would this system's discovery machinery actually find it?

`forecast/positive_control.py` established that `pattern_significance`
fires on a planted effect in isolation. That is a narrower claim than it
sounds. This project's thesis is a CONJUNCTION -- specific market
conditions AND a news/sentiment event -- so the question that matters is
whether a planted sentiment signal survives being combined with every
market-state term in the whitelist, tested through the same nested
(incremental) baseline every real on-thesis condition must pass.

This is the same sweep as `forecast/grammar_sweep.py`, with the real
news/macro event terms swapped for synthetic ones of KNOWN quality:

    planted20/35/50 -- sentiment that genuinely predicts (by lookahead
                       construction), at three strengths
    real_news       -- the dated crypto-news events (open question)
    random          -- pure noise (must not be discovered)

Reading it. The planted arms are the ceiling: they are what a PERFECT
news feed would look like. The random arm is the floor. Where the real
grammar sweep landed relative to those two says whether the architecture
is capable of the job it was built for, independently of whether crypto
news happens to contain signal.

This bears directly on whether the GDELT news backfill is worth doing: if
even a perfect sentiment signal cannot be discovered through this
grammar, better news data will not help, and the constraint is the
method rather than the inputs.

Nothing here is evidence about markets. The planted arms use future
information deliberately.

Run:  python3 -m forecast.control_sweep
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parent / "control_sweep.json"

from forecast.grammar_sweep import COINS, DIRECTIONS, STATE_TERMS, WITHIN_DAYS
from forecast.positive_control import make_planted, make_random, make_real_news

ARMS = {
    "planted20": lambda: make_planted(0.20),
    "planted35": lambda: make_planted(0.35),
    "planted50": lambda: make_planted(0.50),
    "real_news": make_real_news,
    "random": make_random,
}


def register_arms() -> None:
    """Install every synthetic sentiment indicator into the REAL registry.

    Each must go into BOTH `NEWS_EVENT_INDICATORS` (what may be proposed)
    and `EVENT_INDICATORS` (what is stripped to build the control group).
    Registering in only the first leaves the indicator inside its own
    control, which yields an empty contrast and a silent
    `insufficient_data` -- see positive_control.py.
    """
    import llm_pipeline.novel_condition_tester as N

    for name, factory in ARMS.items():
        key = f"sent_{name}"
        N.SUPPORTED_INDICATORS[key] = factory()
        N.NEWS_EVENT_INDICATORS = frozenset(set(N.NEWS_EVENT_INDICATORS) | {key})
        N.EVENT_INDICATORS = frozenset(set(N.EVENT_INDICATORS) | {key})
        N.DAILY_NATIVE_INDICATORS = frozenset(set(N.DAILY_NATIVE_INDICATORS) | {key})
        N.INDICATOR_PLAIN_NAMES[key] = f"synthetic sentiment [{name}] (TEST ONLY)"


def build_specs():
    from llm_pipeline.novel_condition_tester import Clause, ConditionSpec

    specs = []
    for arm in ARMS:
        key = f"sent_{arm}"
        ev = Clause(indicator=key, op=">=", threshold=1.0, within_days=0)
        # The event term on its own, for reference against the combinations.
        for d in DIRECTIONS:
            specs.append(ConditionSpec(label=f"ctl_{arm}__alone__{d}", clauses=(ev,), direction=d))
        for si, so, st in STATE_TERMS:
            for w in WITHIN_DAYS:
                for d in DIRECTIONS:
                    specs.append(ConditionSpec(
                        label=f"ctl_{arm}__{si}_{so}_{st}__w{w}_{d}",
                        clauses=(ev, Clause(indicator=si, op=so, threshold=st, within_days=w)),
                        direction=d))
    return specs


def load_done() -> dict:
    if not RESULTS_PATH.exists():
        return {}
    return {r["label"]: r for r in json.loads(RESULTS_PATH.read_text())}


def main() -> None:
    register_arms()
    from llm_pipeline.novel_condition_tester import test_novel_condition

    specs = build_specs()
    done = load_done()
    todo = [s for s in specs if s.label not in done]
    print(f"{len(specs)} conditions ({len(ARMS)} sentiment arms x the full state grammar); "
          f"{len(done)} done, {len(todo)} to run", flush=True)

    t0 = time.time()
    for i, spec in enumerate(todo):
        try:
            r = test_novel_condition(spec, COINS)
            pat = r.get("pattern_significance") or {}
            done[spec.label] = {
                "label": spec.label, "arm": spec.label.split("__")[0].replace("ctl_", ""),
                "direction": spec.direction, "status": r.get("status"), "n": r.get("n"),
                "p": pat.get("p_value"), "excess": pat.get("excess_return"),
                "mfe_mae": pat.get("mfe_mae_ratio"), "significant": pat.get("significant"),
                "pat_status": pat.get("status"), "baseline_kind": pat.get("baseline_kind"),
            }
        except Exception as e:
            done[spec.label] = {"label": spec.label, "arm": spec.label.split("__")[0].replace("ctl_", ""),
                                "status": "error", "err": str(e)[:120]}
        RESULTS_PATH.write_text(json.dumps(list(done.values()), indent=1))
        if (i + 1) % 40 == 0:
            el = time.time() - t0
            rate = el / (i + 1)
            print(f"  {len(done)}/{len(specs)}  ({el/60:.0f}m, {rate:.1f}s/cond, "
                  f"~{rate*(len(todo)-i-1)/60:.0f}m left)", flush=True)
    print("DONE -- run: python3 -m forecast.analyse_control_sweep", flush=True)


if __name__ == "__main__":
    sys.exit(main())
