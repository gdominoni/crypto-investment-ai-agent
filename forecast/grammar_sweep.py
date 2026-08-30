"""Forecast whether the historical replay can produce an ACCEPTED (and
therefore possibly VALIDATED) candidate -- without spending anything on
Anthropic.

The idea. Sonnet only PROPOSES conditions; whether a proposal is accepted
is decided entirely by `pattern_significance` / `classify_status`, which
are offline and deterministic. So enumerating the grammar Sonnet draws
from and running every combination through the real
`test_novel_condition` pipeline bounds what the replay can possibly find.

Why it is an UPPER BOUND, not merely an approximation. This sweep tests
each condition against the FULL 2017-2026 history. The replay tests each
condition with data only up to its own simulated date, so it always has
less data and less power than this does. If nothing clears the bar here,
nothing can clear it there.

What it cannot tell you: Sonnet does not sample this grammar uniformly --
it picks conditions that look sensible given the market context it is
shown. That could make it better than exhaustive search, or worse. This
measures the space, not the searcher.

Every condition in the sweep satisfies the project's necessary condition
(a news/macro event clause), because `ConditionSpec` refuses to build one
that doesn't -- combinations that fail that check are skipped, not
counted.

Run it:      python3 -m forecast.grammar_sweep
Resume it:   same command -- completed conditions are skipped automatically.
Analyse it:  python3 -m forecast.analyse_sweep

Takes roughly 70 minutes for the full 672 from cold, single-core, no
network. Results are written after EVERY condition, so it is safe to kill
at any time and restart later.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parent / "grammar_sweep.json"

COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LTCUSDT"]

# The news/macro clause is MANDATORY (enforced by ConditionSpec). A surprise
# threshold of +/-1.0 sigma is "a release that actually moved from the prior
# print", not merely "a release happened".
EVENT_TERMS = [
    ("is_macro_day", ">=", 1.0),
    ("cpi_surprise", ">=", 1.0), ("cpi_surprise", "<=", -1.0),
    ("rate_surprise", ">=", 1.0), ("rate_surprise", "<=", -1.0),
    ("jobless_claims_surprise", ">=", 1.0), ("jobless_claims_surprise", "<=", -1.0),
]

# Market-state terms at thresholds a human or an LLM would plausibly choose --
# oversold/overbought, band and channel extremes, funding crowding, volume and
# volatility spikes, clean trends, and recent large moves in both directions.
STATE_TERMS = [
    ("rsi_14d", "<=", 30.0), ("rsi_14d", "<=", 40.0), ("rsi_14d", ">=", 70.0),
    ("shock_zscore", ">=", 2.0), ("shock_zscore", ">=", 3.0),
    ("bollinger_pctb_20d", "<=", 0.1), ("bollinger_pctb_20d", ">=", 0.9),
    ("donchian_pct_20d", "<=", 0.1), ("donchian_pct_20d", ">=", 0.9),
    ("funding_zscore_30d", "<=", -2.0), ("funding_zscore_30d", ">=", 2.0),
    ("volume_zscore_30d", ">=", 2.0),
    ("close_return_5d", "<=", -0.10), ("close_return_5d", ">=", 0.10),
    ("efficiency_ratio_20d", ">=", 0.5),
    ("atr_pct_14d", ">=", 0.05),
]

# 0 = the market state holds on the news day itself; K = it occurred at some
# point in the K days BEFORE the news -- the ordered "crash, THEN news"
# hypothesis this project exists to test.
WITHIN_DAYS = [0, 3, 7]
DIRECTIONS = ["long", "short"]


def build_specs():
    from llm_pipeline.novel_condition_tester import Clause, ConditionSpec

    specs = []
    for ei, eo, et in EVENT_TERMS:
        for si, so, st in STATE_TERMS:
            for w in WITHIN_DAYS:
                for d in DIRECTIONS:
                    label = f"sweep_{ei}_{eo}_{et}__{si}_{so}_{st}__w{w}_{d}"
                    try:
                        specs.append(ConditionSpec(
                            label=label,
                            clauses=(Clause(indicator=ei, op=eo, threshold=et, within_days=0),
                                     Clause(indicator=si, op=so, threshold=st, within_days=w)),
                            direction=d))
                    except ValueError:
                        # Fails the necessary-condition check -- not a sweep entry.
                        continue
    return specs


def load_done() -> dict:
    if not RESULTS_PATH.exists():
        return {}
    return {r["label"]: r for r in json.loads(RESULTS_PATH.read_text())}


def main() -> None:
    from llm_pipeline.novel_condition_tester import test_novel_condition

    specs = build_specs()
    done = load_done()
    todo = [s for s in specs if s.label not in done]
    print(f"{len(specs)} conditions in the grammar sweep "
          f"({len(done)} already done, {len(todo)} to run)", flush=True)
    if not todo:
        print("Nothing to do -- run `python3 -m forecast.analyse_sweep`.", flush=True)
        return

    t0 = time.time()
    for i, spec in enumerate(todo):
        try:
            r = test_novel_condition(spec, COINS)
            pat = r.get("pattern_significance") or {}
            done[spec.label] = {
                "label": spec.label, "direction": spec.direction, "status": r.get("status"),
                "n": r.get("n"), "n_raw_triggers": r.get("n_raw_triggers"), "p": pat.get("p_value"), "excess": pat.get("excess_return"),
                "mfe_mae": pat.get("mfe_mae_ratio"), "pat_status": pat.get("status"),
                "max_coin_share": (r.get("coin_concentration") or {}).get("max_group_share"),
                "max_year_share": (r.get("year_concentration") or {}).get("max_group_share"),
                "coin_concentrated": (r.get("coin_concentration") or {}).get("concentrated"),
                "year_concentrated": (r.get("year_concentration") or {}).get("concentrated"),
            }
        except Exception as e:
            # One malformed condition must not cost the whole sweep -- recorded
            # as an error so it is visible rather than silently missing.
            done[spec.label] = {"label": spec.label, "status": "error", "err": str(e)[:120]}

        # Written after EVERY condition: this run is expected to be interrupted.
        RESULTS_PATH.write_text(json.dumps(list(done.values()), indent=1))

        if (i + 1) % 25 == 0:
            el = time.time() - t0
            rate = el / (i + 1)
            print(f"  {len(done)}/{len(specs)}  ({el/60:.0f}m elapsed, {rate:.1f}s/cond, "
                  f"~{rate*(len(todo)-i-1)/60:.0f}m left)", flush=True)

    print(f"DONE -- {len(done)}/{len(specs)}. "
          f"Now run: python3 -m forecast.analyse_sweep", flush=True)


if __name__ == "__main__":
    sys.exit(main())
