"""How good would a news-sentiment feed have to BE for this system to
detect it? -- the number that decides whether a GDELT backfill is worth
building.

This supersedes the binary-event control in `forecast/control_sweep.py`,
which had three defects, all raised by the project's director:

  1. It planted a RARE BINARY event (440 events, 2% of coin-days). Real
     sentiment is a CONTINUOUS score present on every row, mostly low,
     with a right tail -- and the analyst chooses where to threshold.
     Event rate is therefore a free parameter, not a fixed 2%, and a
     lower threshold buys more events, which is exactly what this
     pipeline is short of. The binary design tested only the weakest,
     smallest-sample corner of the space.
  2. It dropped the REAL macro indicators from the state grammar
     entirely, so "sentiment AND a real CPI/FOMC surprise" -- the actual
     thesis -- was never tested.
  3. Its planted signal was effectively an oracle. "A perfect feed is
     detectable" is a much weaker claim than "a feed of realistic
     quality is detectable".

The model here. Sentiment on day t is

    score_t = rho * z(forward_return_t) + sqrt(1 - rho^2) * noise_t

so `rho` IS the correlation between the sentiment score and the future
return -- one interpretable number for "how informative is this feed".
The score exists on every day and is standard-normal by construction, so
thresholding at 1.0 / 1.5 / 2.0 sigma yields roughly 16% / 7% / 2% of
days, spanning the realistic range of "how selective is our sentiment
trigger".

rho = 0.30 is a spectacular feed. Published work on news-sentiment and
next-week equity returns typically finds single-digit correlations, so a
REAL feed plausibly lands near rho = 0.05. That is why the sweep brackets
0.04 and 0.08: the interesting question is not whether an oracle works,
it is whether anything achievable does.

rho = 0.0 is the false-positive floor and must stay empty.

Nothing here is evidence about markets. The signal is synthetic and uses
future information by construction.

Run:  python3 -m forecast.sentiment_power
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_PATH = Path(__file__).resolve().parent / "sentiment_power.json"

COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LTCUSDT"]

# How informative the feed is: the correlation between its score and the
# realised forward return. The whole point of the experiment.
RHOS = [0.30, 0.15, 0.08, 0.04, 0.00]

# Where the analyst puts the trigger, in sigmas of the score. Controls the
# event rate (~16% / 7% / 2% of days) and therefore the sample size.
SENTIMENT_THRESHOLDS = [1.0, 1.5, 2.0]

# Market-state AND real macro terms -- fixing defect 2. The macro clauses
# make "sentiment AND a real macro surprise" expressible, which is the
# project's actual thesis.
STATE_TERMS = [
    (None, None, None),                       # sentiment alone, for reference
    ("is_macro_day", ">=", 1.0),
    ("cpi_surprise", ">=", 1.0),
    ("rate_surprise", ">=", 1.0),
    ("jobless_claims_surprise", ">=", 1.0),
    ("rsi_14d", "<=", 40.0),
    ("shock_zscore", ">=", 2.0),
    ("bollinger_pctb_20d", ">=", 0.9),
    ("donchian_pct_20d", ">=", 0.9),
    ("funding_zscore_30d", "<=", -2.0),
]
WITHIN_DAYS = [0, 7]
HORIZON = 7


def make_sentiment(rho: float, seed: int = 11):
    """A continuous sentiment score on EVERY day, correlated `rho` with
    that day's forward return. Standard-normal by construction so a
    threshold in sigmas means the same thing at every rho."""
    def indicator(df, funding, scale=1):
        fwd = df["close"].shift(-HORIZON) / df["close"] - 1.0
        z = ((fwd - fwd.mean()) / fwd.std()).fillna(0.0)
        rng = np.random.default_rng(seed + len(df))  # per-coin, deterministic
        noise = pd.Series(rng.standard_normal(len(df)), index=df.index)
        return rho * z + np.sqrt(max(1.0 - rho * rho, 0.0)) * noise
    return indicator


def register(rhos) -> None:
    import llm_pipeline.novel_condition_tester as N
    for rho in rhos:
        key = f"sent_rho{int(rho*100):03d}"
        N.SUPPORTED_INDICATORS[key] = make_sentiment(rho)
        # BOTH sets: NEWS_EVENT_INDICATORS gates what may be proposed,
        # EVENT_INDICATORS is what gets stripped to build the control group.
        N.NEWS_EVENT_INDICATORS = frozenset(set(N.NEWS_EVENT_INDICATORS) | {key})
        N.EVENT_INDICATORS = frozenset(set(N.EVENT_INDICATORS) | {key})
        N.DAILY_NATIVE_INDICATORS = frozenset(set(N.DAILY_NATIVE_INDICATORS) | {key})
        N.INDICATOR_PLAIN_NAMES[key] = f"synthetic sentiment rho={rho} (TEST ONLY)"


def build_specs():
    from llm_pipeline.novel_condition_tester import Clause, ConditionSpec
    specs = []
    for rho in RHOS:
        key = f"sent_rho{int(rho*100):03d}"
        for thr in SENTIMENT_THRESHOLDS:
            ev = Clause(indicator=key, op=">=", threshold=thr, within_days=0)
            for si, so, st in STATE_TERMS:
                if si is None:
                    specs.append(ConditionSpec(
                        label=f"sp_rho{rho}_t{thr}__alone__long", clauses=(ev,), direction="long"))
                    continue
                for w in WITHIN_DAYS:
                    specs.append(ConditionSpec(
                        label=f"sp_rho{rho}_t{thr}__{si}_{so}_{st}__w{w}_long",
                        clauses=(ev, Clause(indicator=si, op=so, threshold=st, within_days=w)),
                        direction="long"))
    return specs


def load_done() -> dict:
    if not RESULTS_PATH.exists():
        return {}
    return {r["label"]: r for r in json.loads(RESULTS_PATH.read_text())}


def main() -> None:
    register(RHOS)
    from llm_pipeline.novel_condition_tester import test_novel_condition

    specs = build_specs()
    done = load_done()
    todo = [s for s in specs if s.label not in done]
    print(f"{len(specs)} conditions: {len(RHOS)} feed qualities x "
          f"{len(SENTIMENT_THRESHOLDS)} trigger thresholds x {len(STATE_TERMS)} state terms")
    print(f"{len(done)} done, {len(todo)} to run\n", flush=True)

    t0 = time.time()
    for i, spec in enumerate(todo):
        rho = float(spec.label.split("_rho")[1].split("_")[0])
        thr = float(spec.label.split("_t")[1].split("__")[0])
        try:
            r = test_novel_condition(spec, COINS)
            pat = r.get("pattern_significance") or {}
            done[spec.label] = {
                "label": spec.label, "rho": rho, "threshold": thr,
                "status": r.get("status"), "n": r.get("n"),
                "n_raw": r.get("n_raw_triggers"), "p": pat.get("p_value"),
                "excess": pat.get("excess_return"), "mfe_mae": pat.get("mfe_mae_ratio"),
                "significant": pat.get("significant"), "pat_status": pat.get("status"),
            }
        except Exception as e:
            done[spec.label] = {"label": spec.label, "rho": rho, "threshold": thr,
                                "status": "error", "err": str(e)[:120]}
        RESULTS_PATH.write_text(json.dumps(list(done.values()), indent=1))
        if (i + 1) % 40 == 0:
            el = time.time() - t0
            rate = el / (i + 1)
            print(f"  {len(done)}/{len(specs)}  ({el/60:.0f}m, {rate:.1f}s/cond, "
                  f"~{rate*(len(todo)-i-1)/60:.0f}m left)", flush=True)
    print("DONE -- run: python3 -m forecast.analyse_sentiment_power", flush=True)


if __name__ == "__main__":
    sys.exit(main())
