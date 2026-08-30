"""POSITIVE CONTROL: can this pipeline detect a signal that is definitely there?

This is a TEST OF THE CODE, not a finding about markets. Nothing here is
evidence about crypto, and no result from it may ever be reported as one.
It exists to answer a single question: when a real, known effect is
present in the data, does `pattern_significance` / `classify_status`
actually flag it? A detector that never fires is indistinguishable from a
detector that is broken, and after a 672-condition sweep found nothing,
that distinction is worth establishing directly.

Three arms, deliberately:

  PLANTED  -- a synthetic "sentiment" event placed, by construction, on
              days that ARE followed by strong positive returns. This uses
              future information on purpose: it is lookahead by design, so
              the effect is guaranteed real and large. The pipeline MUST
              detect this. If it does not, the code is broken.
              Run at several strengths, so we learn not just "does it
              fire" but "how big must an effect be before it fires".

  REAL     -- genuine, dated crypto news events (COVID crash, China ban,
              Terra/Luna, FTX, the ETF approval, halvings...). Whether
              these are detectable is an open empirical question, which is
              exactly why they cannot serve as the control.

  RANDOM   -- events on uniformly random days. The pipeline MUST NOT
              detect these. This is the specificity arm: a detector that
              fires on everything is as useless as one that fires on
              nothing.

Reading the result:
  PLANTED detected + RANDOM not detected -> the code works, and the null
      result from the real grammar sweep is a fact about the market.
  PLANTED not detected -> the code (or its power) is the problem, and
      every null this project has produced needs re-examining.
  RANDOM detected -> the false-positive control is broken.

Run:  python3 -m forecast.positive_control
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LTCUSDT"]

# Real, dated crypto-market news events. Sources: Bitcoin crash-history
# summaries (see the conversation log / bitcoinfoundation.org, paybis.com,
# thestreet.com). Dates are the event's own public date, not a reaction window.
REAL_NEWS_DATES = [
    "2018-01-16",  # January 2018 selloff / South Korea regulation fears
    "2018-11-14",  # Bitcoin Cash hash war, break below $6k
    "2019-06-26",  # 2019 local top ~$13.8k
    "2020-03-12",  # COVID "Black Thursday", BTC ~$8k -> <$4k in a day
    "2020-05-11",  # third halving
    "2020-12-16",  # BTC breaks its 2017 all-time high
    "2021-02-08",  # Tesla discloses $1.5bn BTC purchase
    "2021-04-14",  # Coinbase direct listing (cycle local top)
    "2021-05-19",  # China crackdown crash, ~$1tn wiped that week
    "2021-09-24",  # China blanket ban on crypto transactions
    "2021-11-10",  # cycle peak ~$69k
    "2022-01-21",  # break below $35k
    "2022-05-09",  # Terra/UST depeg begins
    "2022-05-12",  # LUNA collapse completes
    "2022-06-13",  # Celsius freezes withdrawals
    "2022-09-15",  # Ethereum Merge
    "2022-11-08",  # Binance/FTX letter of intent, FTX implosion begins
    "2022-11-11",  # FTX files for bankruptcy
    "2023-03-10",  # SVB collapse, USDC depeg
    "2023-06-05",  # SEC sues Binance
    "2024-01-10",  # US spot bitcoin ETFs approved
    "2024-04-19",  # fourth halving
    "2024-11-06",  # US election result, post-election rally
    "2025-02-03",  # tariff-driven risk-off selloff
]


def _forward_return(df: pd.DataFrame, horizon: int = 7) -> pd.Series:
    return df["close"].shift(-horizon) / df["close"] - 1.0


def make_planted(top_frac: float, event_frac: float = 0.02, seed: int = 0):
    """A 'sentiment' event that fires only on days whose FORWARD return is
    in the top `top_frac` of that coin's own distribution.

    This is deliberate lookahead. That is the entire point: it manufactures
    an effect of known sign and known size so the detector can be tested
    against a ground truth. `top_frac=0.2` is a strong planted signal;
    0.5 is 'better than a coin flip and nothing more'.

    `event_frac` keeps the event RATE realistic (~2% of days) so the test
    exercises the same small-sample regime real conditions live in --
    planting on 20% of all days would prove only that the pipeline can
    detect an effect it will never be given.
    """
    def indicator(df, funding, scale=1, symbol=None):
        fwd = _forward_return(df)
        thresh = fwd.quantile(1.0 - top_frac)
        eligible = (fwd >= thresh).fillna(False)
        rng = np.random.default_rng(seed)
        keep = pd.Series(rng.random(len(df)), index=df.index) < (event_frac / top_frac)
        return (eligible & keep).astype(float)
    return indicator


def make_random(event_frac: float = 0.02, seed: int = 7):
    def indicator(df, funding, scale=1, symbol=None):
        rng = np.random.default_rng(seed)
        return (pd.Series(rng.random(len(df)), index=df.index) < event_frac).astype(float)
    return indicator


def make_real_news():
    dates = pd.DatetimeIndex([pd.Timestamp(d) for d in REAL_NEWS_DATES])

    def indicator(df, funding, scale=1, symbol=None):
        idx = df.index.floor("D") if scale > 1 else df.index
        return pd.Series(idx.normalize().isin(dates), index=df.index).astype(float)
    return indicator


def run_arm(name: str, indicator, state_clause=None) -> dict:
    """Install the synthetic indicator into the REAL registry, then run the
    REAL pipeline. Production files are never edited -- the registry is
    mutated in this process only, and restored afterwards."""
    import llm_pipeline.novel_condition_tester as N
    from llm_pipeline.novel_condition_tester import Clause, ConditionSpec, test_novel_condition

    key = "synthetic_sentiment"
    N.SUPPORTED_INDICATORS[key] = indicator
    # BOTH sets, and they are not the same thing. NEWS_EVENT_INDICATORS
    # enforces the necessary condition (what may be PROPOSED);
    # EVENT_INDICATORS is what `reduced_clauses` strips to build the CONTROL
    # group. Registering in only the first one leaves the indicator in the
    # control as well as the treatment, so the nested test compares the
    # condition against itself-minus-itself, gets an empty control, and
    # returns `insufficient_data` -- which surfaces as status "watch",
    # indistinguishable from a real inconclusive result. That is exactly the
    # failure this control caught on its first run.
    N.NEWS_EVENT_INDICATORS = frozenset(set(N.NEWS_EVENT_INDICATORS) | {key})
    N.EVENT_INDICATORS = frozenset(set(N.EVENT_INDICATORS) | {key})
    N.DAILY_NATIVE_INDICATORS = frozenset(set(N.DAILY_NATIVE_INDICATORS) | {key})
    N.INDICATOR_PLAIN_NAMES[key] = "synthetic sentiment (TEST ONLY)"
    try:
        clauses = [Clause(indicator=key, op=">=", threshold=1.0, within_days=0)]
        if state_clause:
            clauses.append(state_clause)
        spec = ConditionSpec(label=f"control_{name}", clauses=tuple(clauses), direction="long")
        r = test_novel_condition(spec, COINS)
        pat = r.get("pattern_significance") or {}
        return {"arm": name, "status": r.get("status"), "n": r.get("n"),
                "n_raw": r.get("n_raw_triggers"), "p": pat.get("p_value"),
                "excess": pat.get("excess_return"), "mfe_mae": pat.get("mfe_mae_ratio"),
                "significant": pat.get("significant"), "pat_status": pat.get("status"),
                "baseline_kind": pat.get("baseline_kind")}
    finally:
        N.SUPPORTED_INDICATORS.pop(key, None)


def _row(r: dict) -> str:
    def f(v, spec, mult=1.0):
        return format(v * mult, spec) if isinstance(v, (int, float)) and v == v else "--"
    return (f"{r['arm']:<34}{str(r['status']):<18}{str(r['n']):>6}{f(r['p'],'.4f'):>9}"
            f"{f(r['excess'],'+.2f',100):>10}{f(r['mfe_mae'],'.2f'):>9}   {r['significant']}")


def main() -> None:
    print(__doc__.split("Run:")[0].strip()[:0] or "", end="")
    print("=" * 96)
    print("POSITIVE CONTROL -- a test of the CODE. Not evidence about markets.")
    print("The PLANTED arms use future information ON PURPOSE to manufacture a known effect.")
    print("=" * 96 + "\n")

    arms = [
        ("PLANTED top-20% (very strong)", make_planted(0.20)),
        ("PLANTED top-35% (strong)", make_planted(0.35)),
        ("PLANTED top-50% (marginal)", make_planted(0.50)),
        ("REAL news dates", make_real_news()),
        ("RANDOM days (must NOT fire)", make_random()),
    ]
    print(f"{'arm':<34}{'status':<18}{'n':>6}{'p':>9}{'excess%':>10}{'MFE/MAE':>9}   significant")
    print("-" * 96)
    results = []
    for name, ind in arms:
        try:
            r = run_arm(name, ind)
        except Exception as e:
            r = {"arm": name, "status": f"ERROR: {str(e)[:40]}", "n": None, "p": None,
                 "excess": None, "mfe_mae": None, "significant": None}
        results.append(r)
        print(_row(r), flush=True)

    print("\nSame arms, now ANDed with a market-state clause (exercises the nested/")
    print("incremental baseline -- 'does the sentiment add anything GIVEN the state?')")
    print("-" * 96)
    from llm_pipeline.novel_condition_tester import Clause
    state = Clause(indicator="rsi_14d", op="<=", threshold=50.0, within_days=0)
    for name, ind in arms:
        try:
            r = run_arm(name + " + RSI<50", ind, state_clause=state)
        except Exception as e:
            r = {"arm": name + " + RSI<50", "status": f"ERROR: {str(e)[:40]}", "n": None,
                 "p": None, "excess": None, "mfe_mae": None, "significant": None}
        results.append(r)
        print(_row(r), flush=True)

    print("\n" + "=" * 96)
    planted = [r for r in results if r["arm"].startswith("PLANTED")]
    random_arms = [r for r in results if r["arm"].startswith("RANDOM")]
    fired = [r for r in planted if r.get("significant")]
    rnd_fired = [r for r in random_arms if r.get("significant")]
    print(f"PLANTED arms flagged significant: {len(fired)}/{len(planted)}")
    print(f"RANDOM  arms flagged significant: {len(rnd_fired)}/{len(random_arms)}  (want 0)")
    if fired and not rnd_fired:
        print("\nVERDICT: the detector WORKS -- it fires on a real planted effect and stays")
        print("silent on noise. The null result from the real grammar sweep is therefore a")
        print("fact about the data, not a broken pipeline.")
    elif not fired:
        print("\nVERDICT: the detector did NOT fire on a signal that is definitely present.")
        print("The problem is the code or its statistical power, and every null result this")
        print("project has produced needs re-examining before it can be trusted.")
    if rnd_fired:
        print("\nWARNING: a RANDOM arm was flagged significant -- false-positive control failed.")


if __name__ == "__main__":
    sys.exit(main())
