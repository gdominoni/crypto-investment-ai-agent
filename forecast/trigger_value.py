"""Do this project's two triggers actually select moments worth paying for?

Sonnet is consulted on two kinds of day: a real macro release, and a
volatility-shock transition. Together they account for essentially the whole
judgment bill -- roughly 1,200 calls over a full replay. Nothing had ever
checked whether either one picks days that differ from ordinary days.

The question is deliberately narrow. A trigger's job is to choose WHEN to spend
a call. It earns its cost if the days it fires on are followed by more market
movement than the days it skips -- if they are not, the calls are being spent on
a sample of ordinary days at $0.0153 each.

Measured per coin and pooled: absolute forward return over H days, divided by a
trailing 180-day standard deviation so a 2020 move and a 2024 move are on the
same scale, with the scaling `.shift(1)`-ed so it uses only past volatility.
Compared against every day the trigger did NOT fire, one-sided Mann-Whitney
(trigger days move MORE) -- the distributions are heavy-tailed and nowhere near
normal, so a t-test would be the wrong instrument.

Swept over horizons because a single horizon choice could drive the result on
its own -- the same defect already found and fixed in the pipeline's own
horizon selection.

WHAT THIS DOES NOT SETTLE, stated because the result is easy to over-read. This
measures UNCONDITIONAL movement: does anything happen on these days, on average.
This project's thesis is conditional -- macro event COMBINED WITH a market state
-- and a release could matter conditionally without lifting the average move.
The finding therefore bears on when it is worth ASKING, not on whether macro
terms belong in conditions. Nothing here is a market prediction.

CANDIDATE TRIGGERS are measured alongside the two in use, with the same
instrument, so "shock is the only one that works" is a measured claim rather
than a conclusion by exhaustion. Whatever a candidate scores, it is scored the
same way as the incumbent.

One honest caveat on the winners. Every trigger that selects here fires on a day
where volatility ALREADY happened, and volatility is strongly autocorrelated --
so a good part of what these triggers "predict" is clustering, a long-known
property of the series rather than a discovery. That does not disqualify them:
the trigger's job is to spend the budget on days where something is happening
and worth explaining. It does mean the trigger itself finds nothing; it only
decides where to look.

A SECOND, DIFFERENT QUESTION -- and the one that matters more. Everything above
asks whether a trigger precedes a bigger MOVE. This project is looking for the
causes of a TREND, and a shock is not a trend: it is a violent move followed
mostly by churn. `_trend_precursors` therefore asks whether a trigger precedes a
DEFINED TREND -- a move of at least one standard deviation whose path is also
directional, scored by the forward efficiency ratio (|net change| divided by the
sum of the absolute daily steps: 0 is chaos, 1 is a straight line).

The two questions give opposite answers for the incumbent trigger, which is why
asking only the first one was not enough.

Run:  python3 -m forecast.trigger_value
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_PATH = Path(__file__).resolve().parent / "trigger_value.json"

START = pd.Timestamp("2019-01-01")   # past the trailing windows' warm-up
END = pd.Timestamp("2025-12-31")
HORIZONS = (1, 3, 7, 14)
VOL_WINDOW = 180


def _scaled_move(df: pd.DataFrame, horizon: int, index) -> pd.Series:
    """|forward return| in units of the coin's own recent volatility.

    Scaling matters more than it looks: crypto's unconditional volatility fell
    by roughly half over this window, so an unscaled comparison would mostly
    measure WHEN each trigger fires in calendar time rather than what follows
    it. `.shift(1)` keeps the scaler backward-looking."""
    r = df["close"].shift(-horizon) / df["close"] - 1.0
    sd = r.rolling(VOL_WINDOW, min_periods=VOL_WINDOW // 3).std().shift(1)
    return (r / sd).abs().loc[index]


def _candidate_triggers(df: pd.DataFrame) -> dict:
    """Trigger rules that could replace or supplement the two in use. Each is a
    boolean series on the coin's daily bars, all strictly backward-looking."""
    from candidates.methodology import shock_zscore_series

    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    out = {}

    z = shock_zscore_series(df)
    out["shock z>=2 (transition)"] = (z >= 2) & (z.shift(1) < 2)

    # Daily range against its OWN trailing 95th percentile, not a fixed
    # percentage: crypto's volatility fell by half over this window, so any
    # absolute threshold would fire constantly early and never late.
    rng = (high - low) / close
    out["daily range > own p95"] = rng > rng.rolling(VOL_WINDOW, min_periods=60).quantile(0.95).shift(1)

    vz = (vol - vol.rolling(30).mean()) / vol.rolling(30).std()
    out["volume z>=2"] = (vz >= 2) & (vz.shift(1) < 2)

    # TODO 2b costed a MACD-reversal trigger at +72% but never measured whether
    # it selects anything. Both the raw crossing and a confirmed version.
    ema = lambda n: close.ewm(span=n, adjust=False).mean()
    hist = (ema(12) - ema(26))
    hist = hist - hist.ewm(span=9, adjust=False).mean()
    cross = (np.sign(hist) != np.sign(hist.shift(1))) & hist.notna() & hist.shift(1).notna()
    out["MACD crossing"] = cross

    k = 100 * (close - low.rolling(14).min()) / (high.rolling(14).max() - low.rolling(14).min())
    out["stochastic exits <20"] = (k > 20) & (k.shift(1) <= 20)

    r5 = close / close.shift(5) - 1.0
    out["5d fall <= -10%"] = (r5 <= -0.10) & (r5.shift(1) > -0.10)
    out["5d rise >= +10%"] = (r5 >= 0.10) & (r5.shift(1) < 0.10)
    return out


def _forward_efficiency(close: pd.Series, horizon: int) -> pd.Series:
    """How DIRECTIONAL the next `horizon` days are: |net change| over the sum of
    the absolute daily steps. 1.0 is a straight line, 0 is a round trip.

    This is the same instrument as `definitions.trend_efficiency_ratio`, pointed
    forwards instead of backwards. It exists because forward RETURN alone cannot
    tell a trend from a spike: +5% at 7 days scores identically whether it came
    from a steady climb, from +15% on day one bleeding back, or from oscillating
    +/-20% and landing at +5%. Those are different phenomena and only one of them
    is what this project is looking for."""
    net = (close.shift(-horizon) - close).abs()
    steps = close.diff().abs()
    total = steps[::-1].rolling(horizon).sum()[::-1].shift(-1)
    return net / total


def _is_trend(df: pd.DataFrame, horizon: int, min_efficiency: float = 0.5) -> pd.Series:
    """A trend is a move that is both LARGE and DIRECTIONAL. Either alone is not
    one: a big chaotic swing is not a trend, and a tidy 0.2% drift is not either."""
    r = df["close"].shift(-horizon) / df["close"] - 1.0
    sd = r.rolling(VOL_WINDOW, min_periods=VOL_WINDOW // 3).std().shift(1)
    return ((r / sd).abs() >= 1.0) & (_forward_efficiency(df["close"], horizon) >= min_efficiency)


def _trend_precursors(df: pd.DataFrame) -> dict:
    """Triggers that might precede a trend, as opposed to following a move.

    The compression family is here because of what the incumbent trigger turned
    out NOT to do. Volatility compression preceding expansion is a long-known
    property of financial series (the Bollinger squeeze), so a positive result
    here is a confirmation, not a discovery -- but it is the right shape for this
    system: it says a directional move is brewing WITHOUT saying which way, which
    leaves the direction to be explained by the macro context and market state.
    That is exactly the question the pipeline exists to ask."""
    from candidates.definitions import trend_efficiency_ratio
    from candidates.methodology import shock_zscore_series

    close, high, low = df["close"], df["high"], df["low"]
    rng = (high - low) / close
    bb_width = (close.rolling(20).std() * 4) / close.rolling(20).mean()
    rank = lambda x: x.rolling(VOL_WINDOW, min_periods=60).rank(pct=True)
    eff20 = trend_efficiency_ratio(close, 20)
    z = shock_zscore_series(df)
    return {
        "shock z>=2 (incumbent)": (z >= 2) & (z.shift(1) < 2),
        "range compressed p<0.10": rank(rng) <= 0.10,
        "bollinger tight p<0.10": rank(bb_width) <= 0.10,
        "bollinger tight 5d running": (rank(bb_width) <= 0.20).rolling(5).min().fillna(0).astype(bool),
        "choppy market eff20<0.2": eff20 < 0.2,
        "compression + chop": (rank(bb_width) <= 0.25) & (eff20 < 0.25),
    }


def main() -> int:
    from scipy.stats import mannwhitneyu

    from candidates.data_loading import load_daily
    from candidates.macro_vintage import MACRO_SERIES, release_dates
    from candidates.methodology import shock_zscore_series
    from candidates.run_battery import COINS

    per_series = {k: set(release_dates(k, START, END, new_periods_only=True))
                  for k in MACRO_SERIES}
    any_release = set().union(*per_series.values())

    results = {}
    print("Does each trigger select days that move more than the days it skips?")
    print("Absolute forward return / trailing 180d sd. One-sided Mann-Whitney,")
    print("median across the 7 coins.\n")
    print(f"{'trigger':<28}{'days':>7}" + "".join(f"{f'p @{h}d':>10}" for h in HORIZONS))
    print("-" * 76)

    triggers: list[tuple[str, object]] = [
        ("macro release (any)", any_release),
        *[(f"  {lbl}", per_series[k]) for k, lbl in MACRO_SERIES.items()],
        ("volatility shock z>=2", "shock"),
    ]

    for name, spec in triggers:
        pvals = {h: [] for h in HORIZONS}
        medians = {h: [] for h in HORIZONS}
        n_days = 0
        for coin in COINS:
            df = load_daily(coin)
            idx = df.index[(df.index >= START) & (df.index <= END)]
            if spec == "shock":
                z = shock_zscore_series(df).loc[idx]
                # The transition INTO shock, matching replay/engine.py rather than
                # every day a multi-day episode persists.
                mask = (z >= 2.0) & (z.shift(1) < 2.0)
            else:
                mask = pd.Series(idx.isin(spec), index=idx)
            mask = mask.reindex(idx).fillna(False)
            n_days += int(mask.sum())
            for h in HORIZONS:
                move = _scaled_move(df, h, idx)
                fired, skipped = move[mask].dropna(), move[~mask].dropna()
                if len(fired) < 20:
                    continue
                pvals[h].append(mannwhitneyu(fired, skipped, alternative="greater").pvalue)
                medians[h].append(float(fired.median() - skipped.median()))
        cells = "".join(f"{np.median(pvals[h]):>10.3f}" if pvals[h] else f"{'--':>10}"
                        for h in HORIZONS)
        print(f"{name:<28}{n_days:>7}{cells}")
        results[name.strip()] = {
            "days": n_days,
            "p": {h: (float(np.median(pvals[h])) if pvals[h] else None) for h in HORIZONS},
            "median_excess_move": {h: (float(np.median(medians[h])) if medians[h] else None)
                                    for h in HORIZONS},
        }

    # If the macro trigger were dropped, how much macro context would a shock day
    # still carry? The prompt already includes releases from the last 10 simulated
    # days, so a sequenced "shock following a CPI print" stays expressible exactly
    # to the extent that shocks are preceded by releases.
    shock_days = set()
    for coin in COINS:
        df = load_daily(coin)
        idx = df.index[(df.index >= START) & (df.index <= END)]
        z = shock_zscore_series(df).loc[idx]
        shock_days |= set(idx[(z >= 2.0) & (z.shift(1) < 2.0)])
    rel = np.array(sorted(any_release))
    print("\nIf the macro trigger were removed, macro context surviving on shock days:")
    coverage = {}
    for w in (3, 7, 10):
        n = sum(1 for d in sorted(shock_days)
                if (rel <= d).any() and (d - rel[rel <= d][-1]).days <= w)
        coverage[w] = n / len(shock_days) if shock_days else 0.0
        print(f"  a macro release within the previous {w:>2} days: "
              f"{n}/{len(shock_days)} ({coverage[w]:.0%})")
    results["_shock_macro_coverage"] = coverage

    # Candidate triggers, same instrument.
    print(f"\n{'candidate trigger':<28}{'days':>7}" + "".join(f"{f'p @{h}d':>10}" for h in HORIZONS))
    print("-" * 76)
    cand = {}
    for coin in COINS:
        df = load_daily(coin)
        idx = df.index[(df.index >= START) & (df.index <= END)]
        for name, mask in _candidate_triggers(df).items():
            mask = mask.reindex(idx).fillna(False)
            e = cand.setdefault(name, {"n": 0, **{h: [] for h in HORIZONS}})
            e["n"] += int(mask.sum())
            for h in HORIZONS:
                move = _scaled_move(df, h, idx)
                a, b = move[mask].dropna(), move[~mask].dropna()
                if len(a) >= 20:
                    e[h].append(mannwhitneyu(a, b, alternative="greater").pvalue)
    for name in sorted(cand, key=lambda n: np.median(cand[n][1]) if cand[n][1] else 1.0):
        e = cand[name]
        cells = "".join(f"{np.median(e[h]):>10.4f}" if e[h] else f"{'--':>10}" for h in HORIZONS)
        print(f"{name:<28}{e['n']:>7}{cells}")
        results[name] = {"days": e["n"],
                         "p": {h: (float(np.median(e[h])) if e[h] else None) for h in HORIZONS}}

    # The question that decides whether a winner is worth ADDING rather than
    # merely being correlated with the shock trigger already in place: on the
    # days it fires and shock does not, does anything still happen?
    print("\nIncremental over the shock trigger -- days it catches that shock misses:")
    print(f"{'':<30}{'days':>7}" + "".join(f"{f'p @{h}d':>10}" for h in (1, 3, 7)))
    inc = {}
    for coin in COINS:
        df = load_daily(coin)
        idx = df.index[(df.index >= START) & (df.index <= END)]
        t = {k: v.reindex(idx).fillna(False) for k, v in _candidate_triggers(df).items()}
        sh = t["shock z>=2 (transition)"]
        quiet = ~(sh | t["daily range > own p95"] | t["volume z>=2"])
        for name in ("daily range > own p95", "volume z>=2"):
            extra = t[name] & ~sh
            e = inc.setdefault(name, {"n": 0, **{h: [] for h in (1, 3, 7)}})
            e["n"] += int(extra.sum())
            for h in (1, 3, 7):
                move = _scaled_move(df, h, idx)
                a, b = move[extra].dropna(), move[quiet].dropna()
                if len(a) >= 20:
                    e[h].append(mannwhitneyu(a, b, alternative="greater").pvalue)
    for name, e in inc.items():
        cells = "".join(f"{np.median(e[h]):>10.4f}" if e[h] else f"{'--':>10}" for h in (1, 3, 7))
        print(f"{name + ' minus shock':<30}{e['n']:>7}{cells}")
        results[f"{name} minus shock"] = {"days": e["n"],
                                           "p": {h: float(np.median(e[h])) for h in (1, 3, 7) if e[h]}}

    # The question that actually matters: does the trigger precede a TREND?
    for horizon in (14, 21):
        print(f"\nTrend precursors -- trend = |move| >= 1sd AND forward efficiency >= 0.5, "
              f"{horizon}d")
        print(f"{'trigger':<30}{'days':>7}{'trend after':>13}{'base':>8}{'p':>9}")
        print("-" * 68)
        agg = {}
        for coin in COINS:
            df = load_daily(coin)
            idx = df.index[(df.index >= START) & (df.index <= END)]
            trend = _is_trend(df, horizon).reindex(idx).fillna(False)
            for name, mask in _trend_precursors(df).items():
                mask = mask.reindex(idx).fillna(False)
                e = agg.setdefault(name, {"n": 0, "a": [], "b": []})
                e["n"] += int(mask.sum())
                e["a"].append(trend[mask]); e["b"].append(trend[~mask])
        for name, e in agg.items():
            a, b = pd.concat(e["a"]), pd.concat(e["b"])
            if len(a) < 50:
                continue
            pv = mannwhitneyu(a.astype(float), b.astype(float), alternative="greater").pvalue
            print(f"{name:<30}{e['n']:>7}{a.mean():>12.1%}{b.mean():>8.1%}{pv:>9.4f}")
            results[f"{name} -> trend @{horizon}d"] = {
                "days": e["n"], "trend_rate": float(a.mean()),
                "base_rate": float(b.mean()), "p": float(pv)}

    RESULTS_PATH.write_text(json.dumps(results, indent=1, default=str))
    print(f"\nWritten to {RESULTS_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
