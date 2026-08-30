"""Does market-relative testing recover a COIN-SPECIFIC signal that raw
testing misses? -- the XRP/SEC case, isolated.

`forecast/market_relative.py` found that market-relative testing made
things WORSE (10 accepted -> 7). That result is real but answers a
different question than intended: its planted signal fires on days when
the COIN's RAW forward return is high, and with a 0.54 cross-coin
correlation those are largely days when the whole market rises. It was a
market-timing signal, so subtracting the market removed the effect (17.29%
-> 4.66% excess) faster than it removed the noise (sigma 0.70x). Correct
behaviour, wrong test subject.

A genuine coin-specific event -- "SEC sues Ripple" -- moves ONE coin
against its peers. This plants exactly that: the signal fires on days
whose MARKET-RELATIVE forward return is high for that coin, and the same
conditions are then measured BOTH ways. The prediction is explicit and
falsifiable: raw testing should miss it (the coin-specific move is
drowned by market noise), market-relative testing should find it.

If that prediction fails, market-relative is not worth building.

RESULT, and a correction to how it was first attributed. Re-run once
`ConditionSpec.coins`/`outcome` shipped, so the test uses the declared
fields rather than a monkeypatch:

    condition                RAW p      REL p     both
    alone                    0.0140    0.0005   accepted
    rsi_14d<=40 w0           0.0415    0.0045   accepted
    rsi_14d<=40 w7           0.0480    0.0000   accepted
    is_macro_day w7          0.0130    0.0010   accepted
    -> accepted: 6/9 raw, 6/9 market-relative

The first version of this test reported 2/9 raw vs 6/9 market-relative and
credited the gap to market-relative measurement. That was only half right.
At the time, the coin-concentration check was still gating a declared
single-coin spec, and it was killing the raw arm. With that fixed, the two
contributions separate cleanly:

  * declaring the spec coin-scoped (which waives a coin-concentration check
    that is meaningless for a single-coin hypothesis) does most of the work:
    0/9 -> 6/9 acceptances.
  * market-relative measurement no longer changes the COUNT at this signal
    strength, but shrinks every p-value by roughly 5-10x (0.0140 -> 0.0005,
    0.0480 -> 0.0000). It buys margin, not new acceptances -- which is what
    matters for a weaker signal, and for surviving family-level FDR.

Both are worth having, and the honest split is not the one originally
reported.
"""
import sys

import numpy as np
import pandas as pd

from forecast.market_relative import COINS, _basket_forward_returns

H = 7
_BASKET = None


def _basket():
    """Lazy: loading seven coins' price history at import time makes merely
    importing this module slow, and the other forecast modules do not."""
    global _BASKET
    if _BASKET is None:
        _BASKET = _basket_forward_returns()
    return _BASKET


def make_planted_relative(top_frac=0.20, event_frac=0.02, seed=5, only: str | None = None):
    """Fires on days whose MARKET-RELATIVE forward return is in the top
    `top_frac` -- a coin outperforming its peers, not a rising tide.

    `only` restricts it to one coin, using the `symbol` parameter that now
    exists on every indicator. The first version of this test had to
    identify a coin by its price series LENGTH, because the signature
    carried no identity at all -- which is precisely the gap that made
    coin-attributed indicators unwritable and motivated the change."""
    def indicator(df, funding, scale=1, symbol=None):
        if only is not None and symbol != only:
            return pd.Series(0.0, index=df.index)
        fwd = df["close"].shift(-H) / df["close"] - 1.0
        rel = fwd - _basket()[H].reindex(df.index)
        thresh = rel.quantile(1.0 - top_frac)
        elig = (rel >= thresh).fillna(False)
        rng = np.random.default_rng(seed)
        keep = pd.Series(rng.random(len(df)), index=df.index) < (event_frac / top_frac)
        return (elig & keep).astype(float)
    return indicator


def run(relative_outcome: bool):
    """Exercises the SHIPPED code path, not a patch.

    This module originally predated `ConditionSpec.coins`/`outcome` and had to
    monkeypatch `_forward_return` to get a market-relative measurement, then
    reload the module to undo it. Both fields now exist, so the test declares
    what it wants the same way any real hypothesis would -- which means it is
    also a regression test for the feature, rather than a parallel
    implementation of it that could quietly diverge."""
    import llm_pipeline.novel_condition_tester as N
    from llm_pipeline.novel_condition_tester import Clause, ConditionSpec, test_novel_condition

    key = "sent_coinspecific"
    # Restricted to ONE coin: that is what makes this a coin-specific test
    # rather than a market-wide one. Without the restriction the signal
    # fires everywhere, becomes a market-timing signal, and raw vs
    # market-relative measurement stops differing at all.
    # event_frac is a fraction of ONE coin's days here, so the same 2% used by
    # the seven-coin controls yields roughly a seventh of the absolute events
    # and drops most conditions under the sample gate before anything can be
    # measured. 10% restores the event count the documented result was obtained
    # at -- matching the original experiment's conditions, not searching for a
    # favourable one.
    N.SUPPORTED_INDICATORS[key] = make_planted_relative(only="XRPUSDT", event_frac=0.10)
    N.NEWS_EVENT_INDICATORS = frozenset(set(N.NEWS_EVENT_INDICATORS) | {key})
    N.EVENT_INDICATORS = frozenset(set(N.EVENT_INDICATORS) | {key})
    N.DAILY_NATIVE_INDICATORS = frozenset(set(N.DAILY_NATIVE_INDICATORS) | {key})
    N.INDICATOR_PLAIN_NAMES[key] = "coin-specific sentiment (TEST)"

    ev = Clause(indicator=key, op=">=", threshold=1.0, within_days=0)
    states = [None, ("rsi_14d", "<=", 40.0), ("shock_zscore", ">=", 2.0),
              ("is_macro_day", ">=", 1.0), ("donchian_pct_20d", ">=", 0.9)]
    out = []
    for st in states:
        for w in ([0] if st is None else [0, 7]):
            clauses = (ev,) if st is None else (ev, Clause(indicator=st[0], op=st[1], threshold=st[2], within_days=w))
            lbl = "alone" if st is None else f"{st[0]}{st[1]}{st[2]}_w{w}"
            try:
                r = test_novel_condition(ConditionSpec(
                    label=f"cs_{lbl}", clauses=clauses, direction="long",
                    coins=("XRPUSDT",),
                    outcome="market_relative" if relative_outcome else "raw"), COINS)
                pat = r.get("pattern_significance") or {}
                out.append({"cond": lbl, "status": r.get("status"), "n": r.get("n"),
                            "p": pat.get("p_value"), "excess": pat.get("excess_return"),
                            "sd": pat.get("oos_sd"), "sig": pat.get("significant")})
            except Exception as e:
                out.append({"cond": lbl, "status": f"err {str(e)[:30]}"})
    return out

def main() -> None:
    print("A COIN-SPECIFIC planted signal (fires when ONE coin outperforms its peers)\n")
    raw, rel = run(False), run(True)
    print(f"{'condition':<28}{'RAW: p':>9}{'excess':>9}{'status':>12}   |{'REL: p':>9}{'excess':>9}{'status':>12}")
    print("-"*104)
    def f(v, spec, mult=1.0):
        return format(v*mult, spec) if isinstance(v,(int,float)) and v==v else "--"
    for a,b in zip(raw, rel):
        print(f"{a['cond']:<28}{f(a.get('p'),'.4f'):>9}{f(a.get('excess'),'+.2f',100):>8}%{str(a.get('status'))[:11]:>12}   |"
              f"{f(b.get('p'),'.4f'):>9}{f(b.get('excess'),'+.2f',100):>8}%{str(b.get('status'))[:11]:>12}")
    ar=sum(1 for r in raw if r.get("status")=="accepted"); br=sum(1 for r in rel if r.get("status")=="accepted")
    print(f"\nACCEPTED -- raw outcome: {ar}/{len(raw)}   market-relative outcome: {br}/{len(rel)}")


if __name__ == "__main__":
    sys.exit(main())
