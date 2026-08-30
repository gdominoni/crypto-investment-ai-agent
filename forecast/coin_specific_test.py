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
"""
import sys, json; sys.path.insert(0, ".")
import numpy as np, pandas as pd

from forecast.market_relative import COINS, _basket_forward_returns, install_market_relative

BASKET = _basket_forward_returns()
H = 7


def make_planted_relative(top_frac=0.20, event_frac=0.02, seed=5):
    """Fires on days whose MARKET-RELATIVE forward return is in the top
    `top_frac` -- a coin outperforming its peers, not a rising tide."""
    def indicator(df, funding, scale=1):
        fwd = df["close"].shift(-H) / df["close"] - 1.0
        rel = fwd - BASKET[H].reindex(df.index)
        thresh = rel.quantile(1.0 - top_frac)
        elig = (rel >= thresh).fillna(False)
        rng = np.random.default_rng(seed)
        keep = pd.Series(rng.random(len(df)), index=df.index) < (event_frac / top_frac)
        return (elig & keep).astype(float)
    return indicator


def run(relative_outcome: bool):
    import importlib
    import candidates.methodology as M
    importlib.reload(M)                    # restore pristine _forward_return
    import llm_pipeline.novel_condition_tester as N
    importlib.reload(N)
    if relative_outcome:
        install_market_relative()
    from llm_pipeline.novel_condition_tester import Clause, ConditionSpec, test_novel_condition

    key = "sent_coinspecific"
    N.SUPPORTED_INDICATORS[key] = make_planted_relative()
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
                r = test_novel_condition(ConditionSpec(label=f"cs_{lbl}", clauses=clauses, direction="long"), COINS)
                pat = r.get("pattern_significance") or {}
                out.append({"cond": lbl, "status": r.get("status"), "n": r.get("n"),
                            "p": pat.get("p_value"), "excess": pat.get("excess_return"),
                            "sd": pat.get("oos_sd"), "sig": pat.get("significant")})
            except Exception as e:
                out.append({"cond": lbl, "status": f"err {str(e)[:30]}"})
    return out

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
