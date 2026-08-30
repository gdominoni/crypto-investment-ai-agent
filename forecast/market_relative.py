"""Does testing the MARKET-RELATIVE return instead of the raw return
rescue candidates the significance gate is currently killing?

The autopsy on 294 known-good (planted) conditions found that 92% of
those with a valid test and adequate data die at `p >= 0.05` -- not at
the sample-size gate, not at concentration (4 deaths), not at MFE/MAE
(zero deaths). Statistical power is the binding constraint, and this
whole project has been attacking it from the `n` side only.

Power goes as effect / (sigma / sqrt(n)). Sigma has never been touched,
and it enters quadratically. Measured on real data: the pooled SD of
7-day forward returns is 16.18% raw but 11.41% after subtracting the
equal-weight basket return, because the average cross-coin correlation
of forward returns is 0.54 -- most of any coin's move IS the market's
move. That is a 0.71x on sigma, so HALF the sample for the same power.

This module re-runs the planted and random arms with both the treated
returns and the baseline returns measured relative to the equal-weight
basket, and re-runs the same autopsy. Two numbers decide whether the
lever is worth building:

    * how many of the 154 significance deaths come back, and
    * whether the RANDOM arm stays silent.

The second is not optional. A change that buys power by inflating the
false-positive rate is not a power gain, it is a broken test, and the
random arm is the only thing that can tell the difference.

IMPORTANT SCOPE. Market-relative is the right outcome for a COIN-SPECIFIC
hypothesis ("does XRP outperform after XRP news"). It is the WRONG
outcome for a market-wide one: if a CPI print moves all of crypto
together, subtracting the basket deletes the very effect being measured.
So this can never be a global switch -- it must be chosen by hypothesis
type and declared in the spec, exactly like the concentration rule. This
experiment measures the ceiling of the idea, using planted signals that
are per-coin by construction.

Run:  python3 -m forecast.market_relative
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_PATH = Path(__file__).resolve().parent / "market_relative.json"
COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LTCUSDT"]


def _basket_forward_returns() -> dict[int, pd.Series]:
    """Equal-weight basket forward return, per horizon, indexed by date."""
    from candidates.data_loading import load_daily
    closes = {c: load_daily(c)["close"] for c in COINS}
    idx = sorted(set().union(*[set(s.index) for s in closes.values()]))
    frame = pd.DataFrame({c: closes[c].reindex(idx) for c in COINS})
    out = {}
    for h in (1, 3, 7, 14, 21):
        out[h] = (frame.shift(-h) / frame - 1.0).mean(axis=1)
    return out


def install_market_relative():
    """Patch the outcome measurement -- treated AND baseline alike.

    Both must be patched. Measuring the treated events relative to the
    basket while leaving the baseline raw would compare two different
    quantities and produce a meaningless excess return that would look
    like a huge discovery.
    """
    import candidates.methodology as M

    basket = _basket_forward_returns()
    raw_fwd, raw_base = M._forward_return, M._baseline_forward_returns

    def rel_forward_return(entry_loc, ohlc, direction, horizon):
        v = raw_fwd(entry_loc, ohlc, direction, horizon)
        if v != v:
            return v
        mkt = basket.get(int(horizon))
        if mkt is None:
            return v
        try:
            m = mkt.get(ohlc.index[entry_loc], np.nan)
        except Exception:
            return v
        if m != m:
            return v
        sign = 1.0 if direction == "long" else -1.0
        return v - sign * float(m)

    def rel_baseline(ohlc, direction, horizon, start_loc, end_loc):
        v = raw_base(ohlc, direction, horizon, start_loc, end_loc)
        if len(v) == 0:
            return v
        mkt = basket.get(int(horizon))
        if mkt is None:
            return v
        last = min(end_loc, len(ohlc.index) - horizon)
        dates = ohlc.index[start_loc:last]
        m = mkt.reindex(dates).to_numpy(dtype=float)
        sign = 1.0 if direction == "long" else -1.0
        out = v - sign * np.nan_to_num(m, nan=0.0)
        return out[np.isfinite(out)]

    M._forward_return = rel_forward_return
    M._baseline_forward_returns = rel_baseline
    return raw_fwd, raw_base


def main() -> None:
    install_market_relative()
    from forecast.control_sweep import ARMS, build_specs, register_arms
    from llm_pipeline.novel_condition_tester import test_novel_condition

    register_arms()
    # Only the arms with known ground truth: planted (signal is real) and
    # random (signal is absent). real_news is an open question and cannot
    # validate anything.
    specs = [s for s in build_specs()
             if any(k in s.label for k in ("planted20", "planted35", "planted50", "random"))]
    done = {}
    if RESULTS_PATH.exists():
        done = {r["label"]: r for r in json.loads(RESULTS_PATH.read_text())}
    todo = [s for s in specs if s.label not in done]
    print(f"{len(specs)} conditions (planted + random arms), {len(todo)} to run\n", flush=True)

    t0 = time.time()
    for i, spec in enumerate(todo):
        try:
            r = test_novel_condition(spec, COINS)
            pat = r.get("pattern_significance") or {}
            done[spec.label] = {
                "label": spec.label, "arm": spec.label.split("__")[0].replace("ctl_", ""),
                "status": r.get("status"), "n": r.get("n"), "p": pat.get("p_value"),
                "excess": pat.get("excess_return"), "mfe_mae": pat.get("mfe_mae_ratio"),
                "significant": pat.get("significant"), "pat_status": pat.get("status"),
                "oos_sd": pat.get("oos_sd"),
            }
        except Exception as e:
            done[spec.label] = {"label": spec.label, "arm": spec.label.split("__")[0].replace("ctl_", ""),
                                "status": "error", "err": str(e)[:120]}
        RESULTS_PATH.write_text(json.dumps(list(done.values()), indent=1))
        if (i + 1) % 40 == 0:
            el = time.time() - t0
            print(f"  {len(done)}/{len(specs)}  ({el/60:.0f}m, "
                  f"~{(el/(i+1))*(len(todo)-i-1)/60:.0f}m left)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
