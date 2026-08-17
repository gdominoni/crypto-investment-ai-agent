"""Phase 9, Module B redesign: multi-factor confluence grids for 1h/4h/1d
(15m dropped -- Phase 1 showed uniform, large losses there, consistent
with fee drag from very high trade frequency).

SUPERSEDES the isolated single-family testing in coarse_grid.py (kept,
not deleted -- see README.md and decisions-log.md for why the isolated
approach was replaced, not just extended).

Design: instead of testing each family alone, MultiFactorConfluence
(the strategy) combines all three on every candle, in asymmetric roles:
- Family 2 (volatility breakout / squeeze release) is the TIMING TRIGGER
  -- a squeeze release is a naturally rare, punctual event, well suited
  to being the thing that actually opens a trade.
- Family 1 (RSI-based) is a CONFIRMING FILTER: "not already overbought,"
  guarding against chasing an extended move.
- Family 3 (volume) is a CONFIRMING FILTER: "real buying interest behind
  this breakout," guarding against a low-volume fakeout/chop breakout.

Family 1 offers two variants, per the explicit instruction to also test
an RSI-derived indicator rather than only classic fixed-threshold RSI:
- "classic": RSI < a fixed overbought threshold.
- "rsi_bb": Bollinger Bands applied to the RSI *series itself* (not
  price) -- RSI < its own upper RSI-BB band. Adaptive: the "overbought"
  line moves with how volatile RSI itself has recently been, rather than
  sitting at a fixed level regardless of regime.

Per-family option counts are deliberately small (2 each, not 3) because
this grid multiplies three dimensions together (Cartesian product) rather
than testing them separately -- 3x3x3 would already be 27 entry
combinations before even adding exits and timeframes; keeping this a
genuine *coarse* screen means shrinking each dimension, not just letting
the product grow.

Exits: per instruction, only 2 presets per timeframe now (down from 4) --
"null" (pure indicator-driven exit) and one "wide" SL+TP pair, materially
wider than Phase 1's tightest tiers. Phase 1's finding (null outperformed
every fixed-percentage exit on average) is the direct reason for this
narrowing, and for weighting the exit logic itself toward indicator/
trend-reversal signals (see the strategy file) even when SL/TP aren't null.
"""

from dataclasses import dataclass

TIMEFRAMES = ["1h", "4h", "1d"]


@dataclass(frozen=True)
class ExitPreset:
    name: str
    stoploss: float  # negative fraction; -0.99 for "null"
    take_profit: float | None  # fraction; None for "null"


NULL_SL = -0.99

EXIT_GRIDS: dict[str, list[ExitPreset]] = {
    "1h": [
        ExitPreset("null", stoploss=NULL_SL, take_profit=None),
        ExitPreset("wide", stoploss=-0.06, take_profit=0.12),
    ],
    "4h": [
        ExitPreset("null", stoploss=NULL_SL, take_profit=None),
        ExitPreset("wide", stoploss=-0.10, take_profit=0.20),
    ],
    "1d": [
        ExitPreset("null", stoploss=NULL_SL, take_profit=None),
        ExitPreset("wide", stoploss=-0.15, take_profit=0.30),
    ],
}


# Family 1 (RSI-based confirming filter: "not overbought").
FAMILY1_GRIDS: dict[str, list[dict]] = {
    "1h": [
        {"variant": "classic", "rsi_period": 14, "rsi_overbought": 70},
        {"variant": "rsi_bb", "rsi_period": 14, "rsi_bb_period": 20, "rsi_bb_std": 2.0},
    ],
    "4h": [
        {"variant": "classic", "rsi_period": 14, "rsi_overbought": 65},
        {"variant": "rsi_bb", "rsi_period": 14, "rsi_bb_period": 20, "rsi_bb_std": 2.5},
    ],
    "1d": [
        {"variant": "classic", "rsi_period": 14, "rsi_overbought": 65},
        {"variant": "rsi_bb", "rsi_period": 14, "rsi_bb_period": 20, "rsi_bb_std": 2.5},
    ],
}

# Family 2 (volatility breakout: the timing trigger).
FAMILY2_GRIDS: dict[str, list[dict]] = {
    "1h": [
        {"bb_period": 20, "bb_std": 2.0, "kc_period": 20, "kc_atr_mult": 1.5, "momentum_period": 20},
        {"bb_period": 20, "bb_std": 2.0, "kc_period": 20, "kc_atr_mult": 2.0, "momentum_period": 30},
    ],
    "4h": [
        {"bb_period": 20, "bb_std": 2.0, "kc_period": 20, "kc_atr_mult": 2.0, "momentum_period": 20},
        {"bb_period": 20, "bb_std": 2.5, "kc_period": 20, "kc_atr_mult": 2.5, "momentum_period": 30},
    ],
    "1d": [
        {"bb_period": 20, "bb_std": 2.0, "kc_period": 20, "kc_atr_mult": 2.0, "momentum_period": 14},
        {"bb_period": 20, "bb_std": 2.5, "kc_period": 20, "kc_atr_mult": 2.5, "momentum_period": 20},
    ],
}

# Family 3 (volume: confirming filter -- real interest, not a fakeout).
FAMILY3_GRIDS: dict[str, list[dict]] = {
    "1h": [
        {"volume_surge_period": 20, "volume_surge_multiple": 1.5, "cmf_period": 20, "cmf_threshold": 0.05},
        {"volume_surge_period": 20, "volume_surge_multiple": 2.0, "cmf_period": 20, "cmf_threshold": 0.10},
    ],
    "4h": [
        {"volume_surge_period": 20, "volume_surge_multiple": 1.5, "cmf_period": 20, "cmf_threshold": 0.05},
        {"volume_surge_period": 20, "volume_surge_multiple": 2.0, "cmf_period": 30, "cmf_threshold": 0.10},
    ],
    "1d": [
        {"volume_surge_period": 14, "volume_surge_multiple": 1.5, "cmf_period": 20, "cmf_threshold": 0.05},
        {"volume_surge_period": 20, "volume_surge_multiple": 2.0, "cmf_period": 20, "cmf_threshold": 0.10},
    ],
}
