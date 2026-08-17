"""TRUE exhaustive Cartesian-product parameter grid for Module B's
multi-factor confluence strategy, 1h timeframe ONLY.

This replaces multi_factor_grid.py's hand-curated presets (2 options per
family) with every discrete combination of the full parameter ranges
below -- no shortcuts, no pre-selected "logical" subsets. The grid itself
is pure data; `generate_grid()` is the harness that expands it into every
individual backtest configuration via itertools.product.

Dimension counts, exactly as specified:
    Family 1 (RSI & RSI-BB):  3 (rsi_length) x 3 (threshold pairs) x 4 (rsi_bb_std) = 36
    Family 2 (Breakout):      3 (bb_std) x 3 (kc_atr_mult)                          = 9
    Family 3 (Volume/CMF):    3 (volume_surge_multiple) x 3 (cmf_threshold)         = 9
    Exit variants:            2 (null, wide)                                        = 2
    TOTAL:                    36 x 9 x 9 x 2                                        = 5,832
"""

import itertools
from dataclasses import dataclass

TIMEFRAME = "1h"

# --- Family 1: RSI & RSI-BB (36 combinations) ---
RSI_LENGTHS = [10, 14, 20]
RSI_THRESHOLD_PAIRS = [(30, 70), (40, 60), (50, 50)]  # (oversold, overbought)
RSI_BB_STD = [1.0, 1.5, 2.0, 3.0]

# --- Family 2: Breakout / Squeeze (9 combinations) ---
BB_STD = [1.0, 2.0, 3.0]
KC_ATR_MULT = [1.2, 1.5, 2.0]

# --- Family 3: Volume & CMF (9 combinations) ---
VOLUME_SURGE_MULTIPLE = [1.2, 1.5, 2.0]
CMF_THRESHOLD = [0.0, 0.05, 0.10]

# --- Exit variants (2 combinations) ---
EXIT_VARIANTS = ["null", "wide"]
WIDE_STOPLOSS = -0.15
WIDE_TAKE_PROFIT = 0.15
NULL_STOPLOSS = -0.99


@dataclass(frozen=True)
class GridCombo:
    index: int
    rsi_length: int
    rsi_oversold: int
    rsi_overbought: int
    rsi_bb_std: float
    bb_std: float
    kc_atr_mult: float
    volume_surge_multiple: float
    cmf_threshold: float
    exit_variant: str

    @property
    def stoploss(self) -> float:
        return NULL_STOPLOSS if self.exit_variant == "null" else WIDE_STOPLOSS

    @property
    def take_profit(self) -> float | None:
        return None if self.exit_variant == "null" else WIDE_TAKE_PROFIT


def generate_grid() -> list[GridCombo]:
    combos = []
    dimensions = itertools.product(
        RSI_LENGTHS, RSI_THRESHOLD_PAIRS, RSI_BB_STD,
        BB_STD, KC_ATR_MULT,
        VOLUME_SURGE_MULTIPLE, CMF_THRESHOLD,
        EXIT_VARIANTS,
    )
    for i, (rsi_len, (oversold, overbought), rsi_bb_std, bb_std, kc_mult, vol_mult, cmf_thresh, exit_variant) in enumerate(dimensions):
        combos.append(
            GridCombo(
                index=i,
                rsi_length=rsi_len,
                rsi_oversold=oversold,
                rsi_overbought=overbought,
                rsi_bb_std=rsi_bb_std,
                bb_std=bb_std,
                kc_atr_mult=kc_mult,
                volume_surge_multiple=vol_mult,
                cmf_threshold=cmf_thresh,
                exit_variant=exit_variant,
            )
        )
    return combos


if __name__ == "__main__":
    grid = generate_grid()
    print(f"Family 1 (RSI & RSI-BB):  {len(RSI_LENGTHS)} x {len(RSI_THRESHOLD_PAIRS)} x {len(RSI_BB_STD)} = {len(RSI_LENGTHS)*len(RSI_THRESHOLD_PAIRS)*len(RSI_BB_STD)}")
    print(f"Family 2 (Breakout):      {len(BB_STD)} x {len(KC_ATR_MULT)} = {len(BB_STD)*len(KC_ATR_MULT)}")
    print(f"Family 3 (Volume/CMF):    {len(VOLUME_SURGE_MULTIPLE)} x {len(CMF_THRESHOLD)} = {len(VOLUME_SURGE_MULTIPLE)*len(CMF_THRESHOLD)}")
    print(f"Exit variants:            {len(EXIT_VARIANTS)}")
    print(f"TOTAL generated: {len(grid)}")
    assert len(grid) == 5832, f"Expected 5832, got {len(grid)}"
    assert len(set(grid)) == 5832, "Duplicate combinations found in grid"
    print("Confirmed: exactly 5,832 unique combinations.")
    print("\nFirst 3 combos:")
    for c in grid[:3]:
        print(f"  {c}")
    print("\nLast combo:")
    print(f"  {grid[-1]}")
