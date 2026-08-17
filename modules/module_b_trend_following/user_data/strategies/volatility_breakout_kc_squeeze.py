"""Module B, Phase 9 coarse-grid Family 2: Volatility Breakout.

Keltner Channels + TTM Squeeze momentum (`pandas_ta.squeeze`). A
"squeeze" is when Bollinger Bands sit inside the Keltner Channel (low
volatility, consolidation); the squeeze releasing (BB moving back outside
KC) with positive momentum signals a breakout starting. Enter long the
candle the squeeze fires (SQZ_OFF flips 0 -> 1) with positive momentum.

Exit signal: momentum turns negative, or price closes back inside the
Keltner Channel (the breakout failed / reverted).

Entry/exit parameter *values* come entirely from the coarse-grid presets
(coarse_grid.py), written into this strategy's auto-loaded
volatility_breakout_kc_squeeze.json before each backtest run.
"""

import pandas_ta as pta
from pandas import DataFrame

from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy

from dynamic_exit_mixin import NULL_ROI, NULL_STOPLOSS


class VolatilityBreakoutKCSqueeze(IStrategy):
    INTERFACE_VERSION = 3
    can_short = False

    minimal_roi = NULL_ROI
    stoploss = NULL_STOPLOSS
    trailing_stop = False

    startup_candle_count = 60

    bb_period = IntParameter(10, 60, default=20, space="buy", optimize=True)
    bb_std = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space="buy", optimize=True)
    kc_period = IntParameter(10, 60, default=20, space="buy", optimize=True)
    kc_atr_mult = DecimalParameter(1.0, 3.0, default=1.5, decimals=1, space="buy", optimize=True)
    momentum_period = IntParameter(10, 40, default=20, space="buy", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        mom_smooth = max(3, self.momentum_period.value // 3)
        sq = pta.squeeze(
            dataframe["high"],
            dataframe["low"],
            dataframe["close"],
            bb_length=self.bb_period.value,
            bb_std=self.bb_std.value,
            kc_length=self.kc_period.value,
            kc_scalar=self.kc_atr_mult.value,
            mom_length=self.momentum_period.value,
            mom_smooth=mom_smooth,
        )
        dataframe["sqz_momentum"] = sq.iloc[:, 0]
        dataframe["sqz_off"] = sq["SQZ_OFF"]

        kc = pta.kc(
            dataframe["high"], dataframe["low"], dataframe["close"],
            length=self.kc_period.value, scalar=self.kc_atr_mult.value,
        )
        dataframe["kc_lower"] = kc.iloc[:, 0]
        dataframe["kc_upper"] = kc.iloc[:, 2]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["sqz_off"] == 1)
                & (dataframe["sqz_off"].shift(1) == 0)
                & (dataframe["sqz_momentum"] > 0)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                ((dataframe["sqz_momentum"] < 0) | (dataframe["close"] < dataframe["kc_lower"]))
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1
        return dataframe
