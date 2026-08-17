"""Module B, Phase 9 coarse-grid Family 1: Mean Reversion.

Bollinger Bands + RSI + rolling Z-score. Enter long when price closes
below the lower Bollinger Band AND RSI is oversold AND the Z-score
confirms an extreme deviation from the rolling mean -- three
overlapping confirmations of the same "oversold" idea, deliberately
redundant (Bollinger position and Z-score measure closely related
things), included because the coarse-grid spec named both explicitly.

Exit signal (what drives an exit when the coarse grid's SL/TP preset is
"null"): price closes back above the middle band (mean reversion
complete) or RSI crosses back above 50.

Entry/exit parameter *values* come entirely from the coarse-grid presets
(coarse_grid.py), written into this strategy's auto-loaded
mean_reversion_bb_rsi.json before each backtest run -- this file defines
the parameter *shape*, not any one timeframe's tuning.
"""

import pandas_ta as pta
from pandas import DataFrame

from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy

from dynamic_exit_mixin import NULL_ROI, NULL_STOPLOSS


class MeanReversionBBRSI(IStrategy):
    INTERFACE_VERSION = 3
    can_short = False

    minimal_roi = NULL_ROI
    stoploss = NULL_STOPLOSS
    trailing_stop = False

    startup_candle_count = 60

    bb_period = IntParameter(10, 60, default=20, space="buy", optimize=True)
    bb_std = DecimalParameter(1.5, 3.5, default=2.0, decimals=1, space="buy", optimize=True)
    rsi_period = IntParameter(7, 21, default=14, space="buy", optimize=True)
    rsi_oversold = IntParameter(15, 40, default=30, space="buy", optimize=True)
    rsi_overbought = IntParameter(60, 85, default=70, space="buy", optimize=True)
    zscore_threshold = DecimalParameter(1.0, 3.5, default=2.0, decimals=1, space="buy", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        bb = pta.bbands(dataframe["close"], length=self.bb_period.value, std=self.bb_std.value)
        # Select by position, not name -- pandas_ta's bbands column names
        # embed the std parameter twice (e.g. "BBL_20_2.0_2.0") in a format
        # that isn't safe to reconstruct from the raw float value.
        dataframe["bb_lower"] = bb.iloc[:, 0]
        dataframe["bb_mid"] = bb.iloc[:, 1]
        dataframe["bb_upper"] = bb.iloc[:, 2]

        dataframe["rsi"] = pta.rsi(dataframe["close"], length=self.rsi_period.value)

        sma = dataframe["close"].rolling(self.bb_period.value).mean()
        std = dataframe["close"].rolling(self.bb_period.value).std()
        dataframe["zscore"] = (dataframe["close"] - sma) / std
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["close"] < dataframe["bb_lower"])
                & (dataframe["rsi"] < self.rsi_oversold.value)
                & (dataframe["zscore"] < -self.zscore_threshold.value)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (((dataframe["close"] > dataframe["bb_mid"]) | (dataframe["rsi"] > 50)) & (dataframe["volume"] > 0)),
            "exit_long",
        ] = 1
        return dataframe
