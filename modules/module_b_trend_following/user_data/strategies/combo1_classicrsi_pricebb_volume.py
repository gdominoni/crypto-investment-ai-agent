"""Combo 1: Classic RSI + Price Bollinger Bands + Volume Surge.

Entry: price closes above its upper Bollinger Band (trigger) while RSI
is above its oversold floor (falling-knife guard -- not entering during
an active capitulation) and volume confirms a real surge (not a
low-volume fakeout breakout).
Exit: RSI crosses into overbought territory, or price closes back below
the Bollinger mid-band (the breakout failed/reverted).

One of 8 combos in Module B's Phase 9 1h hyperopt harness -- see
confluence_indicators.py for shared indicator logic and
run_hyperopt_harness.py for how all 8 (x2 exit modes = 16 runs) are
orchestrated. NOT YET RUN.
"""

from pandas import DataFrame

from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy

from confluence_indicators import classic_rsi, price_bollinger_bands, volume_surge
from dynamic_exit_mixin import NULL_ROI, NULL_STOPLOSS


class Combo1ClassicRsiPriceBBVolume(IStrategy):
    INTERFACE_VERSION = 3
    can_short = False

    minimal_roi = NULL_ROI
    stoploss = NULL_STOPLOSS
    trailing_stop = False
    startup_candle_count = 60

    rsi_period = IntParameter(10, 25, default=15, space="buy", step=5)
    rsi_oversold = IntParameter(20, 40, default=30, space="buy", step=5)
    rsi_overbought = IntParameter(60, 80, default=70, space="sell", step=5)
    bb_std = DecimalParameter(1.0, 3.0, default=2.0, space="buy", step=0.5)
    volume_surge_mult = DecimalParameter(1.0, 2.5, default=1.5, space="buy", step=0.5)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = classic_rsi(dataframe, self.rsi_period.value)
        dataframe = price_bollinger_bands(dataframe, self.bb_std.value)
        dataframe = volume_surge(dataframe, self.volume_surge_mult.value)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["price_bb_upper"])
                & (dataframe["rsi"] > self.rsi_oversold.value)
                & dataframe["volume_surge"]
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                ((dataframe["rsi"] > self.rsi_overbought.value) | (dataframe["close"] < dataframe["price_bb_mid"]))
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1
        return dataframe
