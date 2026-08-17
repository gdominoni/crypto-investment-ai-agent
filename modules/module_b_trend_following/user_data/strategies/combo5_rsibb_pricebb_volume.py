"""Combo 5: RSI-Bollinger-Bands + Price Bollinger Bands + Volume Surge.

Family 1's filter is now adaptive: RSI above its own lower RSI-BB band
(not fixed-threshold oversold) and, symmetrically, exit fires when RSI
crosses back above its own upper RSI-BB band -- both bands share one
std-dev parameter (rsi_bb_std), the natural way Bollinger Bands work.
NOT YET RUN -- see combo1_classicrsi_pricebb_volume.py for the full
harness description.
"""

from pandas import DataFrame

from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy

from confluence_indicators import price_bollinger_bands, rsi_bollinger_bands, volume_surge
from dynamic_exit_mixin import NULL_ROI, NULL_STOPLOSS


class Combo5RsiBBPriceBBVolume(IStrategy):
    INTERFACE_VERSION = 3
    can_short = False

    minimal_roi = NULL_ROI
    stoploss = NULL_STOPLOSS
    trailing_stop = False
    startup_candle_count = 60

    rsi_period = IntParameter(10, 25, default=15, space="buy", step=5)
    rsi_bb_std = DecimalParameter(1.0, 3.0, default=2.0, space="buy", step=0.5)
    bb_std = DecimalParameter(1.0, 3.0, default=2.0, space="buy", step=0.5)
    volume_surge_mult = DecimalParameter(1.0, 2.5, default=1.5, space="buy", step=0.5)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = rsi_bollinger_bands(dataframe, self.rsi_period.value, self.rsi_bb_std.value)
        dataframe = price_bollinger_bands(dataframe, self.bb_std.value)
        dataframe = volume_surge(dataframe, self.volume_surge_mult.value)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["price_bb_upper"])
                & (dataframe["rsi"] > dataframe["rsi_bb_lower"])
                & dataframe["volume_surge"]
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                ((dataframe["rsi"] > dataframe["rsi_bb_upper"]) | (dataframe["close"] < dataframe["price_bb_mid"]))
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1
        return dataframe
