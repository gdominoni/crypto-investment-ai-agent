"""Combo 2: Classic RSI + Price Bollinger Bands + CMF.

Same as Combo 1's trigger/RSI-filter, but volume confirmation via
Chaikin Money Flow (positive = net buying pressure) instead of a raw
volume surge. NOT YET RUN -- see combo1_classicrsi_pricebb_volume.py for
the full harness description.
"""

import pandas_ta as pta
from pandas import DataFrame

from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy

from dynamic_exit_mixin import NULL_ROI, NULL_STOPLOSS

_CMF_PERIOD = 20
_PRICE_BB_PERIOD = 20


def chaikin_money_flow(dataframe, cmf_threshold):
    dataframe["cmf"] = pta.cmf(
        dataframe["high"], dataframe["low"], dataframe["close"], dataframe["volume"], length=_CMF_PERIOD
    )
    dataframe["cmf_positive"] = dataframe["cmf"] > cmf_threshold
    return dataframe


def classic_rsi(dataframe, rsi_period):
    dataframe["rsi"] = pta.rsi(dataframe["close"], length=rsi_period)
    return dataframe


def price_bollinger_bands(dataframe, bb_std):
    bb = pta.bbands(dataframe["close"], length=_PRICE_BB_PERIOD, std=bb_std)
    dataframe["price_bb_mid"] = bb.iloc[:, 1]
    dataframe["price_bb_upper"] = bb.iloc[:, 2]
    return dataframe


class Combo2ClassicRsiPriceBBCmf(IStrategy):
    INTERFACE_VERSION = 3
    can_short = False

    minimal_roi = NULL_ROI
    stoploss = NULL_STOPLOSS
    trailing_stop = False
    startup_candle_count = 60

    rsi_period = IntParameter(10, 25, default=15, space="buy", step=5)
    rsi_oversold = IntParameter(20, 40, default=30, space="buy", step=5)
    rsi_overbought = IntParameter(60, 80, default=70, space="sell", step=5)
    bb_std = DecimalParameter(1.0, 3.0, default=2.0, decimals=None, space="buy", step=0.5)
    cmf_threshold = DecimalParameter(-0.05, 0.15, default=0.05, decimals=None, space="buy", step=0.05)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = classic_rsi(dataframe, self.rsi_period.value)
        dataframe = price_bollinger_bands(dataframe, self.bb_std.value)
        dataframe = chaikin_money_flow(dataframe, self.cmf_threshold.value)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["price_bb_upper"])
                & (dataframe["rsi"] > self.rsi_oversold.value)
                & dataframe["cmf_positive"]
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
