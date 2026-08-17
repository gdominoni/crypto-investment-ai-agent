"""Combo 6: RSI-Bollinger-Bands + Price Bollinger Bands + CMF.

NOT YET RUN -- see combo1_classicrsi_pricebb_volume.py and
combo5_rsibb_pricebb_volume.py for the full harness description and the
RSI-BB adaptive-band design.
"""

import pandas_ta as pta
from pandas import DataFrame

from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy

from dynamic_exit_mixin import NULL_ROI, NULL_STOPLOSS

_CMF_PERIOD = 20
_PRICE_BB_PERIOD = 20
_RSI_BB_PERIOD = 20


def chaikin_money_flow(dataframe, cmf_threshold):
    dataframe["cmf"] = pta.cmf(
        dataframe["high"], dataframe["low"], dataframe["close"], dataframe["volume"], length=_CMF_PERIOD
    )
    dataframe["cmf_positive"] = dataframe["cmf"] > cmf_threshold
    return dataframe


def price_bollinger_bands(dataframe, bb_std):
    bb = pta.bbands(dataframe["close"], length=_PRICE_BB_PERIOD, std=bb_std)
    dataframe["price_bb_mid"] = bb.iloc[:, 1]
    dataframe["price_bb_upper"] = bb.iloc[:, 2]
    return dataframe


def rsi_bollinger_bands(dataframe, rsi_period, rsi_bb_std):
    dataframe["rsi"] = pta.rsi(dataframe["close"], length=rsi_period)
    bb = pta.bbands(dataframe["rsi"], length=_RSI_BB_PERIOD, std=rsi_bb_std)
    dataframe["rsi_bb_lower"] = bb.iloc[:, 0]
    dataframe["rsi_bb_upper"] = bb.iloc[:, 2]
    return dataframe


class Combo6RsiBBPriceBBCmf(IStrategy):
    INTERFACE_VERSION = 3
    can_short = False

    minimal_roi = NULL_ROI
    stoploss = NULL_STOPLOSS
    trailing_stop = False
    startup_candle_count = 60

    rsi_period = IntParameter(10, 25, default=15, space="buy", step=5)
    rsi_bb_std = DecimalParameter(1.0, 3.0, default=2.0, decimals=None, space="buy", step=0.5)
    bb_std = DecimalParameter(1.0, 3.0, default=2.0, decimals=None, space="buy", step=0.5)
    cmf_threshold = DecimalParameter(-0.05, 0.15, default=0.05, decimals=None, space="buy", step=0.05)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = rsi_bollinger_bands(dataframe, self.rsi_period.value, self.rsi_bb_std.value)
        dataframe = price_bollinger_bands(dataframe, self.bb_std.value)
        dataframe = chaikin_money_flow(dataframe, self.cmf_threshold.value)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["price_bb_upper"])
                & (dataframe["rsi"] > dataframe["rsi_bb_lower"])
                & dataframe["cmf_positive"]
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
