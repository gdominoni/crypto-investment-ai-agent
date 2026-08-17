"""Combo 7: RSI-Bollinger-Bands + Keltner Channels + Volume Surge.

NOT YET RUN -- see combo1_classicrsi_pricebb_volume.py and
combo5_rsibb_pricebb_volume.py for the full harness description and the
RSI-BB adaptive-band design.
"""

import pandas_ta as pta
from pandas import DataFrame

from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy

from dynamic_exit_mixin import NULL_ROI, NULL_STOPLOSS

_KC_PERIOD = 20
_RSI_BB_PERIOD = 20
_VOLUME_LOOKBACK = 20


def keltner_channel(dataframe, kc_atr_mult):
    kc = pta.kc(dataframe["high"], dataframe["low"], dataframe["close"], length=_KC_PERIOD, scalar=kc_atr_mult)
    dataframe["kc_mid"] = kc.iloc[:, 1]
    dataframe["kc_upper"] = kc.iloc[:, 2]
    return dataframe


def rsi_bollinger_bands(dataframe, rsi_period, rsi_bb_std):
    dataframe["rsi"] = pta.rsi(dataframe["close"], length=rsi_period)
    bb = pta.bbands(dataframe["rsi"], length=_RSI_BB_PERIOD, std=rsi_bb_std)
    dataframe["rsi_bb_lower"] = bb.iloc[:, 0]
    dataframe["rsi_bb_upper"] = bb.iloc[:, 2]
    return dataframe


def volume_surge(dataframe, volume_surge_mult):
    avg_volume = dataframe["volume"].rolling(_VOLUME_LOOKBACK).mean()
    dataframe["volume_surge"] = dataframe["volume"] > (avg_volume * volume_surge_mult)
    return dataframe


class Combo7RsiBBKeltnerVolume(IStrategy):
    INTERFACE_VERSION = 3
    can_short = False

    minimal_roi = NULL_ROI
    stoploss = NULL_STOPLOSS
    trailing_stop = False
    startup_candle_count = 60

    rsi_period = IntParameter(10, 25, default=15, space="buy", step=5)
    rsi_bb_std = DecimalParameter(1.0, 3.0, default=2.0, decimals=None, space="buy", step=0.5)
    kc_atr_mult = DecimalParameter(1.0, 2.5, default=1.5, decimals=None, space="buy", step=0.5)
    volume_surge_mult = DecimalParameter(1.0, 2.5, default=1.5, decimals=None, space="buy", step=0.5)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = rsi_bollinger_bands(dataframe, self.rsi_period.value, self.rsi_bb_std.value)
        dataframe = keltner_channel(dataframe, self.kc_atr_mult.value)
        dataframe = volume_surge(dataframe, self.volume_surge_mult.value)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["kc_upper"])
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
                ((dataframe["rsi"] > dataframe["rsi_bb_upper"]) | (dataframe["close"] < dataframe["kc_mid"]))
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1
        return dataframe
