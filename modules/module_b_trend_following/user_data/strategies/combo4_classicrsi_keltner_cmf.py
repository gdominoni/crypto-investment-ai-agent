"""Combo 4: Classic RSI + Keltner Channels + CMF.

NOT YET RUN -- see combo1_classicrsi_pricebb_volume.py for the full
harness description.
"""

from pandas import DataFrame

from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy

from confluence_indicators import chaikin_money_flow, classic_rsi, keltner_channel
from dynamic_exit_mixin import NULL_ROI, NULL_STOPLOSS


class Combo4ClassicRsiKeltnerCmf(IStrategy):
    INTERFACE_VERSION = 3
    can_short = False

    minimal_roi = NULL_ROI
    stoploss = NULL_STOPLOSS
    trailing_stop = False
    startup_candle_count = 60

    rsi_period = IntParameter(10, 25, default=15, space="buy", step=5)
    rsi_oversold = IntParameter(20, 40, default=30, space="buy", step=5)
    rsi_overbought = IntParameter(60, 80, default=70, space="sell", step=5)
    kc_atr_mult = DecimalParameter(1.0, 2.5, default=1.5, space="buy", step=0.5)
    cmf_threshold = DecimalParameter(-0.05, 0.15, default=0.05, space="buy", step=0.05)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = classic_rsi(dataframe, self.rsi_period.value)
        dataframe = keltner_channel(dataframe, self.kc_atr_mult.value)
        dataframe = chaikin_money_flow(dataframe, self.cmf_threshold.value)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["kc_upper"])
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
                ((dataframe["rsi"] > self.rsi_overbought.value) | (dataframe["close"] < dataframe["kc_mid"]))
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1
        return dataframe
