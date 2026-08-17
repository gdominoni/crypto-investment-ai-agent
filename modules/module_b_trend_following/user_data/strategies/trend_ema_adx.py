"""First Module B trend-following candidate: EMA crossover, gated by an ADX
trend-strength filter so entries only fire when a real trend is present,
not on noise. ATR-based dynamic stoploss.

This is a starting candidate for the backtest/ranking pipeline
(candidate_ranking.py), not a finished, tuned strategy -- more candidates
get added as Phase 4 continues, and hyperopt sweeps come later.
"""

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class TrendEmaAdx(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"

    # Conservative fixed exits; refined once real backtest stats exist.
    minimal_roi = {"0": 0.10, "240": 0.05, "1440": 0.02}
    stoploss = -0.08
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True

    startup_candle_count = 60

    ema_fast_period = 20
    ema_slow_period = 50
    adx_trend_threshold = 25

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast_period)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow_period)
        dataframe["adx"] = ta.ADX(dataframe)
        dataframe["atr"] = ta.ATR(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["ema_fast"] > dataframe["ema_slow"])
                & (dataframe["ema_fast"].shift(1) <= dataframe["ema_slow"].shift(1))
                & (dataframe["adx"] > self.adx_trend_threshold)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["ema_fast"] < dataframe["ema_slow"])
                & (dataframe["ema_fast"].shift(1) >= dataframe["ema_slow"].shift(1))
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1
        return dataframe
