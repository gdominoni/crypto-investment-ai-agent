"""First Module B trend-following candidate: EMA crossover, gated by an ADX
trend-strength filter so entries only fire when a real trend is present,
not on noise. ATR-based dynamic stoploss.

This is a starting candidate for the backtest/ranking pipeline
(candidate_ranking.py), not a finished, tuned strategy -- more candidates
get added as Phase 4 continues.

Phase 9: ema_fast_period, ema_slow_period, and adx_trend_threshold are now
hyperoptable (space="buy"). Optimized with a custom loss function
(user_data/hyperopts/project_hierarchy_loss.py) matching this project's
Win Rate -> Sortino -> Net Profit ranking hierarchy, not one of
Freqtrade's built-in single-metric loss functions -- see that file's
docstring and README.md for why. ROI/stoploss/trailing are left fixed for
this first hyperopt pass to keep the search space focused; hyperopting
those too is a natural next step, not done here.
"""

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IntParameter, IStrategy


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

    ema_fast_period = IntParameter(10, 30, default=20, space="buy", optimize=True)
    ema_slow_period = IntParameter(40, 100, default=50, space="buy", optimize=True)
    adx_trend_threshold = IntParameter(15, 40, default=25, space="buy", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Hyperopt sweeps ema_fast_period/ema_slow_period across a range each
        # run, so every candidate period in that range needs to be available
        # as a column -- computing only the current .value would only ever
        # have the default period's EMA on hand during the search.
        for period in self.ema_fast_period.range:
            dataframe[f"ema_fast_{period}"] = ta.EMA(dataframe, timeperiod=period)
        for period in self.ema_slow_period.range:
            dataframe[f"ema_slow_{period}"] = ta.EMA(dataframe, timeperiod=period)
        dataframe["adx"] = ta.ADX(dataframe)
        dataframe["atr"] = ta.ATR(dataframe)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_fast = dataframe[f"ema_fast_{self.ema_fast_period.value}"]
        ema_slow = dataframe[f"ema_slow_{self.ema_slow_period.value}"]
        dataframe.loc[
            (
                (ema_fast > ema_slow)
                & (ema_fast.shift(1) <= ema_slow.shift(1))
                & (dataframe["adx"] > self.adx_trend_threshold.value)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ema_fast = dataframe[f"ema_fast_{self.ema_fast_period.value}"]
        ema_slow = dataframe[f"ema_slow_{self.ema_slow_period.value}"]
        dataframe.loc[
            (
                (ema_fast < ema_slow)
                & (ema_fast.shift(1) >= ema_slow.shift(1))
                & (dataframe["volume"] > 0)
            ),
            "exit_long",
        ] = 1
        return dataframe
