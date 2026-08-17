"""Module B, Phase 9 coarse-grid Family 3: Volume-Driven.

Rolling VWAP + volume-surge detection + Chaikin Money Flow (CMF) as a
buying-pressure confirmation. Enter long when price crosses above its
rolling VWAP with an above-average volume surge, and CMF confirms net
buying pressure.

"Rolling VWAP" is a deliberate simplification of "Anchored VWAP": a true
anchored VWAP resets its cumulative sum at a fixed calendar point
(session/day/week start); this uses a fixed-length rolling window
instead (VWAP over the last N candles), which behaves similarly as a
mean-reversion-to-volume-weighted-price reference but doesn't have a
true reset point. `pandas_ta.vwap` itself requires a DatetimeIndex, which
Freqtrade's `populate_indicators` dataframe doesn't have (it carries a
`date` *column*, not a datetime index) -- computed manually instead.

Exit signal: price closes back below VWAP, or CMF turns negative
(buying pressure has reversed).

Entry/exit parameter *values* come entirely from the coarse-grid presets
(coarse_grid.py), written into this strategy's auto-loaded
volume_driven_vwap_cmf.json before each backtest run.
"""

import pandas_ta as pta
from pandas import DataFrame

from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy

from dynamic_exit_mixin import NULL_ROI, NULL_STOPLOSS


class VolumeDrivenVWAPCMF(IStrategy):
    INTERFACE_VERSION = 3
    can_short = False

    minimal_roi = NULL_ROI
    stoploss = NULL_STOPLOSS
    trailing_stop = False

    startup_candle_count = 200

    vwap_period = IntParameter(20, 200, default=24, space="buy", optimize=True)
    volume_surge_period = IntParameter(10, 30, default=20, space="buy", optimize=True)
    volume_surge_multiple = DecimalParameter(1.2, 3.0, default=1.5, decimals=1, space="buy", optimize=True)
    cmf_period = IntParameter(10, 30, default=20, space="buy", optimize=True)
    cmf_threshold = DecimalParameter(0.0, 0.2, default=0.05, decimals=2, space="buy", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        period = self.vwap_period.value
        price_volume = dataframe["close"] * dataframe["volume"]
        dataframe["vwap"] = price_volume.rolling(period).sum() / dataframe["volume"].rolling(period).sum()

        avg_volume = dataframe["volume"].rolling(self.volume_surge_period.value).mean()
        dataframe["volume_surge"] = dataframe["volume"] > (avg_volume * self.volume_surge_multiple.value)

        dataframe["cmf"] = pta.cmf(
            dataframe["high"], dataframe["low"], dataframe["close"], dataframe["volume"],
            length=self.cmf_period.value,
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["vwap"])
                & (dataframe["close"].shift(1) <= dataframe["vwap"].shift(1))
                & dataframe["volume_surge"]
                & (dataframe["cmf"] > self.cmf_threshold.value)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (((dataframe["close"] < dataframe["vwap"]) | (dataframe["cmf"] < 0)) & (dataframe["volume"] > 0)),
            "exit_long",
        ] = 1
        return dataframe
