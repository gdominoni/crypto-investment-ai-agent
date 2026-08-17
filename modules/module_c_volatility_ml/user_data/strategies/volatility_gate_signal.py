"""Module C, Phase 9: thin strategy consuming the bespoke walk-forward
classifier's precomputed signal (../../train.py) for realistic trade
simulation via Freqtrade's backtesting engine -- fees, ROI, stoploss --
rather than reinventing a trade simulator ourselves. No FreqAI here: all
the ML (Triple Barrier labeling, SHAP feature selection, purged walk-
forward training, threshold calibration) happens outside Freqtrade
entirely. See README.md for why Module C moved off FreqAI for this.
"""

from pathlib import Path

import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy

SIGNAL_PATH = Path("/freqtrade/user_data/signal_output/high_risk_signal.parquet")


class VolatilityGateSignal(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False

    minimal_roi = {"0": 0.03}
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True

    startup_candle_count = 30

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        signal = pd.read_parquet(SIGNAL_PATH)
        signal["date"] = pd.to_datetime(signal["date"], utc=True)
        dataframe = dataframe.merge(signal[["date", "high_risk"]], on="date", how="left")
        # No prediction available for this candle (outside the walk-forward
        # evaluation range) -- default to "high risk" rather than "safe to
        # trade". A risk gate should fail toward caution under uncertainty.
        dataframe["high_risk"] = dataframe["high_risk"].fillna(1)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["high_risk"] == 0) & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["high_risk"] == 1) & (dataframe["volume"] > 0),
            "exit_long",
        ] = 1
        return dataframe
