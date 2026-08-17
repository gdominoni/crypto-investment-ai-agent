"""Module C: probabilistic volatility signal via FreqAI.

Predicts forward-looking realized volatility (std of returns over the next
`label_period_candles`) rather than price direction -- this module's job
is a volatility *read*, not picking direction on its own. The entry/exit
rules here exist only to exercise the pipeline end-to-end (enter when the
model expects calm, exit when it expects turbulence); they are not the
module's real decision logic.

Per the project spec, this signal is always subordinate to the
deterministic safety-kernel circuit breaker (safety/circuit_breaker.py) --
that gating happens at the orchestrator level (Phase 7/8), which will
check safety.circuit_breaker.evaluate_circuit_breaker() before ever
acting on this model's output live. FreqAI backtesting has no live
execution path, so there's nothing here for the circuit breaker to gate
yet; wiring it into backtests would be complexity with nothing to verify.

Feature set is restricted to price-derived technicals plus one curated
macro series (VIX, from Phase 2's data_ingestion/macro_data pipeline) --
matching the project's explicit instruction to keep Module C's feature
set small to protect signal-to-noise (see data_ingestion/macro_data/config.py).
"""

from pathlib import Path

import pandas as pd
import talib.abstract as ta
from freqai_utils import load_vix
from pandas import DataFrame

from freqtrade.strategy import IStrategy

MACRO_DATA_DIR = Path("/freqtrade/macro_data")


class VolatilityGateFreqAI(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False

    minimal_roi = {"0": 0.03}
    stoploss = -0.05

    process_only_new_candles = True
    startup_candle_count = 200

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: dict, **kwargs) -> DataFrame:
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour

        vix = load_vix(MACRO_DATA_DIR)
        dataframe = pd.merge_asof(dataframe.sort_values("date"), vix, on="date", direction="backward")
        dataframe["%-vix_close"] = dataframe["vix_close"]
        dataframe = dataframe.drop(columns=["vix_close"])
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        look_forward = self.freqai_info["feature_parameters"]["label_period_candles"]
        returns = dataframe["close"].pct_change()
        dataframe["&-target"] = returns.rolling(look_forward).std().shift(-look_forward)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["&-target"] < dataframe["&-target"].rolling(200).quantile(0.3))
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["&-target"] > dataframe["&-target"].rolling(200).quantile(0.7))
            & (dataframe["volume"] > 0),
            "exit_long",
        ] = 1
        return dataframe
