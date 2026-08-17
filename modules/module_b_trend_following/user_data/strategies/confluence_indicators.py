"""Shared, pure indicator-computation helpers for the 8 multi-factor
confluence combo strategies (Phase 9, 1h hyperopt harness). Each combo
strategy calls exactly 3 of these (one per family) with its own
hyperopted parameter values -- kept here once, not duplicated 8 times.

Family 2 and Family 3 are now independent alternatives (Price BB *or*
Keltner; Volume Surge *or* CMF), not combined as in the earlier squeeze/
dual-volume design -- see decisions-log.md. Periods not listed as
hyperopt dimensions in the spec (rsi_bb_period, bb_period, kc_period,
volume lookback, cmf_period) are fixed constants here, matching the
spec's explicit parameter list.
"""

import pandas_ta as pta
from pandas import DataFrame

_RSI_BB_PERIOD = 20
_PRICE_BB_PERIOD = 20
_KC_PERIOD = 20
_VOLUME_LOOKBACK = 20
_CMF_PERIOD = 20


def classic_rsi(dataframe: DataFrame, rsi_period: int) -> DataFrame:
    dataframe["rsi"] = pta.rsi(dataframe["close"], length=rsi_period)
    return dataframe


def rsi_bollinger_bands(dataframe: DataFrame, rsi_period: int, rsi_bb_std: float) -> DataFrame:
    dataframe["rsi"] = pta.rsi(dataframe["close"], length=rsi_period)
    bb = pta.bbands(dataframe["rsi"], length=_RSI_BB_PERIOD, std=rsi_bb_std)
    dataframe["rsi_bb_lower"] = bb.iloc[:, 0]
    dataframe["rsi_bb_upper"] = bb.iloc[:, 2]
    return dataframe


def price_bollinger_bands(dataframe: DataFrame, bb_std: float) -> DataFrame:
    bb = pta.bbands(dataframe["close"], length=_PRICE_BB_PERIOD, std=bb_std)
    dataframe["price_bb_mid"] = bb.iloc[:, 1]
    dataframe["price_bb_upper"] = bb.iloc[:, 2]
    return dataframe


def keltner_channel(dataframe: DataFrame, kc_atr_mult: float) -> DataFrame:
    kc = pta.kc(dataframe["high"], dataframe["low"], dataframe["close"], length=_KC_PERIOD, scalar=kc_atr_mult)
    dataframe["kc_mid"] = kc.iloc[:, 1]
    dataframe["kc_upper"] = kc.iloc[:, 2]
    return dataframe


def volume_surge(dataframe: DataFrame, volume_surge_mult: float) -> DataFrame:
    avg_volume = dataframe["volume"].rolling(_VOLUME_LOOKBACK).mean()
    dataframe["volume_surge"] = dataframe["volume"] > (avg_volume * volume_surge_mult)
    return dataframe


def chaikin_money_flow(dataframe: DataFrame, cmf_threshold: float) -> DataFrame:
    dataframe["cmf"] = pta.cmf(
        dataframe["high"], dataframe["low"], dataframe["close"], dataframe["volume"], length=_CMF_PERIOD
    )
    dataframe["cmf_positive"] = dataframe["cmf"] > cmf_threshold
    return dataframe
