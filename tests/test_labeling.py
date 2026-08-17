"""Tests for modules/module_c_volatility_ml/labeling.py."""

import numpy as np
import pandas as pd

from modules.module_c_volatility_ml.labeling import triple_barrier_labels


def test_flat_price_within_atr_band_is_calm():
    n = 30
    close = pd.Series([100.0] * n)
    high = pd.Series([100.5] * n)
    low = pd.Series([99.5] * n)
    atr = pd.Series([1.0] * n)
    labels = triple_barrier_labels(close, high, low, atr, barrier_atr_multiple=2.0, vertical_barrier_candles=5)
    assert (labels.dropna() == 0.0).all()


def test_upper_barrier_touch_flags_high_risk():
    n = 30
    close = pd.Series([100.0] * n)
    high = pd.Series([100.5] * n)
    low = pd.Series([99.5] * n)
    atr = pd.Series([1.0] * n)
    # Blow out the high right after candle 0, within its lookforward window.
    high.iloc[2] = 103.0
    labels = triple_barrier_labels(close, high, low, atr, barrier_atr_multiple=2.0, vertical_barrier_candles=5)
    assert labels.iloc[0] == 1.0


def test_lower_barrier_touch_flags_high_risk_symmetrically():
    n = 30
    close = pd.Series([100.0] * n)
    high = pd.Series([100.5] * n)
    low = pd.Series([99.5] * n)
    atr = pd.Series([1.0] * n)
    low.iloc[2] = 97.0
    labels = triple_barrier_labels(close, high, low, atr, barrier_atr_multiple=2.0, vertical_barrier_candles=5)
    assert labels.iloc[0] == 1.0


def test_last_rows_within_vertical_barrier_are_unlabeled():
    n = 10
    close = pd.Series([100.0] * n)
    high = pd.Series([100.5] * n)
    low = pd.Series([99.5] * n)
    atr = pd.Series([1.0] * n)
    labels = triple_barrier_labels(close, high, low, atr, barrier_atr_multiple=2.0, vertical_barrier_candles=5)
    assert labels.iloc[-5:].isna().all()


def test_missing_atr_leaves_row_unlabeled():
    n = 20
    close = pd.Series([100.0] * n)
    high = pd.Series([100.5] * n)
    low = pd.Series([99.5] * n)
    atr = pd.Series([np.nan] * 5 + [1.0] * 15)
    labels = triple_barrier_labels(close, high, low, atr, barrier_atr_multiple=2.0, vertical_barrier_candles=5)
    assert labels.iloc[:5].isna().all()
