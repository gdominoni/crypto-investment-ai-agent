"""Tests for modules/module_c_volatility_ml/walk_forward.py -- focused on
getting the fold structure itself right (purge gaps, expanding windows,
non-overlapping trading weeks), since that's the part most likely to
silently leak information if it's wrong.
"""

import numpy as np
import pandas as pd
import pytest

from modules.module_c_volatility_ml.walk_forward import purged_expanding_walk_forward


@pytest.fixture
def synthetic_dataset():
    rng = np.random.default_rng(42)
    n = 1000
    index = pd.RangeIndex(n)
    features = pd.DataFrame({"f1": rng.random(n), "f2": rng.random(n)}, index=index)
    # Roughly balanced, weakly-informative labels -- enough for LightGBM to
    # fit without hitting the single-class fail-safe path in every fold.
    labels = pd.Series((features["f1"] + rng.random(n) > 1.0).astype(int), index=index)
    return features, labels


def test_produces_at_least_one_fold_with_enough_data(synthetic_dataset):
    features, labels = synthetic_dataset
    folds = purged_expanding_walk_forward(
        features, labels, purge_candles=5, calibration_candles=50, trading_candles=50, min_train_candles=300
    )
    assert len(folds) > 0


def test_training_window_expands_across_folds(synthetic_dataset):
    features, labels = synthetic_dataset
    folds = purged_expanding_walk_forward(
        features, labels, purge_candles=5, calibration_candles=50, trading_candles=50, min_train_candles=300
    )
    train_ends = [f.train_end for f in folds]
    assert train_ends == sorted(train_ends)
    assert len(set(train_ends)) == len(train_ends)  # strictly increasing, not just non-decreasing
    assert all(f.train_start == folds[0].train_start for f in folds)  # start never moves


def test_purge_gap_is_respected_between_train_and_calibration(synthetic_dataset):
    features, labels = synthetic_dataset
    purge = 7
    folds = purged_expanding_walk_forward(
        features, labels, purge_candles=purge, calibration_candles=50, trading_candles=50, min_train_candles=300
    )
    for fold in folds:
        gap = fold.calibration_start - fold.train_end
        assert gap == purge + 1  # +1 because train_end is the last *included* training position


def test_purge_gap_is_respected_between_calibration_and_trading(synthetic_dataset):
    features, labels = synthetic_dataset
    purge = 7
    folds = purged_expanding_walk_forward(
        features, labels, purge_candles=purge, calibration_candles=50, trading_candles=50, min_train_candles=300
    )
    for fold in folds:
        gap = fold.trading_start - fold.calibration_end
        assert gap == purge + 1


def test_trading_weeks_do_not_overlap_across_folds(synthetic_dataset):
    features, labels = synthetic_dataset
    folds = purged_expanding_walk_forward(
        features, labels, purge_candles=5, calibration_candles=50, trading_candles=50, min_train_candles=300
    )
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.trading_end < later.trading_start


def test_no_folds_when_data_shorter_than_one_full_fold():
    features = pd.DataFrame({"f1": [0.1] * 50, "f2": [0.2] * 50})
    labels = pd.Series([0, 1] * 25)
    folds = purged_expanding_walk_forward(
        features, labels, purge_candles=5, calibration_candles=50, trading_candles=50, min_train_candles=300
    )
    assert folds == []


def test_trading_predictions_are_binary(synthetic_dataset):
    features, labels = synthetic_dataset
    folds = purged_expanding_walk_forward(
        features, labels, purge_candles=5, calibration_candles=50, trading_candles=50, min_train_candles=300
    )
    for fold in folds:
        assert set(fold.trading_predictions.unique()).issubset({0, 1})
