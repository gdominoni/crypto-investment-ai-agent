"""Purged, expanding-window walk-forward evaluation for Module C's
high-risk classifier.

Each fold has three parts, in order, not two -- this matters and is easy
to get wrong:

  [expanding training window] -> [purge] -> [calibration window] -> [purge] -> [trading week]

- Expanding training window: starts at the beginning of history every
  fold; only its end moves forward. More training data each week,
  simulating weekly retraining that never forgets earlier history (unlike
  Module B/the original Module C's fixed-length rolling window).
- Purge buffers: every label looks up to `vertical_barrier_candles`
  candles into the future (labeling.py). Any training or calibration
  sample within that many candles of the next window's start could have
  "seen" data from inside that next window through its own label -- those
  samples are dropped. Standard fix for label leakage in walk-forward CV
  with forward-looking labels (Lopez de Prado, purging).
- Calibration window: the model (already fit on the training window) is
  scored here, and threshold_calibration.find_optimal_threshold() picks
  T* from this window's precision-recall curve.
- Trading week: T* and the trained model are applied here, and this is
  what gets reported as the fold's real, forward-looking result --
  calibrating and evaluating on the *same* window would be circular
  (the threshold would be tuned with knowledge of the very data used to
  judge it), which is why calibration and reporting are different windows,
  not the same one twice.

The next fold's training window expands to include everything through
this fold's trading week -- a true expanding walk-forward across the
whole backtest range.
"""

from dataclasses import dataclass

import pandas as pd
from lightgbm import LGBMClassifier

from modules.module_c_volatility_ml.threshold_calibration import ThresholdResult, find_optimal_threshold


@dataclass(frozen=True)
class FoldResult:
    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp
    trading_start: pd.Timestamp
    trading_end: pd.Timestamp
    threshold: ThresholdResult
    trading_probabilities: pd.Series
    trading_predictions: pd.Series  # binary high-risk decision for the trading week


def purged_expanding_walk_forward(
    features: pd.DataFrame,
    labels: pd.Series,
    purge_candles: int,
    calibration_candles: int,
    trading_candles: int,
    min_train_candles: int,
    random_state: int = 42,
) -> list[FoldResult]:
    valid_idx = labels.dropna().index
    features = features.loc[valid_idx]
    labels = labels.loc[valid_idx]
    n = len(features)

    folds: list[FoldResult] = []
    fold_index = 0
    train_end_pos = min_train_candles

    while True:
        calibration_start_pos = train_end_pos + purge_candles
        calibration_end_pos = calibration_start_pos + calibration_candles
        trading_start_pos = calibration_end_pos + purge_candles
        trading_end_pos = trading_start_pos + trading_candles
        if trading_end_pos > n:
            break

        train_X = features.iloc[:train_end_pos]
        train_y = labels.iloc[:train_end_pos]
        calib_X = features.iloc[calibration_start_pos:calibration_end_pos]
        calib_y = labels.iloc[calibration_start_pos:calibration_end_pos]
        trading_X = features.iloc[trading_start_pos:trading_end_pos]

        n_pos = int((train_y == 1).sum())
        n_neg = int((train_y == 0).sum())
        pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

        model = LGBMClassifier(n_estimators=200, random_state=random_state, scale_pos_weight=pos_weight, verbosity=-1)
        model.fit(train_X, train_y)

        calib_proba = model.predict_proba(calib_X)[:, 1]
        threshold_result = find_optimal_threshold(calib_y.to_numpy(), calib_proba)

        trading_proba = model.predict_proba(trading_X)[:, 1]
        trading_predictions = (trading_proba >= threshold_result.threshold).astype(int)

        folds.append(
            FoldResult(
                fold_index=fold_index,
                train_start=train_X.index[0],
                train_end=train_X.index[-1],
                calibration_start=calib_X.index[0],
                calibration_end=calib_X.index[-1],
                trading_start=trading_X.index[0],
                trading_end=trading_X.index[-1],
                threshold=threshold_result,
                trading_probabilities=pd.Series(trading_proba, index=trading_X.index),
                trading_predictions=pd.Series(trading_predictions, index=trading_X.index),
            )
        )

        fold_index += 1
        train_end_pos = trading_end_pos  # expand training to include everything through this fold's trading week

    return folds
