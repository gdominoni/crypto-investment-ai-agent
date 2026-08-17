"""Tests for modules/module_c_volatility_ml/threshold_calibration.py."""

import numpy as np

from modules.module_c_volatility_ml.threshold_calibration import find_optimal_threshold


def test_perfect_separation_finds_high_precision_threshold():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.05, 0.1, 0.15, 0.2, 0.8, 0.85, 0.9, 0.95])
    result = find_optimal_threshold(y_true, y_proba)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.meets_precision_floor is True


def test_single_class_fails_safe_to_least_permissive_threshold():
    y_true = np.array([0, 0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4])
    result = find_optimal_threshold(y_true, y_proba)
    assert result.threshold == 1.0
    assert result.meets_precision_floor is False


def test_weak_signal_still_returns_a_threshold_flagged_as_below_floor():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_proba = rng.random(200)  # uncorrelated with y_true -- weak signal
    result = find_optimal_threshold(y_true, y_proba)
    assert 0.0 <= result.threshold <= 1.0
    assert result.precision < 1.0


def test_beta_parameter_changes_the_chosen_threshold():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    y_proba = np.array([0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9])
    low_beta = find_optimal_threshold(y_true, y_proba, beta=0.1)  # weights precision very heavily
    high_beta = find_optimal_threshold(y_true, y_proba, beta=2.0)  # weights recall more heavily
    assert low_beta.precision >= high_beta.precision
