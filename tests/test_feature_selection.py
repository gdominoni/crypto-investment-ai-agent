"""Tests for modules/module_c_volatility_ml/feature_selection.py."""

import numpy as np
import pandas as pd

from modules.module_c_volatility_ml.feature_selection import select_top_features


def test_selects_the_genuinely_informative_feature_over_noise():
    rng = np.random.default_rng(42)
    n = 500
    informative = rng.random(n)
    noise_1 = rng.random(n)
    noise_2 = rng.random(n)
    noise_3 = rng.random(n)
    y = (informative > 0.5).astype(int)

    X = pd.DataFrame(
        {"informative": informative, "noise_1": noise_1, "noise_2": noise_2, "noise_3": noise_3}
    )
    selected = select_top_features(X, pd.Series(y), top_n=1)
    assert selected == ["informative"]


def test_top_n_respected():
    rng = np.random.default_rng(7)
    n = 300
    X = pd.DataFrame({f"f{i}": rng.random(n) for i in range(6)})
    y = pd.Series((X["f0"] + X["f1"] > 1.0).astype(int))
    selected = select_top_features(X, y, top_n=3)
    assert len(selected) == 3
    assert set(selected).issubset(set(X.columns))
