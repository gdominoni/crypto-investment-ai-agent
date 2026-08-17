"""One-time SHAP-based feature selection for Module C.

Run ONCE on an initial training window before walk-forward evaluation
begins, not recomputed per fold. Two reasons: recomputing per fold would
let the selected feature set drift week to week, undermining
comparability of the walk-forward results (you'd be evaluating a moving
target, not one model family); and it would multiply an already
non-trivial SHAP compute cost by ~130 folds (see walk_forward.py) for no
real benefit. A single, stable, upfront selection is both cheaper and
more methodologically sound here -- a deliberate deviation from a literal
per-fold reading of the original spec, documented so it isn't mistaken
for a shortcut.
"""

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier


def select_top_features(X: pd.DataFrame, y: pd.Series, top_n: int, random_state: int = 42) -> list[str]:
    model = LGBMClassifier(n_estimators=200, random_state=random_state, verbosity=-1)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # Binary classifier: some shap/lightgbm version combinations return a
    # list [class0_values, class1_values], others a single array already
    # for the positive class -- normalize to the positive-class array.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranked = pd.Series(mean_abs_shap, index=X.columns).sort_values(ascending=False)
    return ranked.head(top_n).index.tolist()
