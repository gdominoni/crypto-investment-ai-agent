"""Per-fold dynamic threshold calibration for Module C's high-risk
classifier -- replaces a fixed 0.5 probability cutoff with a threshold
chosen to maximize F-beta=0.5 (weights precision over recall) on each
fold's calibration window.

The original spec said "maximizes F0.5 (or maintains Precision >= 80%)" --
ambiguous as two separate rules. Resolved to one: always maximize F-beta,
which already weights precision over recall by construction; the 80%
floor becomes a diagnostic flag (`meets_precision_floor`) rather than a
second, competing objective, so a weak fold degrades gracefully (still
gets its best available threshold) instead of blocking training.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import precision_recall_curve

PRECISION_FLOOR = 0.80  # diagnostic only -- see module docstring


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f_beta: float
    meets_precision_floor: bool


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray, beta: float = 0.5) -> ThresholdResult:
    if len(np.unique(y_true)) < 2:
        # Only one class present in this fold's calibration window -- no
        # PR curve is meaningful. Fail safe to the least permissive
        # threshold, matching this module's "when unsure, be cautious" bias.
        return ThresholdResult(threshold=1.0, precision=0.0, recall=0.0, f_beta=0.0, meets_precision_floor=False)

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns one more point than thresholds (the
    # recall=0 endpoint); drop it to align array lengths.
    precision, recall = precision[:-1], recall[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    f_beta = np.nan_to_num(f_beta, nan=0.0)

    best_idx = int(np.argmax(f_beta))
    return ThresholdResult(
        threshold=float(thresholds[best_idx]),
        precision=float(precision[best_idx]),
        recall=float(recall[best_idx]),
        f_beta=float(f_beta[best_idx]),
        meets_precision_floor=bool(precision[best_idx] >= PRECISION_FLOOR),
    )
