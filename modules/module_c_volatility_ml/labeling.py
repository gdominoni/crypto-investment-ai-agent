"""Triple Barrier labeling (Lopez de Prado), adapted for a non-directional
risk gate rather than a directional win/loss label.

The classic Triple Barrier method labels a *directional* bet: upper
barrier = profit target, lower barrier = stop loss, vertical barrier =
max holding period, label = whichever is touched first. Module C isn't
picking a direction -- it's a volatility/risk gate feeding the
orchestrator's caution level (see README.md) -- so there's no natural
"upper = good, lower = bad" here.

Adapted instead as a symmetric, non-directional formulation: barriers are
placed at +/- `barrier_atr_multiple` x ATR from the current close.
High-risk (label 1) if EITHER barrier is touched before the vertical
barrier expires -- i.e. a large move happened in either direction, which
is what a risk gate actually needs to flag. Calm (label 0) if price stays
within the band for the full vertical barrier window.
"""

import numpy as np
import pandas as pd


def triple_barrier_labels(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series,
    barrier_atr_multiple: float,
    vertical_barrier_candles: int,
) -> pd.Series:
    close_arr = close.to_numpy()
    high_arr = high.to_numpy()
    low_arr = low.to_numpy()
    atr_arr = atr.to_numpy()
    n = len(close_arr)
    labels = np.full(n, np.nan)

    for i in range(n - vertical_barrier_candles):
        if np.isnan(atr_arr[i]):
            continue
        upper = close_arr[i] + barrier_atr_multiple * atr_arr[i]
        lower = close_arr[i] - barrier_atr_multiple * atr_arr[i]
        window_high = high_arr[i + 1 : i + 1 + vertical_barrier_candles]
        window_low = low_arr[i + 1 : i + 1 + vertical_barrier_candles]
        touched = bool(np.any(window_high >= upper) or np.any(window_low <= lower))
        labels[i] = 1.0 if touched else 0.0

    return pd.Series(labels, index=close.index, name="high_risk_label")
