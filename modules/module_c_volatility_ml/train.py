"""Module C training pipeline (Phase 9): Triple Barrier labeling, one-time
SHAP feature selection, and purged expanding-window walk-forward
evaluation with per-fold dynamic threshold calibration.

Runs entirely locally, no Docker/FreqAI required -- one concrete benefit
of the bespoke-pipeline decision (see README.md): training is now just
pandas/lightgbm/shap on this machine. Docker/Freqtrade is only used
afterward, to turn the resulting signal into realistic simulated trades
(fees, ROI, stoploss) via a thin strategy that reads this script's output.

Run locally with:
    python -m modules.module_c_volatility_ml.train
"""

from pathlib import Path

import pandas as pd
import pandas_ta as pta

from data_ingestion.macro_data.loaders import load_vix
from modules.module_c_volatility_ml.feature_selection import select_top_features
from modules.module_c_volatility_ml.labeling import triple_barrier_labels
from modules.module_c_volatility_ml.walk_forward import purged_expanding_walk_forward
from safety.circuit_breaker import compute_atr

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BTC_1H_PATH = REPO_ROOT / "data" / "market" / "binance" / "spot" / "BTCUSDT_1h.parquet"
MACRO_DATA_DIR = REPO_ROOT / "data" / "macro"
SIGNAL_OUTPUT_PATH = Path(__file__).resolve().parent / "user_data" / "signal_output" / "high_risk_signal.parquet"

# Triple Barrier
# BARRIER_ATR_MULTIPLE was initially set to a "textbook" 2.0 -- checked
# empirically (see decisions-log.md) and found it produced an 86%
# high-risk base rate at a 24h vertical barrier: for a diffusive price
# process, first-passage-time math means a narrow barrier is touched
# almost certainly given enough candles, so the "gate" degenerated into
# an always-on signal. Widened to 4.5x, giving a ~41% base rate -- swept
# empirically while holding the vertical barrier fixed at the agreed 24h.
BARRIER_ATR_MULTIPLE = 4.5
VERTICAL_BARRIER_CANDLES = 24  # 1 day at 1h timeframe

# Purged expanding walk-forward
PURGE_CANDLES = VERTICAL_BARRIER_CANDLES  # standard Lopez de Prado sizing: purge >= label horizon
CALIBRATION_CANDLES = 168  # 1 week
TRADING_CANDLES = 168  # 1 week
MIN_TRAIN_CANDLES = 180 * 24  # 6 months

TOP_N_FEATURES = 5


def build_features_and_labels() -> tuple[pd.DataFrame, pd.Series]:
    ohlc = pd.read_parquet(BTC_1H_PATH)
    ohlc = ohlc.set_index("timestamp") if "timestamp" in ohlc.columns else ohlc

    atr = compute_atr(ohlc)
    features = pd.DataFrame(index=ohlc.index)
    features["rsi"] = pta.rsi(ohlc["close"], length=14)
    adx = pta.adx(ohlc["high"], ohlc["low"], ohlc["close"], length=14)
    features["adx"] = adx["ADX_14"]
    features["atr"] = atr
    features["returns"] = ohlc["close"].pct_change()
    features["volume"] = ohlc["volume"]
    features["day_of_week"] = ohlc.index.dayofweek
    features["hour_of_day"] = ohlc.index.hour

    vix = load_vix(MACRO_DATA_DIR)
    ohlc_with_date = ohlc.reset_index().rename(columns={ohlc.index.name or "index": "date"})
    merged = pd.merge_asof(ohlc_with_date.sort_values("date"), vix, on="date", direction="backward")
    features["vix_close"] = merged["vix_close"].to_numpy()

    labels = triple_barrier_labels(
        close=ohlc["close"],
        high=ohlc["high"],
        low=ohlc["low"],
        atr=atr,
        barrier_atr_multiple=BARRIER_ATR_MULTIPLE,
        vertical_barrier_candles=VERTICAL_BARRIER_CANDLES,
    )

    valid = features.dropna().index.intersection(labels.dropna().index)
    return features.loc[valid], labels.loc[valid]


def main() -> None:
    print("Building features and Triple Barrier labels...")
    features, labels = build_features_and_labels()
    print(f"{len(features)} labeled candles, {labels.mean():.1%} high-risk")

    print(f"\nSelecting top {TOP_N_FEATURES} features via SHAP (once, on the first training window)...")
    initial_window = features.iloc[:MIN_TRAIN_CANDLES]
    initial_labels = labels.iloc[:MIN_TRAIN_CANDLES]
    selected_features = select_top_features(initial_window, initial_labels, top_n=TOP_N_FEATURES)
    print(f"Selected: {selected_features}")

    print("\nRunning purged expanding-window walk-forward evaluation...")
    folds = purged_expanding_walk_forward(
        features[selected_features],
        labels,
        purge_candles=PURGE_CANDLES,
        calibration_candles=CALIBRATION_CANDLES,
        trading_candles=TRADING_CANDLES,
        min_train_candles=MIN_TRAIN_CANDLES,
    )
    print(f"{len(folds)} folds")

    if not folds:
        print("No folds produced -- not enough data for the configured window sizes.")
        return

    precision_floor_hits = sum(f.threshold.meets_precision_floor for f in folds)
    avg_precision = sum(f.threshold.precision for f in folds) / len(folds)
    avg_recall = sum(f.threshold.recall for f in folds) / len(folds)
    avg_f_beta = sum(f.threshold.f_beta for f in folds) / len(folds)
    print(f"Avg calibration precision: {avg_precision:.2f}, recall: {avg_recall:.2f}, F0.5: {avg_f_beta:.2f}")
    print(f"Folds meeting {80}% precision floor: {precision_floor_hits}/{len(folds)}")

    all_predictions = pd.concat([f.trading_predictions for f in folds]).sort_index()
    all_probabilities = pd.concat([f.trading_probabilities for f in folds]).sort_index()
    trading_true_labels = labels.loc[all_predictions.index]

    tp = int(((all_predictions == 1) & (trading_true_labels == 1)).sum())
    fp = int(((all_predictions == 1) & (trading_true_labels == 0)).sum())
    fn = int(((all_predictions == 0) & (trading_true_labels == 1)).sum())
    overall_precision = tp / (tp + fp) if (tp + fp) else 0.0
    overall_recall = tp / (tp + fn) if (tp + fn) else 0.0
    print(
        f"\nOverall out-of-sample (all trading weeks combined): "
        f"precision={overall_precision:.2f} recall={overall_recall:.2f} "
        f"flagged high-risk {all_predictions.mean():.1%} of the time"
    )

    SIGNAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    signal_df = pd.DataFrame({"date": all_predictions.index, "high_risk": all_predictions.to_numpy(), "probability": all_probabilities.to_numpy()})
    signal_df.to_parquet(SIGNAL_OUTPUT_PATH)
    print(f"\nSaved signal for {len(signal_df)} candles to {SIGNAL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
