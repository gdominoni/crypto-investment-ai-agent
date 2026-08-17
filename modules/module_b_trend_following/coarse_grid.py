"""Timeframe-tailored coarse grids for Module B's Phase 9 Phase-1
screening (hyperopt strictly deferred to a possible Phase 2 -- this is a
hand-curated grid, not a search).

Each strategy family gets 3-4 hand-curated ENTRY presets *per timeframe*
(not shared across timeframes -- 15m needs fast-reacting settings, 4h
needs smoother, wider ones), and each timeframe gets 4 hand-curated EXIT
presets (SL/TP/trailing, including a "null" option to isolate
indicator-driven exits). Entry and exit grids are independent: every
entry preset for a family+timeframe is tested against every exit preset
for that timeframe.

Exit percentages scale with each timeframe's typical volatility per
holding period -- a 1.5% stop that's reasonable on 15m (where trades
often resolve within hours) would be far too tight on 4h (where a trade
might run for days), and a 4h-appropriate 7% stop would rarely ever
trigger on 15m, silently degrading into another "null" test.
"""

from dataclasses import dataclass

# NULL_SL/NULL_TP below must stay numerically in sync with
# user_data/strategies/dynamic_exit_mixin.py's NULL_STOPLOSS/NULL_ROI --
# duplicated rather than imported, since this file runs locally
# (orchestrating `docker compose run`) while the mixin runs inside the
# Freqtrade container, the same Docker/user_data-mount boundary behind
# data_ingestion/macro_data/loaders.py existing separately from Module
# C's freqai_utils.py.


@dataclass(frozen=True)
class ExitPreset:
    name: str
    stoploss: float  # negative fraction, e.g. -0.015 for 1.5%; NULL_STOPLOSS for "null"
    take_profit: float | None  # fraction, e.g. 0.02 for 2%; None for "null"
    trailing_stop: bool = False
    trailing_stop_positive: float | None = None
    trailing_stop_positive_offset: float | None = None


NULL_SL = -0.99
NULL_TP = None  # rendered as NULL_ROI at generation time


EXIT_GRIDS: dict[str, list[ExitPreset]] = {
    "15m": [
        ExitPreset("null_null", stoploss=NULL_SL, take_profit=NULL_TP),
        ExitPreset("tight", stoploss=-0.015, take_profit=0.02),
        ExitPreset("wide", stoploss=-0.025, take_profit=0.04),
        ExitPreset("sl_only_trailing", stoploss=-0.025, take_profit=NULL_TP,
                   trailing_stop=True, trailing_stop_positive=0.01, trailing_stop_positive_offset=0.015),
    ],
    "1h": [
        ExitPreset("null_null", stoploss=NULL_SL, take_profit=NULL_TP),
        ExitPreset("tight", stoploss=-0.02, take_profit=0.03),
        ExitPreset("wide", stoploss=-0.035, take_profit=0.06),
        ExitPreset("sl_only_trailing", stoploss=-0.035, take_profit=NULL_TP,
                   trailing_stop=True, trailing_stop_positive=0.015, trailing_stop_positive_offset=0.02),
    ],
    "4h": [
        ExitPreset("null_null", stoploss=NULL_SL, take_profit=NULL_TP),
        ExitPreset("tight", stoploss=-0.04, take_profit=0.08),
        ExitPreset("wide", stoploss=-0.07, take_profit=0.15),
        ExitPreset("sl_only_trailing", stoploss=-0.07, take_profit=NULL_TP,
                   trailing_stop=True, trailing_stop_positive=0.03, trailing_stop_positive_offset=0.04),
    ],
}


# Entry presets: dict[strategy_class_name][timeframe] -> list of buy_params dicts.
ENTRY_GRIDS: dict[str, dict[str, list[dict]]] = {
    "MeanReversionBBRSI": {
        "15m": [
            {"bb_period": 20, "bb_std": 1.8, "rsi_period": 10, "rsi_oversold": 25, "rsi_overbought": 75, "zscore_threshold": 1.8},
            {"bb_period": 20, "bb_std": 2.0, "rsi_period": 10, "rsi_oversold": 20, "rsi_overbought": 80, "zscore_threshold": 2.0},
            {"bb_period": 20, "bb_std": 2.2, "rsi_period": 14, "rsi_oversold": 25, "rsi_overbought": 75, "zscore_threshold": 2.2},
        ],
        "1h": [
            {"bb_period": 20, "bb_std": 2.0, "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70, "zscore_threshold": 2.0},
            {"bb_period": 20, "bb_std": 2.2, "rsi_period": 14, "rsi_oversold": 25, "rsi_overbought": 75, "zscore_threshold": 2.2},
            {"bb_period": 30, "bb_std": 2.5, "rsi_period": 21, "rsi_oversold": 30, "rsi_overbought": 70, "zscore_threshold": 2.5},
        ],
        "4h": [
            {"bb_period": 20, "bb_std": 2.5, "rsi_period": 14, "rsi_oversold": 35, "rsi_overbought": 65, "zscore_threshold": 2.5},
            {"bb_period": 20, "bb_std": 3.0, "rsi_period": 21, "rsi_oversold": 30, "rsi_overbought": 70, "zscore_threshold": 3.0},
            {"bb_period": 50, "bb_std": 3.0, "rsi_period": 21, "rsi_oversold": 35, "rsi_overbought": 65, "zscore_threshold": 3.0},
        ],
    },
    "VolatilityBreakoutKCSqueeze": {
        "15m": [
            {"bb_period": 20, "bb_std": 2.0, "kc_period": 20, "kc_atr_mult": 1.5, "momentum_period": 12},
            {"bb_period": 20, "bb_std": 2.0, "kc_period": 20, "kc_atr_mult": 2.0, "momentum_period": 20},
            {"bb_period": 14, "bb_std": 1.8, "kc_period": 14, "kc_atr_mult": 1.5, "momentum_period": 12},
        ],
        "1h": [
            {"bb_period": 20, "bb_std": 2.0, "kc_period": 20, "kc_atr_mult": 1.5, "momentum_period": 20},
            {"bb_period": 20, "bb_std": 2.0, "kc_period": 20, "kc_atr_mult": 2.0, "momentum_period": 30},
            {"bb_period": 30, "bb_std": 2.2, "kc_period": 30, "kc_atr_mult": 2.0, "momentum_period": 20},
        ],
        "4h": [
            {"bb_period": 20, "bb_std": 2.0, "kc_period": 20, "kc_atr_mult": 2.0, "momentum_period": 20},
            {"bb_period": 20, "bb_std": 2.5, "kc_period": 20, "kc_atr_mult": 2.5, "momentum_period": 30},
            {"bb_period": 50, "bb_std": 2.5, "kc_period": 50, "kc_atr_mult": 2.5, "momentum_period": 30},
        ],
    },
    "VolumeDrivenVWAPCMF": {
        "15m": [
            {"vwap_period": 48, "volume_surge_period": 20, "volume_surge_multiple": 1.5, "cmf_period": 14, "cmf_threshold": 0.05},
            {"vwap_period": 96, "volume_surge_period": 20, "volume_surge_multiple": 2.0, "cmf_period": 20, "cmf_threshold": 0.1},
            {"vwap_period": 48, "volume_surge_period": 10, "volume_surge_multiple": 2.0, "cmf_period": 14, "cmf_threshold": 0.1},
        ],
        "1h": [
            {"vwap_period": 24, "volume_surge_period": 20, "volume_surge_multiple": 1.5, "cmf_period": 20, "cmf_threshold": 0.05},
            {"vwap_period": 72, "volume_surge_period": 20, "volume_surge_multiple": 2.0, "cmf_period": 20, "cmf_threshold": 0.1},
            {"vwap_period": 168, "volume_surge_period": 24, "volume_surge_multiple": 2.0, "cmf_period": 20, "cmf_threshold": 0.1},
        ],
        "4h": [
            {"vwap_period": 42, "volume_surge_period": 20, "volume_surge_multiple": 1.5, "cmf_period": 20, "cmf_threshold": 0.05},
            {"vwap_period": 126, "volume_surge_period": 20, "volume_surge_multiple": 2.0, "cmf_period": 20, "cmf_threshold": 0.1},
            {"vwap_period": 180, "volume_surge_period": 30, "volume_surge_multiple": 2.0, "cmf_period": 30, "cmf_threshold": 0.1},
        ],
    },
}

TIMEFRAMES = ["15m", "1h", "4h"]
STRATEGIES = ["MeanReversionBBRSI", "VolatilityBreakoutKCSqueeze", "VolumeDrivenVWAPCMF"]
