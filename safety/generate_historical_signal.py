"""Runs the real safety/circuit_breaker.py::evaluate_circuit_breaker() across
the full historical 1h range for every pair Module B trades, and saves the
result as a time-series flag file.

This is *not* a new gate -- it's the existing Phase 3 deterministic circuit
breaker (ATR volatility spike + FOMC/CPI/NFP macro blackout), which until
now had only ever been called live, one timestamp at a time. Nothing here
changes its logic; this just replays it candle-by-candle over history so it
can be used to filter Module B's already-completed backtest trades.

Known, deliberate limitation carried over from the source data: the macro
calendar (safety/macro_calendar.py) only has 2026 FOMC/CPI/NFP dates, so the
macro-blackout half of the breaker cannot fire before 2026 -- not because
nothing happened in 2023-2025, but because no calendar exists for those
years. This is a real blind spot, not an oversight in this script.

Run locally with:
    python -m safety.generate_historical_signal
"""

from pathlib import Path

import pandas as pd

from safety.circuit_breaker import evaluate_circuit_breaker
from safety.limits import ATR_BASELINE_WINDOW, ATR_LOOKBACK_PERIODS
from safety.macro_calendar import MACRO_CALENDAR_2026

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "market" / "binance" / "spot"
OUTPUT_PATH = Path(__file__).resolve().parent / "signal_output" / "module_a_safe_hours.parquet"

PAIRS = {"BTC/USDT": "BTCUSDT_1h.parquet", "ETH/USDT": "ETHUSDT_1h.parquet"}

# Matches Module B's harness DATA_START/DATA_END exactly, so the signal
# covers precisely the range Module B's trades could have opened in.
RANGE_START = pd.Timestamp("2023-08-20", tz="UTC")
RANGE_END = pd.Timestamp("2026-08-17", tz="UTC")

# Trailing window handed to evaluate_circuit_breaker() at each candle. Only
# needs to cover ATR_LOOKBACK_PERIODS (14) + ATR_BASELINE_WINDOW (90) + 1 =
# 105 rows for check_volatility_spike() to have enough history; padded for
# margin. Kept small deliberately so the O(n) rolling call inside stays
# cheap across ~26k candles x 2 pairs.
WINDOW_ROWS = ATR_LOOKBACK_PERIODS + ATR_BASELINE_WINDOW + 20


def _generate_for_pair(pair: str, filename: str) -> pd.DataFrame:
    ohlc = pd.read_parquet(DATA_DIR / filename).set_index("timestamp")
    ohlc = ohlc.loc[(ohlc.index >= RANGE_START - pd.Timedelta(hours=WINDOW_ROWS)) & (ohlc.index <= RANGE_END)]

    rows = []
    for i in range(len(ohlc)):
        now = ohlc.index[i]
        if now < RANGE_START:
            continue
        window = ohlc.iloc[max(0, i - WINDOW_ROWS + 1) : i + 1]
        decision = evaluate_circuit_breaker(now.to_pydatetime(), window, MACRO_CALENDAR_2026)
        rows.append(
            {
                "date": now,
                "pair": pair,
                "atr_spike": "volatility_spike" in decision.reasons,
                "macro_blackout": any(r.startswith("macro_blackout") for r in decision.reasons),
                "unsafe": decision.should_liquidate,
                "reasons": ",".join(decision.reasons),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    frames = []
    for pair, filename in PAIRS.items():
        print(f"Evaluating circuit breaker for {pair}...")
        df = _generate_for_pair(pair, filename)
        unsafe_pct = df["unsafe"].mean()
        atr_pct = df["atr_spike"].mean()
        macro_pct = df["macro_blackout"].mean()
        print(
            f"  {len(df)} candles, {unsafe_pct:.1%} unsafe "
            f"(atr_spike={atr_pct:.1%}, macro_blackout={macro_pct:.1%})"
        )
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True).sort_values(["pair", "date"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH)
    print(f"\nSaved {len(combined)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
