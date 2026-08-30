"""Real-time market-data shock detection -- the counterpart to Haiku's
news-based magnitude scoring, and the trigger for this project's second
stated goal: can the LLM layer recognize and react to a crash or a bull
surge as it's actually happening, not just process routine headlines.

Reuses `candidates/methodology.py::shock_zscore_series` directly (the
exact same statistical definition of 'shock' Phase 1's historical
research uses to exclude extreme events from the static battery's
fitting) so 'shock' means one consistent thing everywhere in this
project, not a second, drifted definition invented for the live path.
"""
from __future__ import annotations

from candidates.data_loading import load_daily
from candidates.methodology import shock_zscore_series

SHOCK_ZSCORE_THRESHOLD = 2.0


def current_shock_status(symbol: str, threshold: float = SHOCK_ZSCORE_THRESHOLD) -> dict:
    daily = load_daily(symbol)
    z_series = shock_zscore_series(daily)
    if len(z_series) == 0:
        return {"symbol": symbol, "is_shock": False, "shock_z": None, "direction": None}
    z = z_series.iloc[-1]
    if z != z:  # NaN check without importing numpy for one comparison
        return {"symbol": symbol, "is_shock": False, "shock_z": None, "direction": None}
    is_shock = z >= threshold
    latest_return = daily["close"].pct_change().iloc[-1]
    direction = None
    if is_shock:
        direction = "crash" if latest_return < 0 else "surge"
    return {"symbol": symbol, "is_shock": bool(is_shock), "shock_z": float(z), "direction": direction,
            "latest_return": float(latest_return) if latest_return == latest_return else None}


def scan_for_shocks(coins: list[str], threshold: float = SHOCK_ZSCORE_THRESHOLD) -> list[dict]:
    return [s for c in coins if (s := current_shock_status(c, threshold))["is_shock"]]
