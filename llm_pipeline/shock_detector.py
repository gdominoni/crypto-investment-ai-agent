"""Real-time shock classification. NO LONGER A TRIGGER -- retained as a
reading.

`scan_for_shocks` used to escalate straight to Sonnet on every hourly pass.
That path was removed in favour of `compression_detector.py`, for two reasons
measured in forecast/trigger_value.py: a shock is the OUTCOME this project
looks for the causes of, and post-shock days are followed by a defined trend
LESS often than ordinary days -- so it selected against the thing being sought.

It also carried a defect worth recording: it tested the volatility STATE, not a
transition, so a multi-day shock re-escalated the same market to Sonnet every
hour the daemon ran. The replay had guarded against this with a transition
check; production never did.

What remains here is the classification itself, still used to label regimes and
to exclude extreme events from the static battery's fitting.

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
