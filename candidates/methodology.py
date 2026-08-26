"""Core event-study methodology shared by every candidate in the battery.

Three invariants are enforced structurally here, not left to each
candidate's own discipline:

1. Entry is always the bar AFTER the one a trigger condition examines.
   A trigger that reads a bar's own close cannot fire an entry at that
   bar's close -- there is no code path in this module that allows it.
2. Exit barriers are duration-bucketed anchors (mean MFE / mean |MAE| per
   horizon, fit from a train set only) -- there is no flat-percentage
   barrier option.
3. `report()` always returns win_rate, strict_win_rate, sortino, and
   timeout_fraction together. Nothing in this module returns a win rate
   on its own.

Works on any OHLC frame indexed by bar (daily or hourly); `horizons` is
expressed in bar counts, not calendar time, so the same functions serve
both granularities.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MethodologyConfig:
    horizons: tuple[int, ...]
    tp_mult_grid: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.5)
    sl_mult_grid: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.5)
    round_trip_fee: float = 0.002
    min_train_events: int = 20
    min_report_events: int = 30


def shock_zscore_series(ohlc: pd.DataFrame, short_window: int = 5, baseline_window: int = 252) -> pd.Series:
    """Rolling z-score of short-term realized volatility against the
    coin's own longer trailing distribution of that same short-term
    volatility -- a 'vol-of-vol' anomaly measure. Every value at position
    `loc` uses only bars up to and including `loc` (pandas rolling is
    backward-looking by construction), so classifying a bar as a shock
    never uses information from after that bar."""
    returns = ohlc["close"].pct_change()
    short_vol = returns.rolling(short_window, min_periods=short_window).std()
    baseline_mean = short_vol.rolling(baseline_window, min_periods=max(baseline_window // 4, short_window)).mean()
    baseline_std = short_vol.rolling(baseline_window, min_periods=max(baseline_window // 4, short_window)).std()
    return (short_vol - baseline_mean) / baseline_std


def classify_regime(shock_z: pd.Series, loc: int, shock_threshold: float = 3.0) -> str:
    """'shock' if the trigger bar's own short-term volatility is an
    extreme outlier (>= shock_threshold std devs) relative to the coin's
    own recent history; 'normal' otherwise, including whenever there
    isn't yet enough history to judge -- classifying early history as a
    shock by default would be a guess, not a measurement."""
    z = shock_z.iloc[loc] if loc < len(shock_z) else np.nan
    return "shock" if (pd.notna(z) and z >= shock_threshold) else "normal"


def build_events(ohlc: pd.DataFrame, trigger: pd.Series, direction: str, horizons: tuple[int, ...],
                  shock_z: pd.Series | None = None, shock_threshold: float = 3.0) -> pd.DataFrame:
    """One row per triggered bar. `entry_price`/`entry_loc` are always the
    NEXT bar's open -- the only causally available price once a trigger
    fires on the current bar. MFE/MAE at each horizon are measured over
    bars strictly after entry.

    If `shock_z` is provided, each row also carries a `regime` tag
    ('normal'/'shock') classified at the TRIGGER bar. This project's
    static candidate battery only ever fits and trades on 'normal'-regime
    events (filtered by the caller, e.g. `run_battery.py`) -- extreme
    historical shocks are deliberately excluded from what a fixed rule
    set is fit and graded against, so a handful of crash days can't
    distort the barriers applied to ordinary conditions. Shock-regime
    events are not discarded; they're routed to the live shock-detection
    pathway (`llm_pipeline/shock_detector.py`) instead of the weekly-
    refreshed static battery."""
    assert direction in ("long", "short")
    idx = ohlc.index
    trigger_locs = np.flatnonzero(trigger.reindex(idx, fill_value=False).to_numpy())
    max_h = max(horizons)
    rows = []
    for loc in trigger_locs:
        entry_loc = loc + 1
        if entry_loc >= len(idx) or entry_loc + max_h >= len(idx):
            continue
        entry_price = ohlc["open"].iloc[entry_loc]
        if pd.isna(entry_price) or entry_price <= 0:
            continue
        row = {"trigger_time": idx[loc], "entry_time": idx[entry_loc], "entry_loc": entry_loc, "entry_price": entry_price}
        row["regime"] = classify_regime(shock_z, loc, shock_threshold) if shock_z is not None else "normal"
        for h in horizons:
            seg_hi = ohlc["high"].iloc[entry_loc + 1: entry_loc + 1 + h]
            seg_lo = ohlc["low"].iloc[entry_loc + 1: entry_loc + 1 + h]
            if len(seg_hi) == 0:
                row[f"mfe_{h}"], row[f"mae_{h}"] = np.nan, np.nan
                continue
            if direction == "long":
                row[f"mfe_{h}"] = seg_hi.max() / entry_price - 1
                row[f"mae_{h}"] = seg_lo.min() / entry_price - 1
            else:
                row[f"mfe_{h}"] = 1 - seg_lo.min() / entry_price
                row[f"mae_{h}"] = 1 - seg_hi.max() / entry_price
        rows.append(row)
    events = pd.DataFrame(rows)
    if len(events):
        events["direction"] = direction
    return events


def compute_anchors(events: pd.DataFrame, horizons: tuple[int, ...]) -> dict[int, dict[str, float]]:
    return {h: {"mfe": events[f"mfe_{h}"].mean(), "mae": events[f"mae_{h}"].abs().mean()} for h in horizons}


def bucket_for_elapsed(elapsed_periods: int, horizons: tuple[int, ...]) -> int | None:
    """Which duration bucket applies right now, given how many periods
    have elapsed since entry -- shared by the batch simulator below and
    the live execution engine's per-candle exit check, so both walk the
    exact same ladder."""
    for h in horizons:
        if elapsed_periods <= h:
            return h
    return None  # past the last bucket -- caller's timeout path


def barrier_prices(entry_price: float, direction: str, anchors: dict, horizon: int,
                    tp_mult: float, sl_mult: float) -> tuple[float, float, float, float]:
    """Returns (tp_price, sl_price, tp_magnitude, sl_magnitude)."""
    tp_mag, sl_mag = anchors[horizon]["mfe"] * tp_mult, anchors[horizon]["mae"] * sl_mult
    if direction == "long":
        return entry_price * (1 + tp_mag), entry_price * (1 - sl_mag), tp_mag, sl_mag
    return entry_price * (1 - tp_mag), entry_price * (1 + sl_mag), tp_mag, sl_mag


def simulate_trade(entry_loc: int, entry_price: float, ohlc: pd.DataFrame, direction: str,
                    anchors: dict, tp_mult: float, sl_mult: float, horizons: tuple[int, ...],
                    fee: float) -> tuple[str, float]:
    """Walks the duration buckets in order; the SL check is evaluated
    before the TP check within any single bar where a daily-resolution
    frame can't otherwise disambiguate same-bar ordering -- a deliberate,
    conservative tie-break, not an oversight."""
    idx = ohlc.index
    prev_bound = 0
    for h in horizons:
        tp_price, sl_price, tp_mag, sl_mag = barrier_prices(entry_price, direction, anchors, h, tp_mult, sl_mult)
        seg_start, seg_end = entry_loc + 1 + prev_bound, min(entry_loc + 1 + h, len(idx))
        if seg_start < seg_end:
            for hi, lo in zip(ohlc["high"].iloc[seg_start:seg_end], ohlc["low"].iloc[seg_start:seg_end]):
                sl_hit = (lo <= sl_price) if direction == "long" else (hi >= sl_price)
                tp_hit = (hi >= tp_price) if direction == "long" else (lo <= tp_price)
                if sl_hit:
                    return "loss", -sl_mag - fee
                if tp_hit:
                    return "win", tp_mag - fee
        prev_bound = h
    last_loc = min(entry_loc + 1 + horizons[-1], len(idx) - 1)
    mid = (ohlc["high"].iloc[last_loc] + ohlc["low"].iloc[last_loc]) / 2
    raw = (mid / entry_price - 1) if direction == "long" else (1 - mid / entry_price)
    return "timeout", raw - fee


def sortino_ratio(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    """Semi-deviation over the full sample (not just the losing subset) --
    a sample std of just the negative returns can collapse toward zero
    when many losses land on an identical barrier value, blowing the
    ratio up to nonsense."""
    returns = np.asarray(returns, dtype=float)
    downside_dev = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2))
    return float(returns.mean() / downside_dev * np.sqrt(periods_per_year)) if downside_dev > 0 else float("nan")


def walk_forward(events: pd.DataFrame, ohlc_by_group: dict[str, pd.DataFrame], direction: str,
                  cfg: MethodologyConfig, min_train_periods: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window, refit every fold on periods strictly before the
    test period. `events` must carry a `period` column (e.g. year) and a
    `group` column (e.g. coin) so anchors/multipliers are fit pooled but
    graded per-group afterward."""
    periods = sorted(events["period"].unique())
    if len(periods) <= min_train_periods:
        return pd.DataFrame(), pd.DataFrame()
    oos_rows, params_log = [], []
    for test_period in periods[min_train_periods:]:
        train = events[events["period"] < test_period]
        test = events[events["period"] == test_period]
        if len(train) < cfg.min_train_events or len(test) == 0:
            continue
        anchors = compute_anchors(train, cfg.horizons)
        best = None
        for tp_mult in cfg.tp_mult_grid:
            for sl_mult in cfg.sl_mult_grid:
                rets = [simulate_trade(r.entry_loc, r.entry_price, ohlc_by_group[r.group], direction,
                                        anchors, tp_mult, sl_mult, cfg.horizons, cfg.round_trip_fee)[1]
                        for r in train.itertuples()]
                s = sortino_ratio(np.array(rets))
                if best is None or (not np.isnan(s) and s > best[0]):
                    best = (s, tp_mult, sl_mult)
        _, tp_mult, sl_mult = best
        params_log.append({"test_period": test_period, "tp_mult": tp_mult, "sl_mult": sl_mult, "n_train": len(train)})
        for r in test.itertuples():
            outcome, ret = simulate_trade(r.entry_loc, r.entry_price, ohlc_by_group[r.group], direction,
                                           anchors, tp_mult, sl_mult, cfg.horizons, cfg.round_trip_fee)
            oos_rows.append({"group": r.group, "period": test_period, "trigger_time": r.trigger_time,
                              "outcome": outcome, "net_return": ret})
    return pd.DataFrame(oos_rows), pd.DataFrame(params_log)


def report(oos: pd.DataFrame) -> dict:
    """`total_expectancy` sums per-trade returns rather than compounding
    them geometrically -- this system runs several coins/candidates with
    overlapping open positions, not one account staking 100% into a
    single sequential trade chain, so geometric compounding of hundreds
    of pooled events produces a number with no realistic interpretation.
    A fixed-fractional-stake sum is the honest analogue of "total
    expectancy across N independently-sized trades.\""""
    if len(oos) == 0:
        return {"n": 0, "win_rate": np.nan, "strict_win_rate": np.nan, "sortino": np.nan,
                "total_expectancy": np.nan, "timeout_fraction": np.nan}
    n = len(oos)
    wins = int((oos.outcome == "win").sum())
    losses = int((oos.outcome == "loss").sum())
    timeouts = int((oos.outcome == "timeout").sum())
    return {
        "n": n,
        "win_rate": wins / (wins + losses) if (wins + losses) else np.nan,
        "strict_win_rate": wins / n,
        "sortino": sortino_ratio(oos["net_return"].to_numpy()),
        "total_expectancy": float(oos["net_return"].sum()),
        "timeout_fraction": timeouts / n,
    }


def concentration_check(oos: pd.DataFrame, group_col: str, max_share: float = 0.6) -> dict:
    """Flags whether one group (a coin, a year) is carrying more than
    `max_share` of total net return -- the failure mode that invalidated
    several nominally-positive candidates in the prior research (one coin
    or one year dressed up as a general result). Uses summed, not
    compounded, per-group return -- see `report()`'s `total_expectancy`
    for why."""
    if len(oos) == 0 or group_col not in oos.columns:
        return {"concentrated": None, "max_group_share": np.nan, "dominant_group": None}
    per_group_return = oos.groupby(group_col)["net_return"].sum()
    positive = per_group_return[per_group_return > 0]
    total_positive = positive.sum()
    if total_positive <= 0:
        return {"concentrated": False, "max_group_share": 0.0, "dominant_group": None}
    share = (positive / total_positive).sort_values(ascending=False)
    return {"concentrated": bool(share.iloc[0] > max_share), "max_group_share": float(share.iloc[0]),
            "dominant_group": share.index[0]}


def classify_status(rep: dict, coin_concentration: dict, period_concentration: dict, cfg: MethodologyConfig) -> str:
    """validated: clears a minimum sample size, a real Sortino edge, a
    strict win rate that isn't propped up entirely by timeouts, and isn't
    carried by one coin or one period.
    watch: positive Sortino but fails a robustness check.
    rejected: everything else."""
    if rep["n"] < cfg.min_report_events or np.isnan(rep["sortino"]):
        return "rejected" if rep["n"] >= 10 else "insufficient_data"
    if rep["sortino"] <= 0:
        return "rejected"
    if coin_concentration.get("concentrated") or period_concentration.get("concentrated"):
        return "watch"
    if rep["strict_win_rate"] < 0.45:
        return "watch"
    return "validated"
