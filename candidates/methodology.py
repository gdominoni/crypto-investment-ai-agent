"""Core event-study methodology shared by every candidate in the battery.

Three invariants are enforced structurally here, not left to each
candidate's own discipline:

1. Entry is always the bar AFTER the one a trigger condition examines.
   A trigger that reads a bar's own close cannot fire an entry at that
   bar's close -- there is no code path in this module that allows it.
2. `pattern_significance()` is the acceptance gate, and its test is
   DIRECTIONAL: the horizon is selected by signed (not absolute) mean
   forward return, and the p-value is a pre-specified upper tail in the
   direction the candidate actually trades. A pattern running opposite
   to its own direction can never be `significant`, and `classify_status`
   independently re-checks the sign of `excess_return`.
3. The TP/SL barrier machinery below (`barrier_prices`, `simulate_trade`,
   `bucket_for_elapsed`, the tp/sl grids) does NOT gate anything and is
   not how a live test opens or resolves -- a live test holds for a fixed
   horizon with no barrier at all. It survives solely to produce the
   informational "if this were traded with a barrier structure" reference
   line, and to be reused by the local-only Freqtrade hyperopt cross-check
   (`execution/hyperopt_runner.py`). Do not read it as the exit model.
4. `report()` always returns win_rate, strict_win_rate, sortino, and
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


# The single source of truth for the concentration rule. Previously the
# number lived in three places (concentration_check's own default, plus a
# hardcoded 0.6 AND a hardcoded "60%" string in explain_non_acceptance) --
# exactly the duplicated-threshold drift hazard already documented for
# TRIGGER_NUMERIC_DEFINITIONS. Every reader and every gate now reads this.
MAX_GROUP_SHARE = 0.6

# Block length for the baseline bootstrap, as a multiple of the holding
# horizon. NOT a guess -- calibrated by measuring the false-positive rate
# under a TRUE null (observed sample drawn from the same process as the
# baseline, so there is no effect to find) on overlapping forward-return
# windows, target 5%:
#
#   i.i.d. resampling (what this module shipped)  -> 43.3%  <- 9x over-rejecting
#   block_len = 1x horizon                        -> 15.3%
#   block_len = 3x horizon                        ->  8.7%   <- chosen
#   block_len = 4x horizon                        ->  7.7%   (power 23.5%)
#
# Power against a real +4% effect falls from 39% (1x) to 25% (3x) to 23.5%
# (4x), so 3x is where calibration stops improving materially but power is
# still being paid for. The residual ~8.7% is stated honestly rather than
# rounded to "5%": the calibration harness draws the observed sample as a
# fully CONTIGUOUS slice (maximum serial dependence), while real trigger
# events are clustered but scattered, so the real-world rate sits below
# this figure -- it is an upper bound, not the expected value.
BASELINE_BLOCK_HORIZON_MULTIPLE = 3


@dataclass(frozen=True)
class MethodologyConfig:
    horizons: tuple[int, ...]
    tp_mult_grid: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.5)
    sl_mult_grid: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.5)
    round_trip_fee: float = 0.002
    min_train_events: int = 20
    min_report_events: int = 50  # classify_status requires n STRICTLY greater than this


def shock_zscore_series(ohlc: pd.DataFrame, short_window: int = 5, baseline_window: int = 252) -> pd.Series:
    """Rolling z-score of short-term realized volatility against the
    coin's own longer trailing distribution of that same short-term
    volatility -- a 'vol-of-vol' anomaly measure. Every value at position
    `loc` uses only bars up to and including `loc` (pandas rolling is
    backward-looking by construction), so classifying a bar as a shock
    never uses information from after that bar.

    NOTE on the z>=3.0 threshold used downstream (`classify_regime`,
    `shock_detector.py`): this is NOT a "3-sigma event" in the textbook
    normal-distribution sense -- measured on real data across this
    project's coin universe, z>=3.0 actually occurs ~2% of the time
    (about 15x more often than 3-sigma-under-normality would predict),
    because this series is strongly right-skewed (empirical skew
    1.8-3.3 per coin), not normal. 3.0 was kept after checking a
    bootstrap comparison of forward returns above vs. below threshold
    across z=1.5-4.5: the effect is present and similarly sized across
    that whole range (no sharp natural cutoff), but loses statistical
    reliability past z~4.0-4.5 as the sample thins. 3.0 sits comfortably
    inside the range that stays both reliable and reasonably extreme
    (~2% of observations), not at either edge -- see
    docs/case_study/methodology-decisions.md for the full numbers."""
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
    ratio up to nonsense.

    `downside_dev == 0` has TWO genuinely different causes that must not
    return the same thing (a real bug, caught by audit): an empty sample
    (nothing to say -> nan) versus a sample with no losing trade at all
    (a flawless candidate -> +inf, an unboundedly good Sortino). The old
    unconditional `nan` meant `classify_status`'s `np.isnan(sortino)`
    branch REJECTED a candidate precisely because it never lost."""
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return float("nan")
    downside_dev = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2))
    if downside_dev > 0:
        return float(returns.mean() / downside_dev * np.sqrt(periods_per_year))
    return float("inf") if returns.mean() > 0 else float("nan")


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


def _forward_return(entry_loc: int, ohlc: pd.DataFrame, direction: str, horizon: int) -> float:
    """NaN (not a silently-clamped short hold) when the full horizon
    isn't available yet -- a real audit finding: this used to clamp
    `exit_loc` to the last bar, so an event at the very end of the
    series returned a 0-bar hold dressed up as a full-horizon result and
    was pooled into the significance test as if it were one.
    `build_events`'s own `entry_loc + max_h >= len(idx)` filter means
    this is unreachable from the battery (verified), but it IS reachable
    from live resolution, which is exactly where a wrong number would be
    recorded as real evidence."""
    idx = ohlc.index
    exit_loc = entry_loc + horizon
    if exit_loc > len(idx) - 1:
        return float("nan")
    entry_price, exit_price = ohlc["close"].iloc[entry_loc], ohlc["close"].iloc[exit_loc]
    return (exit_price / entry_price - 1) if direction == "long" else (1 - exit_price / entry_price)


def path_outcome(entry_price: float, entry_loc: int, ohlc: pd.DataFrame, direction: str, horizon: int) -> dict:
    """The realized outcome of holding a single LIVE occurrence for
    exactly `horizon` bars, no TP/SL barrier involved -- the forward
    return at that bar's close, plus the MFE/MAE observed along the way,
    using the EXACT SAME convention as build_events's own per-row
    computation (bars entry_loc+1 through entry_loc+1+horizon, relative
    to entry_price), so a live occurrence's resolution is directly
    comparable to the backtest sample it's meant to extend -- this is
    what actually gets recorded when a live test resolves (see
    replay/engine.py's _check_live_tests). `entry_loc`/`entry_price`
    should be exactly what was recorded when the live test was opened,
    not re-derived here.

    Returns all-NaN when the full horizon hasn't elapsed in the data
    yet, rather than clamping to the last available bar -- see
    `_forward_return`'s own note. A caller that gets NaN back should
    leave the live test OPEN and retry once the bar exists, never record
    the partial hold as a resolved result."""
    idx = ohlc.index
    exit_loc = entry_loc + horizon
    if exit_loc > len(idx) - 1:
        return {"forward_return": float("nan"), "mfe": float("nan"), "mae": float("nan")}
    seg_hi = ohlc["high"].iloc[entry_loc + 1: entry_loc + 1 + horizon]
    seg_lo = ohlc["low"].iloc[entry_loc + 1: entry_loc + 1 + horizon]
    exit_price = float(ohlc["close"].iloc[exit_loc])
    if direction == "long":
        forward_return = exit_price / entry_price - 1
        mfe = float(seg_hi.max() / entry_price - 1) if len(seg_hi) else float("nan")
        mae = float(seg_lo.min() / entry_price - 1) if len(seg_lo) else float("nan")
    else:
        forward_return = 1 - exit_price / entry_price
        mfe = float(1 - seg_lo.min() / entry_price) if len(seg_lo) else float("nan")
        mae = float(1 - seg_hi.max() / entry_price) if len(seg_hi) else float("nan")
    return {"forward_return": forward_return, "mfe": mfe, "mae": abs(mae)}


def _baseline_forward_returns(ohlc: pd.DataFrame, direction: str, horizon: int, start_loc: int, end_loc: int) -> np.ndarray:
    """Every horizon-day forward return starting from each bar in
    [start_loc, end_loc) -- the coin's own UNCONDITIONAL distribution
    over that same calendar stretch, so the comparison is period-matched
    (not, say, a triggered sample from a volatile year vs. a baseline
    pulled from the whole multi-year history). These windows overlap
    (day 2's 7-day window shares 6 days with day 1's), so this is not an
    i.i.d. sample -- a known, accepted limitation of overlapping-window
    event studies; the bootstrap below is more robust to it than a
    textbook t-test would be, but does not eliminate it."""
    last = min(end_loc, len(ohlc.index) - horizon)
    if last <= start_loc:
        return np.array([])
    close = ohlc["close"].to_numpy()
    entry = close[start_loc:last]
    exit_ = close[start_loc + horizon:last + horizon]
    return (exit_ / entry - 1) if direction == "long" else (1 - exit_ / entry)


def _block_bootstrap_means(chunks: list[np.ndarray], n: int, n_bootstrap: int, block_len: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap over the baseline chunks, replacing an
    i.i.d. `rng.choice` draw. Overlapping h-day forward-return windows
    are strongly serially dependent (day 2's 7-day window shares 6 days
    with day 1's), so resampling them independently understates the
    null distribution's own variance and therefore biases every p-value
    DOWNWARD -- the module previously acknowledged this in a docstring
    but did nothing about it, while simultaneously running a test whose
    threshold (p<0.05) is exactly where that bias does damage.

    Sampling contiguous blocks of length ~= the holding horizon keeps
    that dependence inside the resample. Blocks are drawn within a
    single chunk, never across a boundary, since chunks are separate
    (fold, coin) stretches with no real continuity between them."""
    usable = [c for c in chunks if len(c) > 0]
    if not usable:
        return np.array([])
    lengths = np.array([len(c) for c in usable], dtype=float)
    weights = lengths / lengths.sum()
    means = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        drawn, total = [], 0
        while total < n:
            c = usable[rng.choice(len(usable), p=weights)]
            L = min(block_len, len(c))
            start = int(rng.integers(0, len(c) - L + 1))
            piece = c[start:start + L]
            drawn.append(piece)
            total += len(piece)
        means[b] = np.concatenate(drawn)[:n].mean()
    return means


def pattern_significance(events: pd.DataFrame, ohlc_by_group: dict[str, pd.DataFrame], direction: str,
                          cfg: MethodologyConfig, min_train_periods: int = 3, n_bootstrap: int = 2000,
                          seed: int = 0) -> dict:
    """Answers a genuinely different question than classify_status does:
    not "is trading this condition with THIS TP/SL ladder profitable"
    (Sortino, win_rate -- both conditioned on the barrier structure), but
    "does the market actually behave differently after this condition,
    at all" -- the market's own forward return vs. its own unconditional
    forward-return distribution over the same stretch, no TP/SL/fee
    structure involved. A condition can fail this and still be a real,
    small, reliable effect that classify_status's barriers are simply
    too wide to catch (a low-timeout-fraction candidate) -- and vice
    versa, a condition can clear classify_status by getting lucky on a
    barrier structure fit to the same noise. The two are meant to be
    read together, not as a single verdict.

    The horizon is chosen empirically per fold, ON THE TRAIN SET ONLY,
    then evaluated out-of-sample on the held-out test fold -- the same
    walk-forward discipline already used for tp_mult/sl_mult, extended
    to horizon selection, specifically to avoid picking-and-testing on
    the same data (the exact trap an earlier version of this project's
    own methodology fell into).

    THE TEST IS DIRECTIONAL, in the direction this candidate actually
    trades. Both halves of that used to be broken (found by audit,
    confirmed against the real battery -- see
    docs/case_study/methodology-decisions.md):

      * the horizon was selected by the strongest |mean| forward return,
        so a `long` candidate could have its horizon chosen precisely
        BECAUSE the effect was strongly negative there;
      * the p-value's tail was picked after seeing which way the effect
        went (`>= observed` if above baseline else `<= observed`), which
        is a two-sided procedure priced as one-sided.

    Together those let a pattern that runs OPPOSITE to its own traded
    direction be labelled `significant`, and `classify_status` never
    read `excess_return` to notice. Four of six static candidates were
    "significant" with a negative excess return. `_forward_return` already
    signs its output by direction, so a positive excess always means
    "works in the direction actually traded" -- selecting the horizon by
    signed mean and testing a pre-specified upper tail fixes both, and
    needs no doubling because the side is now fixed in advance rather
    than read off the data.

    `p_value_two_sided` is still reported alongside, as the honest answer
    to the different question "does the market behave differently after
    this condition AT ALL, either way" -- a diagnostic, never the gate."""
    periods = sorted(events["period"].unique())
    if len(periods) <= min_train_periods:
        return {"status": "insufficient_data"}
    rng = np.random.default_rng(seed)
    oos_rows, baseline_chunks, fold_ratios, horizons_used = [], [], [], []
    chosen_horizon = None
    for test_period in periods[min_train_periods:]:
        train = events[events["period"] < test_period]
        test = events[events["period"] == test_period]
        if len(train) < cfg.min_train_events or len(test) == 0:
            continue
        best_h, best_score = cfg.horizons[0], -np.inf
        for h in cfg.horizons:
            rets = [_forward_return(r.entry_loc, ohlc_by_group[r.group], direction, h) for r in train.itertuples()]
            rets = [x for x in rets if x == x]  # drop NaN (horizon not fully available)
            score = float(np.mean(rets)) if rets else -np.inf  # SIGNED, not abs() -- see docstring
            if score > best_score:
                best_h, best_score = h, score
        chosen_horizon = best_h  # persists as the LAST fold's choice -- what a live occurrence discovered now should be held for
        horizons_used.append(int(best_h))
        fold_mfe, fold_mae = [], []
        for r in test.itertuples():
            fr = _forward_return(r.entry_loc, ohlc_by_group[r.group], direction, best_h)
            if fr != fr:
                continue
            oos_rows.append({"group": r.group, "period": test_period, "forward_return": fr})
            mfe, mae = getattr(r, f"mfe_{best_h}"), abs(getattr(r, f"mae_{best_h}"))
            if mfe == mfe and mae == mae:
                fold_mfe.append(mfe)
                fold_mae.append(mae)
        # MFE/MAE is aggregated PER FOLD, at that fold's own horizon, then
        # averaged across folds -- never pooled raw across folds. Excursions
        # scale with sqrt(horizon), so pooling a 1-day MFE with a 21-day one
        # (a real case: c1_long's folds chose 21, 14, 1, 21, 21) produced a
        # single "risk path" number blending incomparable magnitudes.
        if fold_mfe and float(np.mean(fold_mae)) > 0:
            fold_ratios.append(float(np.mean(fold_mfe)) / float(np.mean(fold_mae)))
        for group in test["group"].unique():
            locs = test.loc[test["group"] == group, "entry_loc"]
            chunk = _baseline_forward_returns(ohlc_by_group[group], direction, best_h, int(locs.min()), int(locs.max()) + 1)
            if len(chunk):
                baseline_chunks.append(chunk)

    if not oos_rows or not baseline_chunks:
        return {"status": "insufficient_data"}
    oos_frame = pd.DataFrame(oos_rows)
    oos_returns = oos_frame["forward_return"].to_numpy()
    baseline_pool = np.concatenate(baseline_chunks)
    if len(baseline_pool) < 20:
        return {"status": "insufficient_data"}

    observed_mean = float(oos_returns.mean())
    baseline_mean = float(baseline_pool.mean())
    n = len(oos_returns)
    boot_means = _block_bootstrap_means(
        baseline_chunks, n, n_bootstrap,
        block_len=max(int(chosen_horizon) * BASELINE_BLOCK_HORIZON_MULTIPLE, 1), rng=rng)
    if boot_means.size == 0:
        return {"status": "insufficient_data"}
    # Pre-specified upper tail in the traded direction -- NOT a tail chosen
    # from the data. A negative excess return now correctly yields a large p.
    p_value = float(np.mean(boot_means >= observed_mean))
    p_two_sided = float(np.mean(np.abs(boot_means - baseline_mean) >= abs(observed_mean - baseline_mean)))
    mfe_mae_ratio = float(np.mean(fold_ratios)) if fold_ratios else float("nan")
    return {
        "status": "ok", "n": n, "mean_return": observed_mean, "baseline_mean_return": baseline_mean,
        "excess_return": observed_mean - baseline_mean, "p_value": p_value,
        "significant": bool(p_value < 0.05 and observed_mean > baseline_mean),
        "p_value_two_sided": p_two_sided,
        # Risk dimension, deliberately separate from the directional significance test above:
        # `sortino` here is computed on the RAW forward returns (no fee -- these were never
        # executed trades, no TP/SL/win/loss classification needed, unlike report()'s Sortino which
        # is conditioned on the barrier structure). `mfe_mae_ratio` > 1 means the typical favorable
        # excursion during the hold exceeds the typical adverse one -- a good path-risk sign
        # independent of what the ending return at horizon H happens to be; < 1 is a red flag on
        # the path even when the ending return looks fine.
        "sortino": sortino_ratio(oos_returns), "mfe_mae_ratio": mfe_mae_ratio,
        # Per-event OOS forward returns, so the concentration check can run on the
        # SAME quantity the acceptance gate is decided on (see concentration_check's
        # own `value_col` note) rather than on walk_forward's TP/SL-conditioned returns.
        "oos_events": oos_frame,
        # Which horizons the folds actually chose. More than one distinct value means
        # the "risk path" number above is an average across differently-scaled folds --
        # worth seeing rather than hiding behind a single figure.
        "horizons_used": horizons_used,
        # The horizon a live occurrence of this pattern should be held for --
        # the same one this test was actually evaluated at (the last fold's
        # choice), so live testing measures the identical concept.
        "horizon": int(chosen_horizon),
    }


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


def concentration_check(oos: pd.DataFrame, group_col: str, max_share: float = MAX_GROUP_SHARE,
                         value_col: str = "net_return") -> dict:
    """Flags whether one group (a coin, a year) is carrying more than
    `max_share` of total positive return -- the failure mode that
    invalidated several nominally-positive candidates in the prior
    research (one coin or one year dressed up as a general result). Uses
    summed, not compounded, per-group return -- see `report()`'s
    `total_expectancy` for why.

    `value_col`: WHICH return to measure concentration on. This matters
    and was a real audit finding: acceptance is decided by
    `pattern_significance`'s raw forward returns, but this check used to
    only ever run on `walk_forward`'s TP/SL-conditioned `net_return` --
    two different quantities that genuinely disagree (measured on real
    events: 96.8% concentration on the TP/SL basis vs. 50.0% on the
    forward-return basis for the identical sample). The acceptance gate
    and its robustness check now read the SAME numbers; the TP/SL basis
    is still computed and reported, but purely as a diagnostic.

    `concentrated=None` means "cannot assess", NOT "passed": with no
    positive return anywhere there is nothing for a single group to
    concentrate. The old code returned `False` here, so a candidate
    losing money on every single coin sailed through both concentration
    checks -- harmless while significance was a real gate, not harmless
    once combined with a significance test that ignored direction."""
    if len(oos) == 0 or group_col not in oos.columns or value_col not in oos.columns:
        return {"concentrated": None, "max_group_share": np.nan, "dominant_group": None}
    per_group_return = oos.groupby(group_col)[value_col].sum()
    positive = per_group_return[per_group_return > 0]
    total_positive = positive.sum()
    if total_positive <= 0:
        return {"concentrated": None, "max_group_share": np.nan, "dominant_group": None}
    share = (positive / total_positive).sort_values(ascending=False)
    return {"concentrated": bool(share.iloc[0] > max_share), "max_group_share": float(share.iloc[0]),
            "dominant_group": share.index[0]}


# Plain-English gloss for each classify_status() verdict -- every message
# (fixed template or Sonnet's own free-text context) that shows a raw
# status code to a human should read it through this, not the bare word.
# Added after a real observed case: Sonnet, given only "insufficient_data"
# with no gloss, echoed it back to a human verbatim with no explanation
# of what it meant or what threshold it's short of.
STATUS_PLAIN: dict[str, str] = {
    "accepted": "accepted -- cleared the statistical bar, its trigger opens live tests automatically",
    "watch": "watch -- a real pattern signal, but fails a robustness check (concentration or an unfavorable risk profile), or too little data for the risk check yet",
    "rejected": "rejected -- no statistically significant pattern found against this coin's own baseline",
    "insufficient_data": "insufficient data -- fewer than the required historical occurrences of this trigger to run the test at all yet",
    "error": "error -- this run failed to process it, will retry automatically next time",
    "dropped": "dropped -- a human explicitly removed it from testing, it will never be re-tested automatically",
}


def _sortino_unusable(sortino: float) -> bool:
    """NaN Sortino means "couldn't be computed" -- but +inf means the
    opposite: a candidate with no losing trade at all. The old check was
    a bare `np.isnan(...)`, which was correct only because sortino_ratio
    used to collapse both cases into NaN and therefore REJECTED a
    flawless candidate. Now that the two are distinguished, only the
    genuinely-unusable one gates."""
    return sortino is None or (isinstance(sortino, float) and np.isnan(sortino))


def classify_status(rep: dict, coin_concentration: dict, period_concentration: dict, pattern: dict,
                     cfg: MethodologyConfig) -> str:
    """accepted: the system recognizes a real PATTERN in backtesting --
    `pattern` (from pattern_significance) shows a statistically
    significant, out-of-sample directional effect vs. this coin's own
    unconditional baseline, with a favorable risk path (mean MFE >
    mean MAE at the horizon the effect was found at) -- and it isn't
    carried by one coin or one period. This is now the actual
    acceptance gate. Sortino/win_rate/strict_win_rate (`rep`, from the
    TP/SL-conditioned backtest) are still computed and still reported --
    they describe how well the CURRENT trading structure would have
    captured this pattern historically -- but they no longer gate
    acceptance: this project's purpose is finding real patterns, not
    optimizing a barrier structure around noise (see win_rate's own
    history here -- it was a P&L-conditioned proxy for exactly this
    question before pattern_significance existed to answer it directly).
    Once accepted, the candidate's own trigger opens a LIVE TEST the
    moment it next fires -- no TP/SL, no funded position (this project
    is a pattern-discovery investigation, never an investment strategy):
    held for the horizon `pattern` found significant at, resolved by
    measuring the real forward return/MFE/MAE, the same thing
    pattern_significance itself measures. This is NOT the same claim as
    "validated": that word is reserved everywhere else in this project
    for a candidate that has actually lived through its first 50
    resolved live tests (real, or in the replay, simulated) and still
    held 'accepted' status at that point -- see
    candidates/status_history.py's and replay/status_history.py's
    milestone tracking.
    watch: a real pattern signal that fails a robustness check
    (concentration, or an unfavorable risk path), or too little data for
    the pattern test to say either way.
    rejected: no significant pattern, a pattern pointing the WRONG WAY
    for the direction this candidate trades, or no edge at the
    sample-size gate below."""
    if rep["n"] <= cfg.min_report_events or _sortino_unusable(rep["sortino"]):
        return "rejected" if rep["n"] >= 10 else "insufficient_data"
    if pattern.get("status") != "ok":
        return "watch"
    # `concentrated is None` means "cannot assess" (no positive return anywhere
    # for a single group to concentrate), NOT "passed" -- an unassessable
    # robustness check holds a candidate at `watch` rather than waving it through.
    if coin_concentration.get("concentrated") is None or period_concentration.get("concentrated") is None:
        return "watch"
    if coin_concentration.get("concentrated") or period_concentration.get("concentrated"):
        return "watch"
    # Belt-and-braces alongside pattern_significance's own directional test: a
    # candidate whose measured effect runs OPPOSITE to the direction it trades
    # must never be `accepted`, whatever its p-value says. This gate used to be
    # absent entirely -- `excess_return` was computed, reported to humans, and
    # never once read here (four of six static candidates were "significant"
    # with a negative excess return; a synthetic case with p=0.001 and
    # excess=-5% returned `accepted`).
    excess = pattern.get("excess_return")
    if not pattern.get("significant") or excess is None or excess <= 0:
        return "rejected"
    mfe_mae_ratio = pattern.get("mfe_mae_ratio")
    if mfe_mae_ratio is None or np.isnan(mfe_mae_ratio) or mfe_mae_ratio <= 1:
        return "watch"
    return "accepted"


def _format_dominant_year(value) -> str:
    """`dominant_year` comes out of a pandas groupby keyed on a 'period'
    column that's an int at the point it's assigned (`.dt.year`) but can
    get upcast to float64 by an unrelated NaN elsewhere in the same
    frame -- rendering it straight would leak a trailing '.0' into text
    a human reads (e.g. "a single year (2023.0)"). Real, observed case:
    caught by testing this against real battery data."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def explain_non_acceptance(row: dict, min_report_events: int = 50) -> str:
    """Turns a run_battery.py/run_replay_battery result row into a
    SPECIFIC, honest reason the candidate hasn't reached 'accepted' --
    mirrors classify_status's own branching exactly, in the same order,
    so a keep-or-drop notice (or a trigger summary line) never falls back
    on a generic (or actively wrong) explanation. "Not enough data" and
    "not statistically significant" and "unfavorable risk profile" are
    three different, non-interchangeable reasons a candidate can still
    be un-accepted -- collapsing them into one vague line would mislead
    the reader, or (a real observed case: a 'rejected' candidate with a
    favorable p-value AND a favorable MFE/MAE ratio, rejected purely
    because its raw N was below the threshold) let raw stats sit next to
    a verdict they didn't actually determine, implying they did. `row`
    needs at least: n, pattern_significant, pattern_p_value,
    pattern_mfe_mae_ratio, max_coin_share, max_year_share, dominant_coin,
    dominant_year (all present on every non-"insufficient_data"/"error"
    row). `min_report_events` matches MethodologyConfig's own default --
    every real caller runs with that default, so callers that don't
    already have a MethodologyConfig in hand (e.g. a trigger summary
    line) don't need to construct one just for this one field."""
    n = row.get("n") or 0
    if n <= min_report_events:
        return f"only {n} qualifying historical occurrence(s) so far (needs more than {min_report_events} to even be evaluated)"
    significant = row.get("pattern_significant")
    if significant is None:
        return "not enough data yet for the pattern-significance test itself, despite enough raw occurrences for the reference backtest above"
    # Concentration is checked BEFORE significance here, matching classify_status's
    # own order exactly -- getting this backwards is a real bug that was shipped
    # and observed: a candidate concentrated in one coin/year is "watch" regardless
    # of its p-value, but the wrong order made it look like "not significant" was
    # the reason even when the true (and often unrelated) p-value said otherwise.
    max_coin_share, max_year_share = row.get("max_coin_share") or 0, row.get("max_year_share") or 0
    if max_coin_share > MAX_GROUP_SHARE or max_year_share > MAX_GROUP_SHARE:
        qualifier = "a statistically significant pattern" if significant else "a pattern"
        rule = f"no single coin or year may carry more than {MAX_GROUP_SHARE:.0%} of the positive return"
        if max_coin_share >= max_year_share:
            return f"{qualifier}, but {max_coin_share:.0%} of it comes from a single coin ({row.get('dominant_coin')}) -- too concentrated to trust as general ({rule})"
        return f"{qualifier}, but {max_year_share:.0%} of it comes from a single year ({_format_dominant_year(row.get('dominant_year'))}) -- too concentrated to trust as general ({rule})"
    # A pattern pointing the WRONG WAY for the direction this candidate trades is
    # its own distinct, non-interchangeable reason -- and the most misleading one
    # to collapse into "not significant", since the effect may be strongly real,
    # just inverted. Checked before the generic significance line for that reason.
    excess = row.get("pattern_excess_return")
    if excess is not None and excess <= 0:
        return (f"a real effect was measured, but it runs OPPOSITE to this candidate's own direction "
                f"({excess:+.2%} vs. this coin's own baseline) -- the pattern may well be real, it just "
                f"doesn't work the way this candidate trades it")
    if not significant:
        p_value = row.get("pattern_p_value")
        return (f"a real pattern was tested for but not found -- not statistically significant vs. this coin's own "
                f"baseline (p={p_value:.3f})" if p_value is not None else
                "a real pattern was tested for but not found -- not statistically significant vs. this coin's own baseline")
    mfe_mae_ratio = row.get("pattern_mfe_mae_ratio")
    if mfe_mae_ratio is None or (isinstance(mfe_mae_ratio, float) and np.isnan(mfe_mae_ratio)):
        return "a statistically significant pattern, but its risk profile (MFE/MAE ratio) couldn't be computed"
    if mfe_mae_ratio <= 1:
        return (f"a statistically significant pattern, but an unfavorable risk profile (MFE/MAE ratio={mfe_mae_ratio:.2f} -- "
                f"the risk taken during the hold exceeds the eventual reward)")
    return "did not clear the acceptance bar (see the current status above)"


def format_candidate_details(candidate: str, row: dict, definition: str | None = None, horizon: int | None = None,
                              milestone: dict | None = None, tp_mult: float | None = None, sl_mult: float | None = None,
                              min_report_events: int = 50) -> str:
    """Full numeric breakdown for one candidate, in bullet points --
    powers Telegram's `/details <name>`/`/replay_details <name>`.
    Deliberately the detail `_trigger_summary_line()`/`format_trigger_summary()`
    leave out to keep /summary short: a human reading "elevated futures
    concentration" or "not statistically significant" in a proposal or
    summary line has no way to ask "how elevated, exactly?" without
    this. `definition` is the trigger's own numeric definition (e.g.
    "funding z-score below -2.0"), `horizon` is the empirically-derived
    number of bars a new live test is currently held for, and
    `milestone` is the caller's `all_latest_statuses()[candidate]`
    entry (`milestone_reported`/`milestone_cleared`/`last_checkpoint_n`)
    -- `status` alone (accepted/watch/rejected/...) is a DIFFERENT claim
    than `validated` (see docs/case_study/methodology-decisions.md's
    "accepted vs validated" entry): a real, observed case of exactly
    this confusion is why this parameter exists -- `status` can say
    'accepted' while the candidate is ALSO already validated, and
    nothing about the word 'accepted' alone tells a reader that.
    `tp_mult`/`sl_mult` are the project's own walk-forward grid search's
    chosen multipliers (against the duration-bucketed MFE/MAE anchors)
    for the reference TP/SL backtest below -- only ever set for a
    currently-`accepted` candidate (`live_state`/`battery["candidates"]`
    from `run_all()`/`run_replay_battery()`'s own side effect,
    `replay/state.py::load_battery_status()`, only ever populated when
    `status == "accepted"`), so a real N/win_rate/Sortino number was
    previously shown with no way to know what TP/SL structure actually
    produced it. All five are passed in by the caller rather than looked
    up here: this module has no dependency on candidates/definitions.py,
    the dynamic-condition registry, or execution/live_test_state.py
    (production) / replay/state.py (replay) / candidates/status_history.py
    (production) / replay/status_history.py (replay) -- any of which
    would break the production/replay symmetry this module is shared
    by, and isn't the place to add one just for this."""
    status = row.get("status", "unknown")
    lines = [f"<b>{_escape_html(candidate)}</b>"]
    if definition:
        lines.append(f"What triggers it: {_escape_html(definition)}")
    direction = row.get("direction")
    lines.append(f"Status: {_escape_html(STATUS_PLAIN.get(status, status))}" + (f"  (direction: {direction})" if direction else ""))
    if horizon is not None:
        lines.append(f"Held for: {horizon}d (empirically-derived -- the last walk-forward fold's best-performing horizon; re-checked every week, see the README's Phase 3)")
    # Gated on CURRENT status being 'accepted' -- milestone_reported/milestone_cleared
    # can persist from a PAST checkpoint reached while this candidate used to be
    # accepted, before later degrading to watch/rejected. Showing "VALIDATED"/"NOT
    # validated" next to a currently-rejected candidate would read as describing
    # today's status when it's really stale history -- the exact "two facts shown
    # together that don't actually relate" confusion this project has caught
    # (and fixed) more than once already; explain_non_acceptance() below already
    # gives the real, current reason.
    if status == "accepted" and milestone and milestone.get("milestone_reported"):
        n_reached, cleared = milestone.get("last_checkpoint_n"), milestone.get("milestone_cleared")
        verb = "VALIDATED" if cleared else "NOT validated"
        lines.append(f"{verb} -- {'cleared' if cleared else 'did not clear'} the acceptance bar at its {n_reached}-occurrence checkpoint "
                     f"(re-checked fresh again at {n_reached + 50}, not a permanent badge)")
    elif status == "accepted" and milestone is not None:
        lines.append("Not yet validated -- hasn't reached its first 50-occurrence checkpoint yet")
    lines.append("")

    n = row.get("n")
    lines.append(f"• Historical occurrences (N): {n if n is not None else 'n/a'}")

    sig, p, excess = row.get("pattern_significant"), row.get("pattern_p_value"), row.get("pattern_excess_return")
    if sig is not None:
        verdict = "significant" if sig else "not significant"
        p_bit = f" (p={p:.3f})" if p is not None else ""
        excess_bit = f", excess return vs. this coin's own baseline: {excess:+.2%}" if excess is not None else ""
        lines.append(f"• Statistical significance: {verdict}{p_bit}{excess_bit}")

    ratio = row.get("pattern_mfe_mae_ratio")
    if ratio is not None and not (isinstance(ratio, float) and np.isnan(ratio)):
        lines.append(f"• Risk path (mean favorable / mean adverse excursion): {ratio:.2f}  (favorable if > 1.0)")

    # "-- flagged above 60%" used to print unconditionally, so a perfectly
    # diversified 32% read as "32% ... flagged" at a glance. Say whether THIS
    # value trips the rule, not merely what the rule is.
    def _conc_verdict(share: float) -> str:
        return (f"  -- OVER the {MAX_GROUP_SHARE:.0%} limit" if share > MAX_GROUP_SHARE
                else f"  -- within the {MAX_GROUP_SHARE:.0%} limit")

    max_coin_share, dominant_coin = row.get("max_coin_share"), row.get("dominant_coin")
    if max_coin_share is not None and not (isinstance(max_coin_share, float) and np.isnan(max_coin_share)):
        coin_bit = f" ({_escape_html(dominant_coin)})" if dominant_coin else ""
        lines.append(f"• Coin concentration: {max_coin_share:.0%} of total positive return comes from a single coin{coin_bit}{_conc_verdict(max_coin_share)}")

    max_year_share, dominant_year = row.get("max_year_share"), row.get("dominant_year")
    if max_year_share is not None and not (isinstance(max_year_share, float) and np.isnan(max_year_share)):
        year_bit = f" ({_escape_html(_format_dominant_year(dominant_year))})" if dominant_year else ""
        lines.append(f"• Year concentration: {max_year_share:.0%} of total positive return comes from a single year{year_bit}{_conc_verdict(max_year_share)}")

    win_rate, sortino = row.get("win_rate"), row.get("sortino")
    if win_rate is not None and not (isinstance(win_rate, float) and np.isnan(win_rate)):
        expectancy = row.get("total_expectancy")
        expectancy_bit = f", total expectancy={expectancy:+.1%}" if expectancy is not None else ""
        tpsl_bit = f"TP={tp_mult:.2f}x/SL={sl_mult:.2f}x (of the anchors), " if tp_mult is not None and sl_mult is not None else ""
        lines.append(f"• Reference TP/SL backtest: {tpsl_bit}win rate={win_rate:.1%}, Sortino={sortino:.2f}{expectancy_bit}"
                      f"  (informational only, doesn't gate status -- see README's Phase 2)")

    if status in ("watch", "rejected"):
        lines.append("")
        lines.append(f"Why not accepted: {_escape_html(explain_non_acceptance(row, min_report_events))}")
    return "\n".join(lines)


def _escape_html(text: str) -> str:
    """Local copy of the same one-liner escape_html() every other module
    already has its own copy of -- kept here rather than imported so this
    module (imported by novel_condition_tester.py, in turn imported by
    llm_pipeline/haiku_sonnet_pipeline.py) never gains a dependency on
    llm_pipeline, which would be circular."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _trigger_summary_line(name: str, row: dict, milestone: dict | None = None) -> str:
    n = row.get("n")
    bits = [f"N={n}"]
    sig, p = row.get("pattern_significant"), row.get("pattern_p_value")
    if sig is not None:
        bits.append(f"p={p:.3f}" if p is not None else ("significant" if sig else "not significant"))
    ratio = row.get("pattern_mfe_mae_ratio")
    if ratio is not None and not (isinstance(ratio, float) and np.isnan(ratio)):
        bits.append(f"MFE/MAE={ratio:.2f}")
    # `status` (accepted/watch/rejected) and `validated` are different claims
    # (see docs/case_study/methodology-decisions.md's "accepted vs validated"
    # entry) -- without this tag, a candidate could be accepted AND already
    # VALIDATED with nothing in /summary distinguishing it from one that's
    # merely accepted, a real gap caught the same way format_candidate_details's
    # own missing validated line was (a human noticing the mismatch live).
    # Gated on CURRENT status being 'accepted' -- a real, separately-caught bug:
    # milestone_reported/milestone_cleared can persist from a PAST checkpoint
    # reached while a candidate USED to be accepted, before later degrading to
    # watch/rejected. Tagging a currently-rejected line "not validated" reads as
    # describing today's status when it's really stale history unrelated to the
    # real reason (already given below via "Why:").
    if row.get("status") == "accepted" and milestone and milestone.get("milestone_reported"):
        bits.append("VALIDATED" if milestone.get("milestone_cleared") else "not validated")
    line = f"  <b>{_escape_html(name)}</b> -- {', '.join(bits)}"
    if row.get("status") in ("watch", "rejected"):
        # Without this, a rejected/watch candidate can show a favorable p-value
        # AND a favorable MFE/MAE ratio right next to its verdict -- looking like
        # the numbers contradict the status -- when the real reason is a DIFFERENT
        # field entirely (e.g. raw N below the sample-size gate, or concentration).
        # explain_non_acceptance() names the actual reason, in classify_status's
        # own branching order, so the line never implies a stat determined the
        # verdict when it didn't (a real, observed case of exactly this confusion).
        line += f"\n    Why: {_escape_html(explain_non_acceptance(row))}"
    return line


def _insufficient_data_block(items: list[tuple[str, dict]]) -> list[str]:
    """A one-line-per-candidate list gets unreadable fast once the
    dynamic registry grows -- most 'insufficient_data' candidates sit at
    N=0 with literally nothing to say beyond that. Those are collapsed
    into a single compact name list; only candidates with SOME real
    progress (N>0, still short of the threshold) get their own line,
    since that's the genuinely informative case ('how close is it')."""
    zero = sorted(name for name, row in items if not row.get("n"))
    partial = sorted((name, row) for name, row in items if row.get("n"))
    lines = []
    if partial:
        lines.append("Some data so far, still below the threshold to test:")
        for name, row in partial:
            lines.append(_trigger_summary_line(name, row))
    if zero:
        if partial:
            lines.append("")
        lines.append(f"No historical occurrences yet ({len(zero)}): " + ", ".join(f"<b>{_escape_html(n)}</b>" for n in zero))
    return lines


def format_trigger_summary(status_summary: dict, dropped_extra: dict | None = None) -> tuple[str, str]:
    """Two-part human-readable report over every tracked trigger --
    'still under test' (accepted/watch/insufficient_data/error) and
    'already discarded' (rejected, or explicitly dropped) -- grouped by
    status with an indented list rather than a fixed-width column table,
    since dynamic candidate labels vary wildly in length (a Sonnet-chosen
    snake_case label can run 40+ characters) and a rigid grid breaks
    badly on a narrow mobile screen the moment one name is much longer
    than the rest. Returned as two SEPARATE strings (not one combined
    message) -- both because the split itself is the useful structure
    (what's still being evaluated vs. what's already settled), and
    because a single combined message risks exceeding Telegram's
    4096-char limit once enough dynamic candidates accumulate.

    `status_summary`: a fresh run_all()/run_replay_battery() result dict
    (candidate -> row with at least 'status', usually also 'n',
    'pattern_significant', 'pattern_p_value', 'pattern_mfe_mae_ratio').
    `dropped_extra`: the caller's all_latest_statuses() output, used both
    to surface explicitly-dropped candidates (excluded from status_summary
    entirely -- run_all/run_replay_battery skip dropped candidates, so
    they'd otherwise vanish from the report rather than showing up as
    'discarded') and to tag a VALIDATED candidate's line (milestone info
    lives in status_history.py, not in status_summary's own rows)."""
    groups: dict[str, list[tuple[str, dict]]] = {}
    for name, row in status_summary.items():
        groups.setdefault(row.get("status", "unknown"), []).append((name, row))
    for name, info in (dropped_extra or {}).items():
        if info.get("dropped") and name not in status_summary:
            groups.setdefault("dropped", []).append((name, {"status": "dropped"}))

    def _section(header: str, keys_and_labels: list[tuple[str, str]]) -> str:
        lines = [f"<b>{header}</b>"]
        found_any = False
        for key, label in keys_and_labels:
            items = groups.get(key, [])
            if not items:
                continue
            found_any = True
            lines.append(f"\n<b>{label} ({len(items)})</b>")
            # No group-level gloss for watch/rejected -- STATUS_PLAIN names ONE
            # generic reason (e.g. "not statistically significant"), but
            # classify_status can reach either status through several different,
            # non-interchangeable paths (too little N, concentration, unfavorable
            # risk profile...). Each line already gets its own SPECIFIC "Why:" via
            # explain_non_acceptance() below -- showing a generic gloss above that
            # doesn't just repeat it, it can flatly contradict it (a real observed
            # case: the group gloss said "not significant" while every actual line
            # was rejected for low N instead, with a perfectly good p-value).
            if key not in ("watch", "rejected"):
                gloss = STATUS_PLAIN.get(key)
                if gloss:
                    lines.append(f"<i>{gloss.split(' -- ', 1)[-1]}</i>")
            if key == "insufficient_data":
                lines.extend(_insufficient_data_block(items))
            else:
                for name, row in sorted(items):
                    lines.append(_trigger_summary_line(name, row, milestone=(dropped_extra or {}).get(name)))
        if not found_any:
            lines.append("\nNothing here right now.")
        return "\n".join(lines)

    under_test = _section("Still under test", [
        ("accepted", "Accepted"), ("watch", "Watch"),
        ("insufficient_data", "Insufficient data"), ("error", "Processing error (will retry automatically)"),
    ])
    discarded = _section("Already discarded", [("rejected", "Rejected"), ("dropped", "Dropped")])
    return under_test, discarded
