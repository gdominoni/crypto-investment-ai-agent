"""Tests a novel condition Haiku/Sonnet flagged and a human approved --
via a small, whitelisted spec (one or more indicator+comparison+threshold
clauses, ANDed together), never by executing LLM-generated code. Sonnet
can only ever choose indicators from `SUPPORTED_INDICATORS` below; if the
condition it wants to test isn't expressible with what's here, the
correct outcome is a human adding a new indicator to this registry
deliberately, not the system running arbitrary generated logic
unattended. This applies identically in production and in the replay --
both import this exact module, there is no separate "simulation-only"
version of the whitelist.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from candidates.data_loading import load_daily, load_funding, zscore
from candidates.definitions import trend_efficiency_ratio
from candidates.macro_calendar import macro_release_days
from candidates.methodology import (
    MethodologyConfig, build_events, classify_status, concentration_check, pattern_significance, report,
    shock_zscore_series, walk_forward,
)

SHOCK_ZSCORE_THRESHOLD = 3.0  # matches run_battery.py / shock_detector.py -- one consistent definition of "shock" everywhere


def _rsi(df: pd.DataFrame, funding: pd.Series | None, scale: int = 1, window: int = 14) -> pd.Series:
    window *= scale
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr_pct(df: pd.DataFrame, funding: pd.Series | None, scale: int = 1, window: int = 14) -> pd.Series:
    """Average True Range as a % of price, so it's comparable across
    coins of very different nominal price -- true range (not just
    high-low) accounts for gaps against the prior close, unlike
    `daily_range_pct`."""
    window *= scale
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return atr / df["close"]


def _donchian_pct(df: pd.DataFrame, funding: pd.Series | None, scale: int = 1, window: int = 20) -> pd.Series:
    """Where today's close sits within the prior N-day high/low channel:
    0 = at the channel low, 1 = at the channel high, >1 or <0 = broke out
    beyond the channel entirely. Uses only the coin's own already-elapsed
    bars -- no future data, same as every other indicator here."""
    window *= scale
    upper = df["high"].rolling(window).max()
    lower = df["low"].rolling(window).min()
    return (df["close"] - lower) / (upper - lower)


def _bollinger_pctb(df: pd.DataFrame, funding: pd.Series | None, scale: int = 1, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """%B: 0 = at the lower band, 1 = at the upper band, same
    out-of-[0,1] breakout reading as the Donchian version above but
    based on rolling mean/std instead of rolling high/low extremes."""
    window *= scale
    ma = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    upper, lower = ma + num_std * std, ma - num_std * std
    return (df["close"] - lower) / (upper - lower)


# Every function takes (df, funding, scale=1) -- `scale` reinterprets
# every DAY-defined window as its hour-equivalent (scale=24) when
# evaluated against an hourly frame for live trigger detection (never
# the backtest, which always calls with scale=1); see
# docs/case_study/methodology-decisions.md. `shock_zscore` is the one
# deliberate exception -- a shock stays a daily-window concept regardless
# of `scale`, matching classify_regime/shock_detector.py everywhere else.
SUPPORTED_INDICATORS: dict[str, Callable[..., pd.Series]] = {
    "close_return_1d": lambda df, funding, scale=1: df["close"].pct_change(1 * scale),
    "close_return_5d": lambda df, funding, scale=1: df["close"].pct_change(5 * scale),
    "daily_range_pct": lambda df, funding, scale=1: (df["high"] - df["low"]) / df["close"],
    "volume_zscore_30d": lambda df, funding, scale=1: zscore(df["volume"], 30 * scale),
    "funding_zscore_30d": lambda df, funding, scale=1: zscore(funding.reindex(df.index).ffill(), 30 * scale) if funding is not None else pd.Series(np.nan, index=df.index),
    "efficiency_ratio_20d": lambda df, funding, scale=1: trend_efficiency_ratio(df["close"], 20 * scale),
    "is_macro_day": lambda df, funding, scale=1: pd.Series(
        (df.index.floor("D") if scale > 1 else df.index).isin(macro_release_days()), index=df.index).astype(float),
    # the same 'vol-of-vol' shock measure Phase 1's methodology uses to
    # isolate extreme historical events from the static battery -- tested
    # here on ITS OWN excluded population when a live shock fires, giving
    # it a real, walk-forward-validated anchor set rather than an
    # invented one. Deliberately ignores `scale` -- see module note above.
    "shock_zscore": lambda df, funding, scale=1: shock_zscore_series(df),
    "rsi_14d": _rsi,
    "atr_pct_14d": _atr_pct,
    "donchian_pct_20d": _donchian_pct,
    "bollinger_pctb_20d": _bollinger_pctb,
}

_OPERATORS: dict[str, Callable[[pd.Series, float], pd.Series]] = {
    "<": operator.lt, ">": operator.gt, "<=": operator.le, ">=": operator.ge,
}

# Plain-English gloss for each indicator, keyed identically to
# SUPPORTED_INDICATORS -- every message that shows a raw condition
# clause to a human (proposal, test result, milestone, checkpoint) reads
# through this, so nobody has to already know what "rsi_14d" means to
# understand what's being tested. The threshold/operator stay as
# numbers (still precise), only the indicator name gets translated.
INDICATOR_PLAIN_NAMES: dict[str, str] = {
    "close_return_1d": "1-day price change",
    "close_return_5d": "5-day price change",
    "daily_range_pct": "daily high-low range (% of price)",
    "volume_zscore_30d": "30-day volume z-score (how unusual today's volume is)",
    "funding_zscore_30d": "30-day funding-rate z-score",
    "efficiency_ratio_20d": "20-day trend efficiency ratio (0=choppy, 1=straight trend)",
    "is_macro_day": "is a macro-release day (CPI/FOMC)",
    "shock_zscore": "shock z-score (how extreme today's price move is vs. this coin's own history)",
    "rsi_14d": "14-day RSI (momentum, 0-100 scale)",
    "atr_pct_14d": "14-day average true range (% of price, a volatility measure)",
    "donchian_pct_20d": "position within the 20-day price channel (0=bottom, 1=top)",
    "bollinger_pctb_20d": "position within the 20-day Bollinger Bands (0=lower band, 1=upper band)",
}

# Renders each clause's comparison in plain English instead of raw `<`/`>`
# symbols -- two reasons: it reads more naturally ("above 3.0" vs "> 3.0"),
# and it sidesteps a real bug observed live (2026-08-28 replay run): a
# raw `>`/`<` HTML-escaped to `&gt;`/`&lt;` sometimes showed up as literal
# text in Telegram instead of rendering back to the symbol. Never
# generating the symbol at all removes the failure mode entirely instead
# of relying on escaping/decoding to round-trip correctly.
OPERATOR_PLAIN: dict[str, str] = {"<": "below", ">": "above", "<=": "at most", ">=": "at least"}


def build_indicator_snapshot(coin: str, as_of: pd.Timestamp | None = None) -> str:
    """The most recent real value of every whitelisted indicator for
    `coin`, as of `as_of` (default: right now) -- gives Sonnet actual
    numbers to reason from when it's asked to explain a shock or propose
    a compound novel_condition_spec, instead of guessing at indicator
    readings it can't actually see. Used identically by production
    (`as_of=None`) and the replay (its own simulated `as_of`)."""
    daily = load_daily(coin)
    if as_of is not None:
        daily = daily.loc[:as_of]
    funding = load_funding(coin)
    if funding is not None and as_of is not None:
        funding = funding.loc[:as_of]
    if len(daily) == 0:
        return f"{coin}: no data available."
    parts = []
    for name, fn in SUPPORTED_INDICATORS.items():
        try:
            val = fn(daily, funding).iloc[-1]
            parts.append(f"{name}={val:.3f}" if pd.notna(val) else f"{name}=n/a (not enough history yet)")
        except Exception:
            parts.append(f"{name}=n/a")
    return f"{coin} current indicator readings: " + ", ".join(parts)


@dataclass(frozen=True)
class Clause:
    indicator: str
    op: str
    threshold: float

    def __post_init__(self):
        if self.indicator not in SUPPORTED_INDICATORS:
            raise ValueError(f"Unsupported indicator '{self.indicator}' -- must be one of {list(SUPPORTED_INDICATORS)}")
        if self.op not in _OPERATORS:
            raise ValueError(f"Unsupported operator '{self.op}' -- must be one of {list(_OPERATORS)}")


@dataclass(frozen=True)
class ConditionSpec:
    """`clauses` are ANDed together -- one clause is just the single-
    condition case, there's no separate code path for it. No cap on how
    many clauses: the real constraint on over-specific combinations is
    statistical (each extra AND shrinks the sample, and N > 100 is
    already a hard gate in classify_status), not an arbitrary limit
    here."""
    label: str
    clauses: tuple[Clause, ...]
    direction: str  # "long" or "short"
    horizons: tuple[int, ...] = (1, 3, 7, 14, 21)

    def __post_init__(self):
        if len(self.clauses) == 0:
            raise ValueError("ConditionSpec needs at least one clause")
        if self.direction not in ("long", "short"):
            raise ValueError("direction must be 'long' or 'short'")


def format_pattern_significance(pattern: dict) -> str:
    """One-line rendering of `pattern_significance`'s result, shared by
    every message (production and replay) that shows a test-it result --
    deliberately separate from the accepted/watch/rejected verdict line,
    since it answers a different question (see that function's own
    docstring)."""
    if pattern.get("status") != "ok":
        return "Pattern signal: not enough data to test independently of the TP/SL structure."
    verdict = "significant" if pattern["significant"] else "not significant"
    ratio = pattern.get("mfe_mae_ratio")
    if ratio is None or ratio != ratio:  # NaN -- no fold had a usable excursion pair
        risk = "Risk profile: MFE/MAE ratio couldn't be computed for this condition."
    else:
        risk = (f"Risk profile: Sortino={pattern['sortino']:.2f} on the raw path (no fee, no TP/SL), "
                f"MFE/MAE ratio={ratio:.2f} (favorable vs. adverse excursion during the hold -- "
                f"{'reward tends to exceed the risk taken to get there' if ratio > 1 else 'the path risk exceeds the eventual reward, even if the ending return looks fine'}).")
    # The test is one-sided in the direction this candidate actually trades, so a
    # NEGATIVE excess return means the effect runs the wrong way -- never dressed
    # up as a near-miss on significance (see pattern_significance's own docstring).
    direction_note = ("" if pattern["excess_return"] > 0 else
                      "\nNote: this effect runs OPPOSITE to the direction this condition trades.")
    return (f"Pattern signal (independent of TP/SL): excess return {pattern['excess_return']:+.2%} vs. this "
            f"coin's own baseline over the same period, p={pattern['p_value']:.3f} ({verdict} at the 5% level, "
            f"one-sided in the traded direction, N={pattern['n']}).{direction_note}\n{risk}")


def condition_desc(spec: ConditionSpec) -> str:
    """Single shared, human-readable rendering of a (possibly multi-
    clause) spec -- every message that shows a novel condition to a
    human (proposal, result, milestone, checkpoint) uses this, so a
    2-clause spec never silently gets rendered as if it had one. Each
    clause's indicator is translated through INDICATOR_PLAIN_NAMES and
    its comparison through OPERATOR_PLAIN, so the rendering never assumes
    the reader already knows what "rsi_14d" or a raw "<"/">" means."""
    return " AND ".join(f"{INDICATOR_PLAIN_NAMES.get(c.indicator, c.indicator)} {OPERATOR_PLAIN.get(c.op, c.op)} {c.threshold}" for c in spec.clauses)


def test_novel_condition(spec: ConditionSpec, coins: list[str], as_of: pd.Timestamp | None = None) -> dict:
    """Runs the same walk-forward, concentration-checked pipeline
    `run_battery.py` uses for the static candidates -- including the same
    shock-regime exclusion, with one deliberate exception: when the spec
    being tested IS the `shock_zscore` indicator itself (Mode B, see
    `shock_detector.py`), the trigger condition already selects extreme-
    volatility bars by construction, so excluding "shock"-regime events
    here would exclude the very population this test exists to evaluate.

    `as_of`, if given, truncates every coin's data to that date before
    anything else runs -- used by the historical replay (`replay/`) so a
    "test it" resolved on a simulated date can never see real data from
    after it. `None` (the default, and every production call site) means
    unrestricted -- current behavior, unchanged."""
    cfg = MethodologyConfig(horizons=spec.horizons)
    # If ANY clause is shock_zscore, the combined trigger already selects
    # (at least partly) extreme-volatility bars by construction, so the
    # usual shock-regime exclusion would exclude part of the very
    # population this test exists to evaluate -- same reasoning as the
    # single-indicator case, just checked across all clauses now.
    is_shock_indicator = any(c.indicator == "shock_zscore" for c in spec.clauses)

    all_events = []
    ohlc_by_coin = {}
    n_shock_excluded = 0
    for coin in coins:
        daily = load_daily(coin)
        if as_of is not None:
            daily = daily.loc[:as_of]
        ohlc_by_coin[coin] = daily
        funding = load_funding(coin)
        if funding is not None and as_of is not None:
            funding = funding.loc[:as_of]
        trigger = pd.Series(True, index=daily.index)
        for clause in spec.clauses:
            signal = SUPPORTED_INDICATORS[clause.indicator](daily, funding)
            trigger &= _OPERATORS[clause.op](signal, clause.threshold).fillna(False)
        shock_z = None if is_shock_indicator else shock_zscore_series(daily)
        ev = build_events(daily, trigger, spec.direction, spec.horizons,
                           shock_z=shock_z, shock_threshold=SHOCK_ZSCORE_THRESHOLD)
        if len(ev):
            ev["group"] = coin
            ev["period"] = ev["trigger_time"].dt.year
            if is_shock_indicator:
                all_events.append(ev)
            else:
                n_shock_excluded += int((ev["regime"] == "shock").sum())
                all_events.append(ev[ev["regime"] == "normal"])

    if not all_events or not any(len(e) for e in all_events):
        return {"spec": spec, "status": "insufficient_data", "n_raw_triggers": 0, "n_shock_excluded": n_shock_excluded}

    events = pd.concat(all_events, ignore_index=True)
    oos, params_log = walk_forward(events, ohlc_by_coin, spec.direction, cfg)
    rep = report(oos)
    # pattern_significance is now the actual acceptance gate -- see
    # classify_status's own docstring. `rep` (Sortino/win_rate, from the
    # TP/SL-conditioned backtest) is still computed and still reported,
    # but no longer decides accepted/watch/rejected.
    pattern = pattern_significance(events, ohlc_by_coin, spec.direction, cfg)
    # Concentration on the same forward returns the gate reads -- mirrors
    # candidates/run_battery.py::_concentration_for (see concentration_check's
    # own `value_col` note for why the TP/SL basis genuinely disagrees).
    _pat_frame = pattern.get("oos_events") if pattern.get("status") == "ok" else None
    if _pat_frame is not None and len(_pat_frame):
        coin_conc = concentration_check(_pat_frame, "group", value_col="forward_return")
        year_conc = concentration_check(_pat_frame, "period", value_col="forward_return")
    else:
        coin_conc = concentration_check(oos, "group")
        year_conc = concentration_check(oos, "period")
    status = classify_status(rep, coin_conc, year_conc, pattern, cfg)

    # Full-data anchors + the most recent fold's multipliers -- what a
    # caller actually needs to push this condition as a live signal
    # (`execution/signal_store.py::push_manual_signal`) once accepted,
    # not just a pass/fail verdict.
    live_anchors, live_tp_mult, live_sl_mult = None, None, None
    if status == "accepted":
        from candidates.methodology import compute_anchors
        full_anchors = compute_anchors(events, spec.horizons)
        live_anchors = {str(h): {"mfe": full_anchors[h]["mfe"], "mae": full_anchors[h]["mae"]} for h in spec.horizons}
        if len(params_log):
            live_tp_mult = float(params_log.iloc[-1]["tp_mult"])
            live_sl_mult = float(params_log.iloc[-1]["sl_mult"])

    return {
        "spec": spec, "status": status, "n_raw_triggers": len(events), "n_shock_excluded": n_shock_excluded, **rep,
        "coin_concentration": coin_conc, "year_concentration": year_conc, "pattern_significance": pattern,
        "params_log": params_log.to_dict("records"),
        "live_anchors": live_anchors, "live_tp_mult": live_tp_mult, "live_sl_mult": live_sl_mult,
    }
