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


MAX_WITHIN_DAYS = 14  # a "sequence" longer than this stops being one event and becomes a regime


# The indicators shown in the day-by-day lead-up below. Deliberately a
# SUBSET, not all twelve: the lead-up is 7 rows deep, so showing every
# indicator would multiply this block's token cost for detail nobody
# reasons from -- and this project has already been bitten once by
# unbounded LLM context (see PROJECT_MAP.md's Cost Optimization).
# These six are the ones an ordering hypothesis is actually built from:
# what moved, how violently, how stretched, how crowded.
LEADUP_INDICATORS = ("close_return_1d", "shock_zscore", "rsi_14d",
                     "volume_zscore_30d", "funding_zscore_30d", "bollinger_pctb_20d")


def build_indicator_leadup(coin: str, days: int = 7, as_of: pd.Timestamp | None = None) -> str:
    """The last `days` days of the key indicators, one row per day --
    what was happening BEFORE and INTO the event, not just at it.

    `build_indicator_snapshot` gives a single instant, which is enough to
    answer "is this extreme right now" but structurally cannot support an
    ORDERED hypothesis: with only today's numbers, Sonnet cannot tell
    "the crash was three days ago and the news just landed" from "both
    happened today", and therefore cannot sensibly choose a clause's
    `within_days`. Giving it the run-up is what makes that parameter
    usable rather than decorative."""
    daily = load_daily(coin)
    if as_of is not None:
        daily = daily.loc[:as_of]
    funding = load_funding(coin)
    if funding is not None and as_of is not None:
        funding = funding.loc[:as_of]
    if len(daily) == 0:
        return f"{coin}: no data available."
    cols = {}
    for name in LEADUP_INDICATORS:
        try:
            cols[name] = SUPPORTED_INDICATORS[name](daily, funding).tail(days)
        except Exception:
            continue
    if not cols:
        return f"{coin}: no indicator history available."
    frame = pd.DataFrame(cols)
    lines = [f"{coin} day-by-day lead-up (oldest first; today is the last row):",
             "  date        " + "  ".join(f"{n[:14]:>14}" for n in frame.columns)]
    for ts, row in frame.iterrows():
        vals = "  ".join(f"{row[c]:>14.3f}" if pd.notna(row[c]) else f"{'n/a':>14}" for c in frame.columns)
        lines.append(f"  {ts.date()}  {vals}")
    return "\n".join(lines)


@dataclass(frozen=True)
class Clause:
    """`within_days=0` (the default, and the only behaviour that existed
    before) means "true on the trigger bar itself". `within_days=K` means
    "was true at any point in the last K days, inclusive of today" --
    which is what makes an ORDERED hypothesis expressible at all.

    This exists because the conjunction-only grammar could express
    `news AND crash on the same day` but not `crash, THEN news` -- and
    the second is the central case this project is about (a violent move
    happens, and the explanation arrives a day or two later). With this,
    all three orderings are writable and, more importantly, *comparable*:

        crash then news : shock_zscore >= 3 [within 3d] AND news [today]
        news then crash : is_macro_day = 1  [within 2d] AND shock [today]
        simultaneous    : is_macro_day = 1  [today]     AND shock [today]

    Causality is preserved by construction: the rolling window looks
    strictly BACKWARD, the trigger bar is the day the last clause becomes
    true, and `build_events` still enters at the following bar's open."""
    indicator: str
    op: str
    threshold: float
    within_days: int = 0

    def __post_init__(self):
        if self.indicator not in SUPPORTED_INDICATORS:
            raise ValueError(f"Unsupported indicator '{self.indicator}' -- must be one of {list(SUPPORTED_INDICATORS)}")
        if self.op not in _OPERATORS:
            raise ValueError(f"Unsupported operator '{self.op}' -- must be one of {list(_OPERATORS)}")
        if not isinstance(self.within_days, int) or self.within_days < 0:
            raise ValueError(f"within_days must be a non-negative integer, got {self.within_days!r}")
        if self.within_days > MAX_WITHIN_DAYS:
            raise ValueError(f"within_days must be at most {MAX_WITHIN_DAYS} -- a longer lookback is a market regime, not an event sequence")


# Indicators that are NOT distributionally comparable when recomputed on an
# hourly frame with scale=24, and must therefore be computed on the DAILY
# frame and forward-filled for live detection -- generalising the exception
# `shock_zscore` already had. This is a real, measured train/serve skew, not
# a theoretical concern (BTCUSDT, 1st-99th percentile, daily vs hourly@24):
#
#   rsi_14d              22.3 .. 85.4   ->   42.6 .. 58.5   (0.25x the spread)
#   atr_pct_14d          0.021 .. 0.151 ->   0.004 .. 0.032 (0.22x)
#   daily_range_pct      0.008 .. 0.194 ->   0.001 .. 0.047 (0.24x)
#   efficiency_ratio_20d 0.003 .. 0.760 ->   0.001 .. 0.185 (0.24x)
#
# A condition accepted on `rsi_14d < 30` fires 173 times in the daily
# backtest and CAN NEVER fire in the hourly live scan, because a 336-period
# RSI mean-reverts to ~50 and never reaches 30. The candidate would look
# merely rare rather than broken. Two distinct causes, same consequence:
# a longer window smooths an oscillator toward its midpoint (rsi,
# efficiency_ratio), and a per-bar magnitude measures ONE HOUR's range
# instead of one day's (atr_pct, daily_range_pct -- the latter ignores
# `scale` entirely, since it has no window to scale).
#
# The remaining eight are genuinely comparable and stay hourly-native, which
# is the whole point of hourly detection: they can cross intraday and revert
# before the daily close (see docs/case_study/methodology-decisions.md).
DAILY_NATIVE_INDICATORS = frozenset({
    "shock_zscore", "rsi_14d", "atr_pct_14d", "daily_range_pct", "efficiency_ratio_20d",
})

# Which clauses describe an EVENT (something happened in the world / the
# market) as opposed to a market STATE (what the chart looked like). This
# split is what makes the incremental test possible: drop the event
# clauses and you have the "same market conditions, no event" control
# group. Testing `shock AND X` against an ordinary day is close to
# meaningless -- the shock alone already differs from an ordinary day --
# so the honest question is whether the event adds anything GIVEN X.
# Future news/sentiment indicators belong here too.
EVENT_INDICATORS = frozenset({"is_macro_day", "shock_zscore"})


def clause_to_dict(clause: "Clause") -> dict:
    """THE serializer. Every persisted spec goes through this, so a field
    added to Clause can never be silently dropped on the way to disk --
    which is exactly what would have happened to `within_days`: a
    sequenced condition would round-trip back as a same-day one, quietly
    testing and then tracking a different hypothesis than the one that
    was approved."""
    d = {"indicator": clause.indicator, "op": clause.op, "threshold": clause.threshold}
    if clause.within_days:
        d["within_days"] = clause.within_days
    return d


def clause_from_dict(d: dict) -> "Clause":
    """THE deserializer, and the sanitiser for LLM-produced JSON. Ignores
    unknown keys and coerces `within_days` (a model may emit it as null,
    a float, or a numeric string) rather than raising -- same discipline
    as `_normalize_coin`: never trust model output where code can check
    it instead. A genuinely out-of-range value still raises, via Clause's
    own validation, rather than being silently clamped."""
    within = d.get("within_days") or 0
    if isinstance(within, str):
        within = int(float(within))
    elif isinstance(within, float):
        within = int(within)
    return Clause(indicator=d["indicator"], op=d["op"], threshold=float(d["threshold"]), within_days=int(within))


def reduced_spec(spec: "ConditionSpec") -> "ConditionSpec | None":
    """`spec` with its EVENT clauses removed -- the control condition for
    the incremental test. Returns None when no such contrast exists:
    either the spec has no event clause (nothing to remove) or nothing
    BUT event clauses (removing them leaves no condition at all). In both
    cases the caller correctly falls back to the unconditional baseline,
    which for an event-only spec is exactly the right test anyway."""
    kept = tuple(c for c in spec.clauses if c.indicator not in EVENT_INDICATORS)
    if not kept or len(kept) == len(spec.clauses):
        return None
    return ConditionSpec(label=f"{spec.label}__reduced", clauses=kept,
                         direction=spec.direction, horizons=spec.horizons)


def clause_signal_hourly(clause: Clause, hourly: pd.DataFrame, daily: pd.DataFrame,
                          funding: pd.Series | None) -> pd.Series:
    """One clause -> a boolean Series on an HOURLY index, for live/replay
    trigger detection. THE single shared implementation -- both
    `execution/live_testing.py` and `replay/engine.py` call this rather
    than each hand-rolling the daily-native exception, so production and
    the replay cannot drift apart on what a condition means.

    A `DAILY_NATIVE_INDICATORS` clause is evaluated on the daily frame and
    forward-filled across that day's hours: the condition genuinely means
    "the DAILY statistic crossed its threshold", and there is no valid
    intraday version of it to detect. Everything else is evaluated
    hourly-native at scale=24."""
    if clause.indicator in DAILY_NATIVE_INDICATORS:
        signal = SUPPORTED_INDICATORS[clause.indicator](daily, funding)
        fired = _OPERATORS[clause.op](signal, clause.threshold).fillna(False)
        if clause.within_days:
            fired = fired.rolling(clause.within_days + 1, min_periods=1).max().astype(bool)
        return fired.reindex(hourly.index, method="ffill").fillna(False).astype(bool)
    return clause_signal(clause, hourly, funding, scale=24)


def clause_signal(clause: Clause, daily: pd.DataFrame, funding: pd.Series | None, scale: int = 1) -> pd.Series:
    """One clause -> a boolean Series, applying its own backward-looking
    `within_days` window. THE single implementation, shared by the
    backtest (`scale=1`, daily bars) and the live/replay hourly scans
    (`scale=24`) -- so a sequenced condition can never mean one thing in
    the test and a different thing in live tracking.

    `within_days` is expressed in DAYS and converted by `scale`, matching
    how every indicator window in this module is already reinterpreted on
    an hourly frame."""
    signal = SUPPORTED_INDICATORS[clause.indicator](daily, funding, scale) if scale != 1 else \
        SUPPORTED_INDICATORS[clause.indicator](daily, funding)
    fired = _OPERATORS[clause.op](signal, clause.threshold).fillna(False)
    if clause.within_days:
        bars = clause.within_days * scale
        fired = fired.rolling(bars + 1, min_periods=1).max().astype(bool)
    return fired


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
    # Say WHICH question the p-value answers. "+3% vs. an ordinary day" and
    # "+3% beyond what the same market state predicts without the event" are
    # very different claims and must never be reported in identical words.
    if pattern.get("baseline_kind") == "incremental":
        against = ("vs. the SAME market conditions WITHOUT the event (i.e. what the event term adds "
                   f"on top, n={pattern.get('baseline_n')} control occurrences)")
    else:
        against = "vs. this coin's own ordinary forward returns over the same period"
    return (f"Pattern signal (independent of TP/SL): excess return {pattern['excess_return']:+.2%} {against}, "
            f"p={pattern['p_value']:.3f} ({verdict} at the 5% level, "
            f"one-sided in the traded direction, N={pattern['n']}).{direction_note}\n{risk}")


def condition_desc(spec: ConditionSpec) -> str:
    """Single shared, human-readable rendering of a (possibly multi-
    clause) spec -- every message that shows a novel condition to a
    human (proposal, result, milestone, checkpoint) uses this, so a
    2-clause spec never silently gets rendered as if it had one. Each
    clause's indicator is translated through INDICATOR_PLAIN_NAMES and
    its comparison through OPERATOR_PLAIN, so the rendering never assumes
    the reader already knows what "rsi_14d" or a raw "<"/">" means.

    A lagged clause renders its window explicitly ("at any point in the
    last 3 days") -- without that, "crash then news" and "crash and news
    on the same day" would read identically to a human while testing two
    completely different hypotheses."""
    return " AND ".join(
        f"{INDICATOR_PLAIN_NAMES.get(c.indicator, c.indicator)} {OPERATOR_PLAIN.get(c.op, c.op)} {c.threshold}{_within_phrase(c.within_days)}"
        for c in spec.clauses)


def _within_phrase(within_days: int) -> str:
    """Shared by both renderers (typed spec and raw dict) so the two can
    never describe the same lag differently."""
    if not within_days:
        return ""
    return f" (at any point in the last {within_days} day{'s' if within_days > 1 else ''})"


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

    # The control condition for the incremental test: this same spec with its
    # EVENT clauses removed. None when no such contrast exists (see
    # reduced_spec), in which case the unconditional baseline is used and
    # `baseline_kind` reports that honestly.
    control_spec = reduced_spec(spec)

    all_events, control_events = [], []
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

        def _events_for(clauses) -> pd.DataFrame:
            trig = pd.Series(True, index=daily.index)
            for clause in clauses:
                trig &= clause_signal(clause, daily, funding)
            shock_z = None if is_shock_indicator else shock_zscore_series(daily)
            e = build_events(daily, trig, spec.direction, spec.horizons,
                              shock_z=shock_z, shock_threshold=SHOCK_ZSCORE_THRESHOLD)
            if len(e):
                e["group"] = coin
                e["period"] = e["trigger_time"].dt.year
            return e

        ev = _events_for(spec.clauses)
        if len(ev):
            if is_shock_indicator:
                all_events.append(ev)
            else:
                n_shock_excluded += int((ev["regime"] == "shock").sum())
                all_events.append(ev[ev["regime"] == "normal"])
        if control_spec is not None:
            cev = _events_for(control_spec.clauses)
            if len(cev):
                # Same shock-regime treatment as the treated set, so control and
                # treatment are drawn from the same population in every respect
                # except the event clause being tested.
                control_events.append(cev if is_shock_indicator else cev[cev["regime"] == "normal"])

    if not all_events or not any(len(e) for e in all_events):
        return {"spec": spec, "status": "insufficient_data", "n_raw_triggers": 0, "n_shock_excluded": n_shock_excluded}

    events = pd.concat(all_events, ignore_index=True)
    oos, params_log = walk_forward(events, ohlc_by_coin, spec.direction, cfg)
    rep = report(oos)
    # pattern_significance is now the actual acceptance gate -- see
    # classify_status's own docstring. `rep` (Sortino/win_rate, from the
    # TP/SL-conditioned backtest) is still computed and still reported,
    # but no longer decides accepted/watch/rejected.
    baseline_events = pd.concat(control_events, ignore_index=True) if control_events else None
    pattern = pattern_significance(events, ohlc_by_coin, spec.direction, cfg,
                                    baseline_events=baseline_events)
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
