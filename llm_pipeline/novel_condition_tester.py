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
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from candidates.data_loading import load_daily, load_funding, zscore
from candidates.definitions import trend_efficiency_ratio
from candidates.macro_calendar import macro_release_days
from candidates.macro_vintage import surprise_series
from candidates.methodology import (
    basket_forward_returns,
    MethodologyConfig, build_events, classify_status, concentration_check, pattern_significance, report,
    shock_zscore_series, walk_forward,
)

SHOCK_ZSCORE_THRESHOLD = 2.0  # matches run_battery.py / shock_detector.py -- one consistent definition of "shock" everywhere


def _rsi(df: pd.DataFrame, funding: pd.Series | None, scale: int = 1, window: int = 14, symbol: str | None = None) -> pd.Series:
    window *= scale
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr_pct(df: pd.DataFrame, funding: pd.Series | None, scale: int = 1, window: int = 14, symbol: str | None = None) -> pd.Series:
    """Average True Range as a % of price, so it's comparable across
    coins of very different nominal price -- true range (not just
    high-low) accounts for gaps against the prior close, unlike
    `daily_range_pct`."""
    window *= scale
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return atr / df["close"]


def _donchian_pct(df: pd.DataFrame, funding: pd.Series | None, scale: int = 1, window: int = 20, symbol: str | None = None) -> pd.Series:
    """Where today's close sits within the prior N-day high/low channel:
    0 = at the channel low, 1 = at the channel high, >1 or <0 = broke out
    beyond the channel entirely. Uses only the coin's own already-elapsed
    bars -- no future data, same as every other indicator here."""
    window *= scale
    upper = df["high"].rolling(window).max()
    lower = df["low"].rolling(window).min()
    return (df["close"] - lower) / (upper - lower)


def _bollinger_pctb(df: pd.DataFrame, funding: pd.Series | None, scale: int = 1, window: int = 20, num_std: float = 2.0, symbol: str | None = None) -> pd.Series:
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
    "close_return_1d": lambda df, funding, scale=1, symbol=None: df["close"].pct_change(1 * scale),
    "close_return_5d": lambda df, funding, scale=1, symbol=None: df["close"].pct_change(5 * scale),
    # Ignores `scale` DELIBERATELY, same as shock_zscore above: one bar's
    # high-low range is whatever that bar spans, and there is no window to
    # reinterpret. Safe because it is in DAILY_NATIVE_INDICATORS, so the live
    # scan evaluates it on the daily frame and forward-fills; if it were ever
    # removed from that set, an hourly evaluation would silently measure one
    # HOUR's range against a threshold calibrated on a DAY's (measured:
    # 0.008-0.194 daily vs 0.001-0.047 hourly, a 0.24x shift).
    "daily_range_pct": lambda df, funding, scale=1, symbol=None: (df["high"] - df["low"]) / df["close"],
    # The same quantity measured against its OWN recent level, which is the only
    # form a threshold can be set on. Raw range is badly non-stationary: crypto's
    # volatility roughly halved over this project's window, so `daily_range_pct
    # >= 0.05` selects 62% of days in 2021 and 16% in 2026 -- a filter on the
    # calendar wearing the costume of a filter on market state. Measured spread
    # in yearly selection rate: 54.5% raw, 2.3% z-scored.
    "range_zscore_30d": lambda df, funding, scale=1, symbol=None: zscore(
        (df["high"] - df["low"]) / df["close"], 30 * scale),
    "volume_zscore_30d": lambda df, funding, scale=1, symbol=None: zscore(df["volume"], 30 * scale),
    "funding_zscore_30d": lambda df, funding, scale=1, symbol=None: zscore(funding.reindex(df.index).ffill(), 30 * scale) if funding is not None else pd.Series(np.nan, index=df.index),
    "efficiency_ratio_20d": lambda df, funding, scale=1, symbol=None: trend_efficiency_ratio(df["close"], 20 * scale),
    "is_macro_day": lambda df, funding, scale=1, symbol=None: pd.Series(
        (df.index.floor("D") if scale > 1 else df.index).isin(macro_release_days()), index=df.index).astype(float),
    # the same 'vol-of-vol' shock measure Phase 1's methodology uses to
    # isolate extreme historical events from the static battery -- tested
    # here on ITS OWN excluded population when a live shock fires, giving
    # it a real, walk-forward-validated anchor set rather than an
    # invented one. Deliberately ignores `scale` -- see module note above.
    "shock_zscore": lambda df, funding, scale=1, symbol=None: shock_zscore_series(df),
    # GRADED macro surprises, from the ALFRED vintages already on disk. Unlike
    # `is_macro_day` (a binary "did anything publish today"), these carry HOW FAR
    # the print moved from the previous one -- the difference between a hawkish
    # shock and a nothing-burger, which is most of what a macro event actually
    # means. NaN off release days by construction, so a clause using one fires
    # only on real publication dates; pair with a Clause's `within_days` for
    # "a big surprise landed in the last K days, and today X". Ignore `scale`
    # for the same reason shock_zscore does: a macro release is a calendar-day
    # event with no hourly refinement to reinterpret.
    "cpi_surprise": lambda df, funding, scale=1, symbol=None: surprise_series("cpi", df.index),
    "rate_surprise": lambda df, funding, scale=1, symbol=None: surprise_series("fed_funds_rate", df.index),
    "jobless_claims_surprise": lambda df, funding, scale=1, symbol=None: surprise_series("initial_jobless_claims", df.index),
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
    "range_zscore_30d": "30-day range z-score (how wide today's bar is vs recent bars)",
    "volume_zscore_30d": "30-day volume z-score (how unusual today's volume is)",
    "funding_zscore_30d": "30-day funding-rate z-score",
    "efficiency_ratio_20d": "20-day trend efficiency ratio (0=choppy, 1=straight trend)",
    "is_macro_day": "is a macro-release day (CPI/FOMC)",
    "cpi_surprise": "CPI surprise (how far the new inflation print moved vs. recent prints, in std devs; only on CPI release days)",
    "rate_surprise": "Fed funds rate surprise (how far the new rate print moved vs. recent ones, in std devs; only on release days)",
    "jobless_claims_surprise": "jobless-claims surprise (how far the new claims print moved vs. recent ones, in std devs; only on release days)",
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
    "range_zscore_30d",
    # macro surprises are calendar-day events -- there is no intraday version
    "cpi_surprise", "rate_surprise", "jobless_claims_surprise",
})

# Which clauses describe an EVENT (something happened in the world / the
# market) as opposed to a market STATE (what the chart looked like). This
# split is what makes the incremental test possible: drop the event
# clauses and you have the "same market conditions, no event" control
# group. Testing `shock AND X` against an ordinary day is close to
# meaningless -- the shock alone already differs from an ordinary day --
# so the honest question is whether the event adds anything GIVEN X.
# Future news/sentiment indicators belong here too.
EVENT_INDICATORS = frozenset({
    "is_macro_day", "shock_zscore",
    "cpi_surprise", "rate_surprise", "jobless_claims_surprise",
})

# THE NECESSARY CONDITION. This project asks whether specific market
# conditions COMBINED WITH a piece of news or a real-world event produce a
# repeatable pattern. A condition built only from price/volume/funding
# indicators does not test that question at all -- it is a chart pattern,
# indistinguishable from something written directly in Freqtrade with no
# LLM involved. Enforced structurally in ConditionSpec rather than
# requested in a prompt, for the same reason `_normalize_coin` corrects a
# coin symbol instead of trusting the instruction: a prompt is a request,
# code is a guarantee.
#
# `shock_zscore` is deliberately NOT in this set. A violent price move is
# a MARKET event, not news -- and "the price moved a lot, then the price
# did something" is exactly the tautology this requirement exists to
# exclude. A shock is a perfectly good ADDITIONAL clause (it is a market
# condition), it just cannot be the thing that makes a hypothesis
# on-thesis on its own.
#
# Measured before this rule existed: of 92 conditions Sonnet proposed in
# the replay, 31 contained no event term whatsoever and a further 49
# carried only a shock -- 80 of 92 were off-thesis, and every one of them
# was tested, graded and reported as if it answered the project's
# question. News/sentiment indicators join this set once a historical
# archive exists to backtest them against (see docs/case_study/TODO.md).
# What counts as NEWS for the necessary-condition rule. `is_macro_day` is
# deliberately NOT here, though it remains available as a context term.
#
# It is a CALENDAR fact, not an information event: it says "a publication was
# scheduled today", never what the publication said, and release dates are known
# months in advance. After jobless claims joined the calendar it fires on 18.9%
# of days -- roughly one day in five -- so a condition resting on it alone reads
# as "on a day when something came out, whatever it was". Measured on 118
# proposals from a real replay, 21% used it as their only news term, which is a
# fifth of the discovery budget spent on hypotheses that cannot answer this
# project's question.
#
# The graded surprises carry the content: how far a print moved relative to how
# far that series usually moves. Same reasoning that already excludes
# shock_zscore -- a violent price move is a market event, not news; a scheduled
# date is not news either. The news is what the number said.
NEWS_EVENT_INDICATORS = frozenset({
    "cpi_surprise", "rate_surprise", "jobless_claims_surprise",
})

# Indicators an LLM proposal may NOT use at all, as opposed to those that merely
# fail to satisfy the necessary-condition rule on their own.
#
# `is_macro_day` is contentless. It says a publication was scheduled today and
# never what the publication said -- a condition resting on it reads as "on a day
# when something came out, whatever it was", which cannot distinguish a hawkish
# shock from a print that landed exactly on consensus. Release dates are also
# known months ahead, so it carries no information that arrives.
#
# Dropping it from NEWS_EVENT_INDICATORS alone left it usable as a secondary
# clause, which is not worth the grammar: paired with a graded surprise it is
# very nearly redundant (a CPI surprise IS a macro day), and paired with anything
# else it re-admits through the back door the hypothesis it was removed to
# exclude. It is not a weaker version of `cpi_surprise`; it is the version with
# the content removed.
#
# Enforced in code rather than only in the prompt, on this project's standing
# rule that a prompt is a request and code is a guarantee -- and because the two
# had already drifted apart: the prompt listed `is_macro_day` as satisfying the
# HARD REQUIREMENT while the code rejected exactly that, so the system was paying
# for proposals its own instructions induced and its own validator refused. 21%
# of 118 real proposals died that way.
#
# Deliberately NOT removed from SUPPORTED_INDICATORS. The committed sweeps in
# `forecast/` (grammar_sweep, sentiment_power, coin_specific_test) contain arms
# built on it, and their recorded JSON results must stay reproducible -- deleting
# the indicator would silently invalidate published measurements to tidy up a
# grammar rule that belongs to the proposal path only.
# `shock_zscore` is banned for a different and sharper reason: it is the
# EXPLANANDUM. The replay asks Sonnet precisely because a shock occurred, so a
# condition containing a shock restates the reason the question was asked.
#
# The decisive form of the argument is about information, not aesthetics: since
# the trigger IS a shock, a shock is present at every proposal by construction.
# It is a constant of the sampling frame, not a variable -- so it can add nothing
# discriminating when the hypothesis is formed, while still narrowing the
# condition when it is later tested over all history. That is the worst of both.
#
# Measured on 118 real proposals: 11 contained `shock_zscore`, and 9 of those 11
# used `within_days=0` -- "a shock TODAY", the very day that prompted the
# question. Only 2 used it as a genuine antecedent. The sequenced form is more
# defensible, but it is 2 cases out of 118, and it is still the model proposing
# the thing it was shown; a rule with an exception that rare is harder to reason
# about than the rule.
#
# `daily_range_pct` is banned as non-stationary, not as circular. See
# `range_zscore_30d` above: a fixed threshold on the raw range is a filter on the
# year. Neither it nor `atr_pct_14d` appeared in any of the 118 proposals, so
# nothing is lost in practice. Both stay in SUPPORTED_INDICATORS because the
# committed sweeps in `forecast/` are built on them and must stay reproducible.
NON_PROPOSABLE_INDICATORS = frozenset({"is_macro_day", "shock_zscore", "daily_range_pct"})

# The whitelist shown to the model, derived rather than written out by hand: an
# indicator added to NON_PROPOSABLE_INDICATORS disappears from every prompt at
# once. The hand-maintained version is how `is_macro_day` stayed advertised in
# the HARD REQUIREMENT after the validator started rejecting it.
def proposable_indicators() -> list[str]:
    return [k for k in SUPPORTED_INDICATORS if k not in NON_PROPOSABLE_INDICATORS]



# Words in a LABEL that assert a macro/news event is part of the hypothesis.
# Checked against the actual clauses, because a name that describes a test
# it doesn't perform is the same class of defect as every other one found
# in this codebase: a label sitting next to numbers it doesn't describe.
_MACRO_CLAIM_RE = re.compile(r"cpi|fomc|macro|fed[\W_]|jobless|claims|rate[\W_]?(decision|hike|cut)|news|headline", re.I)


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


def spec_from_proposal(d: dict) -> "tuple[ConditionSpec | None, str | None]":
    """Turn Sonnet's raw `novel_condition_spec` JSON into a validated
    ConditionSpec, returning (spec, None) on success or (None, reason) on
    rejection -- never raising.

    Used at PROPOSAL time, not at test time, and that ordering matters. A
    proposal that fails the necessary-condition rule must be discarded the
    moment it is made: it should not be stored as pending, must not halt
    the replay waiting for a human decision about a hypothesis the system
    will refuse to test anyway, and must never surface as a Test It button
    that would fail on press. Rejecting late instead of early was a live
    crash risk in the replay, which saves the raw dict and halts before
    anything validates it."""
    try:
        spec = spec_from_dict(d)
    except (ValueError, KeyError, TypeError) as e:
        return None, str(e)
    banned = sorted({c.indicator for c in spec.clauses} & NON_PROPOSABLE_INDICATORS)
    if banned:
        return None, (f"uses {', '.join(banned)}, which is not proposable: it records that a "
                      f"release was scheduled, never what it said. Use the graded surprise "
                      f"({', '.join(sorted(NEWS_EVENT_INDICATORS))}) instead.")
    return spec, None


# A condition must have occurred at least this many times across the coin
# universe to be worth testing. Below it there is nothing to measure either way,
# and the backtest returns `insufficient_data` -- 193 of 234 candidates in a real
# replay ended there.
#
# This REPLACED a cap on the number of clauses. That cap was an approximation of
# rarity and a poor one: measured on 228 real proposals, 2-clause and 3-clause
# conditions were equally testable (18% each) while every 4-clause one failed --
# so the count correlates with rarity without determining it. A 4-clause
# condition with wide thresholds can fire 300 times; a 2-clause one with extreme
# thresholds can fire 11. The cap blocked the first and admitted the second,
# which is backwards. Counting occurrences takes 0.25s and no API call, so the
# thing that actually matters can simply be measured.
# 35, not 30. Out-of-sample is roughly two thirds of all occurrences (the first
# three years are training folds), so 30 total lands at ~20 OOS -- exactly ON
# classify_status's min_report_events, where it would be admitted here and then
# rejected there. 35 gives ~23 OOS and a little margin, since the two-thirds
# ratio is itself approximate.
MIN_HISTORICAL_OCCURRENCES = 35


def count_occurrences(spec: "ConditionSpec", coins: list[str],
                       as_of: "pd.Timestamp | None" = None) -> int:
    """How many times this condition had fired as of `as_of`, across `coins`.

    Cheap and exact: ~0.25s of local computation, no API call. Used to reject a
    proposal that cannot produce a result BEFORE spending a walk-forward test on
    it.

    `as_of` IS NOT OPTIONAL IN THE REPLAY, and passing it is the whole
    correctness of this function there. Counted over the full history, a
    condition can look plentiful because of occurrences that have not happened
    yet: `jobless_claims_surprise >= 0.5 AND rsi <= 55` has 617 occurrences today
    and had 37 by mid-2018. Deciding in 2018 to test it on the strength of 617
    would be reading the future to choose what to test -- exactly the leak
    replay/time_sandbox.py exists to prevent, arriving through the back door of a
    gate meant to save money.

    Production passes None, which means "as of now", and there the full history
    IS everything knowable.
    """
    return len(occurrence_set(spec, coins, as_of))


def occurrence_set(spec: "ConditionSpec", coins: list[str],
                    as_of: "pd.Timestamp | None" = None) -> set:
    """The exact (coin, day) pairs on which this condition fired, as of `as_of`.

    Exists so two conditions can be compared by what they DO rather than by how
    they are written. Two specs proposed by different models will never be
    textually identical -- different thresholds, different clause order,
    sometimes a different indicator expressing the same idea -- yet if they
    select the same days they are the same hypothesis, and if they select
    disjoint days they are not, whatever their labels say.

    `count_occurrences` is now a length over this, so the two can never
    disagree about what counts as an occurrence."""
    from candidates.data_loading import load_daily, load_funding

    fired = set()
    for coin in coins:
        try:
            daily, funding = load_daily(coin), load_funding(coin)
        except Exception:
            continue
        if as_of is not None:
            daily = daily.loc[:as_of]
            funding = funding.loc[:as_of] if funding is not None else None
        if spec.coins and coin not in spec.coins:
            continue
        sig = pd.Series(True, index=daily.index)
        for clause in spec.clauses:
            sig &= clause_signal(clause, daily, funding, symbol=coin)
        sig = sig.fillna(False)
        fired |= {(coin, ts) for ts in sig.index[sig.to_numpy().astype(bool)]}
    return fired


def behavioural_agreement(a: "ConditionSpec", b: "ConditionSpec", coins: list[str],
                           as_of: "pd.Timestamp | None" = None) -> float:
    """Jaccard overlap of the days two conditions fire on: 1.0 identical
    behaviour, 0.0 disjoint. NaN when neither ever fires, which is not
    agreement -- two conditions that never happen are not the same hypothesis,
    they are both untestable, and averaging that in as a 1.0 would flatter
    whichever model proposes impossible conditions."""
    sa = occurrence_set(a, coins, as_of)
    sb = occurrence_set(b, coins, as_of)
    union = sa | sb
    if not union:
        return float("nan")
    return len(sa & sb) / len(union)


# The value at which each indicator says NOTHING -- the anchor `_loosen` moves
# thresholds toward and is forbidden to cross. RSI's midline, and consensus (a
# zero surprise, a flat return, an average volume) for everything measured as a
# deviation. An indicator absent from this table is never relaxed, which is the
# safe default: an unknown scale has no known neutral, and guessing one would
# reintroduce exactly the sign-flip this table exists to prevent.
RELAXATION_NEUTRAL: dict[str, float] = {
    "rsi_14d": 50.0,
    "bollinger_pctb_20d": 0.5,
    "donchian_pct_20d": 0.5,
    "cpi_surprise": 0.0,
    "rate_surprise": 0.0,
    "jobless_claims_surprise": 0.0,
    "shock_zscore": 0.0,
    "volume_zscore_30d": 0.0,
    "range_zscore_30d": 0.0,
    "funding_zscore_30d": 0.0,
    "close_return_1d": 0.0,
    "close_return_3d": 0.0,
    "close_return_5d": 0.0,
    "close_return_7d": 0.0,
    "close_return_14d": 0.0,
}

# How far a threshold may be loosened, in order, stopping at the first level that
# clears MIN_HISTORICAL_OCCURRENCES. Small steps first so the tested hypothesis
# stays as close as possible to the one that was proposed.
RELAXATION_STEPS = (0.10, 0.25, 0.50)


def relax_to_testable(spec: "ConditionSpec", coins: list[str],
                      as_of: "pd.Timestamp | None" = None) -> "tuple[ConditionSpec, str] | None":
    """Loosen a too-rare condition just enough to be measurable, or give up.

    A proposal can be directionally sensible and still untestable because its
    thresholds are extreme -- "CPI surprise above 2 sigma AND a 25% five-day
    fall" describes about one day in this project's whole history. Rejecting it
    outright discards the idea along with the numbers. Loosening the numbers
    keeps the idea.

    WHY THIS IS NOT P-HACKING, which is the obvious objection. The search
    criterion is the OCCURRENCE COUNT and nothing else: no forward return, no
    p-value, no outcome of any kind is consulted while choosing a threshold. It
    is a power calculation, not a result search. Loosening until a condition
    fires often enough to be measured is legitimate; loosening until it becomes
    significant would not be, and this function cannot do that because it never
    sees a return.

    Two further properties keep it honest:
      * the SMALLEST relaxation that works is taken, so the tested hypothesis is
        the nearest measurable neighbour of the proposed one, not the loosest;
      * `as_of` is respected, so a replay never counts occurrences that have not
        happened yet in order to decide how far to loosen.

    Returns `(relaxed_spec, human-readable note)`, or None when even the loosest
    step leaves too little to measure. The note exists so the change is recorded
    with the candidate rather than applied invisibly -- the tested condition is
    not the proposed one, and a reader has to be able to see that.
    """
    for step in RELAXATION_STEPS:
        clauses = tuple(_loosen(c, step) for c in spec.clauses)
        try:
            candidate = ConditionSpec(label=spec.label, clauses=clauses, direction=spec.direction,
                                       horizons=spec.horizons, coins=spec.coins,
                                       outcome=spec.outcome, prior_weight=spec.prior_weight)
        except ValueError:
            return None
        if count_occurrences(candidate, coins, as_of=as_of) >= MIN_HISTORICAL_OCCURRENCES:
            changed = ", ".join(f"{a.indicator} {a.op} {a.threshold:g} -> {b.threshold:g}"
                                for a, b in zip(spec.clauses, clauses) if a.threshold != b.threshold)
            return candidate, f"thresholds loosened by {step:.0%} to reach a measurable sample ({changed})"
    return None


def _loosen(clause: "Clause", step: float) -> "Clause":
    """Move one threshold `step` of the way from its current value toward the
    indicator's NEUTRAL point -- never past it, and never onto it.

    Measuring the step against the threshold's own magnitude, which is the
    obvious implementation, silently destroys the hypothesis it claims to be
    preserving. Two failures, both observed on real proposals from this
    project's own replay state:

      * "hot CPI with the market OVERBOUGHT, RSI >= 70" loosened to RSI >= 52.5.
        52.5 is not overbought, it is the neutral line; the relaxed condition
        tests something the proposal never claimed. Half the history qualifies,
        which is exactly why the occurrence count looked healthy.
      * "COOL CPI, surprise <= 0" loosened to surprise <= +0.1 -- the threshold
        crossed zero and the condition began matching HOT prints. A magnitude of
        zero has no scale to take a percentage of, so the code fell back to 1.0
        and moved the boundary in absolute units through the sign change.

    Anchoring on the neutral point fixes both, and fixes them for the same
    reason: `neutral` is what makes a threshold MEAN something. RSI 70 is
    overbought because 50 is neutral; a surprise of -1 is dovish because 0 is
    consensus. Loosening toward neutral weakens the claim, which is the
    intent; reaching neutral empties it; passing neutral inverts it.

    A threshold sitting AT its neutral point therefore cannot be loosened at
    all -- distance is zero, so every step is zero. That is correct rather than
    a limitation: `cpi_surprise <= 0` is already "any cool print", and there is
    no weaker version of it that is still the same hypothesis.
    """
    neutral = RELAXATION_NEUTRAL.get(clause.indicator)
    if neutral is None or clause.op not in ("<", "<=", ">", ">="):
        return clause
    distance = clause.threshold - neutral
    if distance == 0:
        return clause
    # Toward neutral is the permissive direction for a threshold on the far
    # side of it, which is the only case that arises: a `>=` clause sits above
    # neutral, a `<=` clause below.
    new = clause.threshold - distance * step
    return Clause(indicator=clause.indicator, op=clause.op, threshold=round(new, 4),
                  within_days=clause.within_days)


def spec_to_dict(spec: "ConditionSpec") -> dict:
    """THE serializer for a ConditionSpec. One implementation, deliberately.

    Every field this project has ever added to `ConditionSpec` has been
    dropped by at least one hand-rolled serializer: `within_days` was
    silently lost by all three of them at once, so a sequenced "crash, THEN
    news" hypothesis round-tripped back as a same-day one and was tested as
    a different claim than the one a human approved. `coins` and `outcome`
    are exactly as easy to lose and would fail the same way -- an XRP-scoped,
    market-relative spec would come back as a whole-universe raw-return spec
    with the same label, and nothing would look wrong.

    Optional fields are omitted when they hold their default, so existing
    registry files stay byte-identical until a spec actually uses one.
    """
    d = {"label": spec.label,
         "clauses": [clause_to_dict(c) for c in spec.clauses],
         "direction": spec.direction,
         "horizons": list(spec.horizons)}
    if spec.coins:
        d["coins"] = list(spec.coins)
    if spec.outcome != "raw":
        d["outcome"] = spec.outcome
    if spec.prior_weight != 1.0:
        d["prior_weight"] = spec.prior_weight
    return d


def spec_from_dict(d: dict) -> "ConditionSpec":
    """THE deserializer. Tolerant of dicts written before `coins`/`outcome`
    existed (both fall back to their defaults), strict about everything
    `ConditionSpec.__post_init__` validates."""
    kwargs = {"label": str(d["label"]),
              "clauses": tuple(clause_from_dict(c) for c in d["clauses"]),
              "direction": str(d["direction"])}
    if d.get("horizons"):
        kwargs["horizons"] = tuple(int(h) for h in d["horizons"])
    if d.get("coins"):
        kwargs["coins"] = tuple(str(c) for c in d["coins"])
    if d.get("outcome"):
        kwargs["outcome"] = str(d["outcome"])
    if d.get("prior_weight") is not None:
        kwargs["prior_weight"] = float(d["prior_weight"])
    return ConditionSpec(**kwargs)


def reduced_clauses(spec: "ConditionSpec") -> "tuple[Clause, ...] | None":
    """`spec`'s clauses with the EVENT ones removed -- the CONTROL group
    for the incremental test ("same market state, no event").

    Returns a bare clause tuple, deliberately NOT a ConditionSpec: the
    control is by construction the version WITHOUT an event clause, so it
    could never satisfy ConditionSpec's own necessary-condition rule. That
    rule governs what may be PROPOSED and tested as a hypothesis; the
    control group is an internal construct, not a hypothesis, and forcing
    it through the same validation would be a category error (and did
    briefly break every incremental test when it was one).

    None when no contrast exists: either nothing but event clauses (the
    control would be empty) or -- unreachable now that an event clause is
    mandatory -- no event clause to remove."""
    kept = tuple(c for c in spec.clauses if c.indicator not in EVENT_INDICATORS)
    return kept or None


def clause_signal_hourly(clause: Clause, hourly: pd.DataFrame, daily: pd.DataFrame,
                          funding: pd.Series | None, symbol: str | None = None) -> pd.Series:
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
        signal = SUPPORTED_INDICATORS[clause.indicator](daily, funding, symbol=symbol)
        fired = _OPERATORS[clause.op](signal, clause.threshold).fillna(False)
        if clause.within_days:
            fired = fired.rolling(clause.within_days + 1, min_periods=1).max().astype(bool)
        return fired.reindex(hourly.index, method="ffill").fillna(False).astype(bool)
    return clause_signal(clause, hourly, funding, scale=24, symbol=symbol)


def clause_signal(clause: Clause, daily: pd.DataFrame, funding: pd.Series | None, scale: int = 1,
                   symbol: str | None = None) -> pd.Series:
    """One clause -> a boolean Series, applying its own backward-looking
    `within_days` window. THE single implementation, shared by the
    backtest (`scale=1`, daily bars) and the live/replay hourly scans
    (`scale=24`) -- so a sequenced condition can never mean one thing in
    the test and a different thing in live tracking.

    `within_days` is expressed in DAYS and converted by `scale`, matching
    how every indicator window in this module is already reinterpreted on
    an hourly frame.

    `symbol` is WHICH COIN this frame belongs to. Every indicator accepts it
    and almost all ignore it -- RSI does not care what it is computing on.
    It exists for indicators that are coin-attributed by nature: a news or
    sentiment score about Ripple is a fact about XRP and about nothing else,
    and without the symbol such an indicator cannot be written at all. The
    absence of this parameter was a real blocker rather than an inelegance:
    a test of a single-coin pattern had to identify the coin by its price
    series LENGTH, which is not something that could ship."""
    signal = SUPPORTED_INDICATORS[clause.indicator](daily, funding, scale, symbol=symbol) if scale != 1 else \
        SUPPORTED_INDICATORS[clause.indicator](daily, funding, symbol=symbol)
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
    statistical (each extra AND shrinks the sample, and classify_status
    already gates on a minimum event count), not an arbitrary limit here."""
    label: str
    clauses: tuple[Clause, ...]
    direction: str  # "long" or "short"
    horizons: tuple[int, ...] = (1, 3, 7, 14, 21)
    # WHICH COINS this hypothesis is about. None = the whole universe, the
    # default and the only thing that existed before. A tuple names the coins
    # the claim is restricted to -- "SEC sues Ripple" is a fact about XRP and
    # testing it on DOGE adds noise, not evidence.
    coins: tuple[str, ...] | None = None
    # WHICH OUTCOME the forward return is measured against.
    #   "raw"             -- the coin's own return. Correct for a MARKET-WIDE
    #                        event: a CPI print moves all of crypto together, so
    #                        subtracting the market would delete the very effect.
    #   "market_relative" -- return minus the equal-weight basket. Correct for a
    #                        COIN-SPECIFIC event, and a large power gain there:
    #                        mean cross-coin correlation of forward returns is
    #                        0.54, so most of any coin's move IS the market's,
    #                        and removing it removes mostly noise. Measured 2/9
    #                        -> 6/9 accepted on a coin-specific planted signal.
    # Measured BOTH ways, because the choice genuinely cuts both directions: on
    # a market-timing signal the same switch made things WORSE (10 accepted ->
    # 7), since it removed the effect faster than the noise. It is therefore a
    # per-hypothesis DECLARATION, never a global setting -- and it is declared
    # up front, because picking it after seeing which measurement scored better
    # would be choosing the answer.
    outcome: str = "raw"
    # PROPOSAL-TIME plausibility, 0.25-4.0, neutral 1.0. Feeds prior-weighted
    # Benjamini-Hochberg (Genovese, Roeder & Wasserman 2006), which redistributes
    # the family's error budget rather than enlarging it: a hypothesis argued for
    # in advance gets a larger share of alpha, and the rest of the family pays for
    # it. Measured in simulation at this project's own power and family size, even
    # a WEAK prior (correlation 0.2 with the truth) yields ~45% more true
    # discoveries, and realised FDR stays at or under alpha at every prior quality
    # -- so a useless prior is merely wasteful, never dangerous.
    #
    # It is load-bearing that this is set when the condition is PROPOSED and never
    # revised afterwards. A weight raised because a result looked good is not a
    # prior; it is choosing the answer, and it voids the guarantee outright.
    prior_weight: float = 1.0

    def __post_init__(self):
        if len(self.clauses) == 0:
            raise ValueError("ConditionSpec needs at least one clause")
        if self.direction not in ("long", "short"):
            raise ValueError("direction must be 'long' or 'short'")
        if self.outcome not in ("raw", "market_relative"):
            raise ValueError(f"outcome must be 'raw' or 'market_relative', got {self.outcome!r}")
        if self.coins is not None and len(self.coins) == 0:
            raise ValueError("coins must be None (whole universe) or a non-empty tuple")
        # Clamped rather than rejected: a model returning 50 is expressing
        # enthusiasm, not a considered 50x error budget, and the range keeps any
        # single proposal from swallowing the family's alpha.
        if not (self.prior_weight == self.prior_weight) or self.prior_weight <= 0:
            raise ValueError(f"prior_weight must be a positive number, got {self.prior_weight!r}")
        object.__setattr__(self, "prior_weight", float(min(max(self.prior_weight, 0.25), 4.0)))
        # The necessary condition -- see NEWS_EVENT_INDICATORS.
        if not any(c.indicator in NEWS_EVENT_INDICATORS for c in self.clauses):
            raise ValueError(
                f"ConditionSpec '{self.label}' has no news/macro event clause. This project tests market "
                f"conditions COMBINED WITH a real-world event; an event term is a NECESSARY condition, not an "
                f"option. Add at least one of {sorted(NEWS_EVENT_INDICATORS)}. "
                f"(shock_zscore is a market event, not news -- it does not satisfy this on its own.) "
                f"Got: {[c.indicator for c in self.clauses]}")
        # The label must not claim an event the clauses don't implement. Real,
        # measured case: "post-CPI extreme volume+breakout momentum" tested only
        # volume_zscore AND bollinger_pctb -- no CPI clause at all -- and 14 of 92
        # proposals had a label asserting a macro event their spec never contained.
        # A human reading /summary reasonably believes the name describes the test.
        if _MACRO_CLAIM_RE.search(self.label) and not any(
                c.indicator in NEWS_EVENT_INDICATORS for c in self.clauses):
            raise ValueError(f"ConditionSpec '{self.label}' names a macro event its clauses do not contain")


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
    # reduced_clauses), in which case the unconditional baseline is used and
    # `baseline_kind` reports that honestly.
    control_clauses = reduced_clauses(spec)

    # A coin-scoped spec is tested ONLY where it claims to apply. Testing an
    # XRP-specific hypothesis on DOGE adds noise, not evidence. `spec.coins`
    # is intersected with the caller's list rather than replacing it, so a
    # caller that legitimately restricts the universe (the replay's own coin
    # set, say) is never silently overridden by the spec.
    if spec.coins:
        coins = [c for c in coins if c in set(spec.coins)]
        if not coins:
            return {"spec": spec, "status": "insufficient_data", "n_raw_triggers": 0,
                    "n_shock_excluded": 0,
                    "note": f"spec is scoped to {list(spec.coins)}, none of which were available"}

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
                trig &= clause_signal(clause, daily, funding, symbol=coin)
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
        if control_clauses is not None:
            cev = _events_for(control_clauses)
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
    # The basket is passed as DATA, not installed by patching the module -- both
    # the treated returns and the baseline must be measured the same way, and a
    # patch that reached one but not the other would compare two different
    # quantities and report the difference as a discovery.
    basket = basket_forward_returns(spec.horizons) if spec.outcome == "market_relative" else None
    pattern = pattern_significance(events, ohlc_by_coin, spec.direction, cfg,
                                    baseline_events=baseline_events, basket=basket)
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
    # A spec scoped to ONE coin cannot meaningfully fail a coin-concentration
    # check: it is 100% concentrated in that coin by its own declaration, and
    # penalising it for that would make a genuine single-coin pattern
    # unacceptable by construction (measured: a real XRP-only signal with
    # p=0.016, +21.45% excess and MFE/MAE 8.57 was held at `watch` for exactly
    # this reason). The check is skipped only when the restriction was DECLARED
    # in the spec up front -- never inferred afterwards from which coin happened
    # to dominate, which would be choosing the answer. The year check is
    # untouched: a single-coin pattern still has to hold across time.
    single_coin = bool(spec.coins) and len(spec.coins) == 1
    status = classify_status(rep, {"concentrated": False, "max_group_share": 1.0,
                                    "dominant_group": spec.coins[0]} if single_coin else coin_conc,
                              year_conc, pattern, cfg)

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
