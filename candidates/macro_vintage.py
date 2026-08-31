"""Real, dated FRED/ALFRED vintage macro release data -- every row
carries its own `realtime_start` (when that number actually became
public, not the period it describes), so a CPI print for June isn't
"known" until its real July publication date. Shared by production
(reads with `as_of=None`, meaning "as of right now") and the historical
replay (`replay/time_sandbox.py`, which always passes its own simulated
`as_of` so it never sees data published after that simulated date) --
one reader, one file per series, no separate "simulation" copy of this
logic.
"""
from __future__ import annotations

import pandas as pd

from candidates.data_loading import DATA_DIR

VINTAGE_DIR = DATA_DIR / "macro" / "fred_vintage"

MACRO_SERIES = {"cpi": "CPI", "fed_funds_rate": "Fed Funds Rate", "initial_jobless_claims": "Initial Jobless Claims"}


def _load_vintage(name: str) -> pd.DataFrame:
    df = pd.read_csv(VINTAGE_DIR / f"{name}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    return df


def vintage_releases_as_of(name: str, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Every (period, value) this series has ever reported, as they stood
    on `as_of` (default: right now) -- for periods later revised, keeps
    only the latest revision whose `realtime_start` is not itself in the
    future relative to `as_of`. Sorted by the period the value
    describes."""
    as_of = pd.Timestamp.now() if as_of is None else as_of
    df = _load_vintage(name)
    known = df[df["realtime_start"] <= as_of]
    if len(known) == 0:
        return known
    value_col = [c for c in known.columns if c not in ("date", "realtime_start")][0]
    latest_per_period = known.sort_values("realtime_start").groupby("date", as_index=False).last()
    return latest_per_period.sort_values("date")[["date", "realtime_start", value_col]]


def surprise_series(name: str, index: pd.DatetimeIndex, window: int = 12) -> pd.Series:
    """A graded MACRO SURPRISE aligned to `index`: on each day this series
    actually published, how far the new print moved from the previous one,
    as a z-score against the trailing distribution of such moves. NaN on
    every other day -- the indicator IS the event, so a clause built on it
    fires only on release days by construction, and combines with a
    Clause's own `within_days` to express "a big surprise landed within
    the last K days".

    Why this exists. `is_macro_day` is a BINARY flag: "was there a release
    today". It cannot tell a hawkish shock from a nothing-burger, which is
    most of the information a macro event carries. This project already
    downloads the ALFRED vintages needed to grade it, and
    `latest_release_with_prior` already computes the release-vs-prior
    delta for Sonnet's prompt -- but that number was never exposed as a
    testable indicator, so the pipeline computed the surprise, showed it
    to the model, and discarded it at exactly the point where it would
    have become a falsifiable hypothesis.

    Point-in-time correct in two distinct senses, both load-bearing:
      * keyed to `realtime_start` (the day it was PUBLISHED), never to the
        period it describes, so a figure is never known before it existed;
      * the z-score's trailing mean/std are `.shift(1)`-ed, so "how
        surprising is this" is judged only on releases that preceded it.
    Uses each period's FIRST print, not its latest revision -- the market
    reacted to the number as originally published.

    A DEGENERATE trailing window yields NaN, not a colossal z-score. Real
    case caught on this data: the Fed funds rate sits flat for years at a
    time, so its rolling std collapses toward zero and a bare `sd > 0`
    guard produced a "surprise" of -2,613,348 standard deviations on the
    first move afterwards. That is not an informative reading, it is a
    division artifact -- and it would have been silently comparable
    against any threshold Sonnet proposed. When the last several releases
    were effectively identical there is genuinely no scale to judge
    surprise against, and saying so (NaN) is the honest answer."""
    df = _load_vintage(name)
    value_col = [c for c in df.columns if c not in ("date", "realtime_start")][0]
    first = df.sort_values("realtime_start").groupby("date", as_index=False).first()
    first = first.sort_values("realtime_start").reset_index(drop=True)
    delta = first[value_col].astype(float).diff()
    mu = delta.rolling(window, min_periods=max(window // 2, 3)).mean().shift(1)
    sd = delta.rolling(window, min_periods=max(window // 2, 3)).std().shift(1)
    # Scale-free floor from the series' own typical move, so this works
    # identically for a rate in percent and claims in hundreds of thousands.
    typical = float(delta.abs().median(skipna=True) or 0.0)
    floor = max(typical * 1e-2, 1e-12)
    z = (delta - mu) / sd.where(sd > floor)
    by_pub = pd.Series(z.to_numpy(), index=pd.DatetimeIndex(first["realtime_start"]).floor("D"))
    by_pub = by_pub[~by_pub.index.duplicated(keep="last")]
    target = pd.DatetimeIndex(index).floor("D")
    return pd.Series(by_pub.reindex(target).to_numpy(), index=index, dtype=float)


def release_dates(name: str, start: pd.Timestamp, end: pd.Timestamp,
                   new_periods_only: bool = False) -> pd.DatetimeIndex:
    """Every distinct real-world date this series published anything within
    [start, end] -- real, dated events rather than an approximation.

    `new_periods_only=True` keeps only dates on which a period was published
    for the FIRST time, dropping days that carry nothing but revisions of
    periods already released.

    Why that distinction is worth a parameter. ALFRED records a
    `realtime_start` for revisions too, and the annual seasonal-adjustment
    pass re-publishes years of history at once. Measured over 2018-2026, 37
    of 687 firings (5.4%, across 19 distinct days) landed on days where no
    new period appeared at all -- and they were not scattered: almost every
    one falls on 1 JANUARY, when no US statistical agency releases anything,
    plus the February CPI seasonal-factor revisions.

    On such a day the replay told Sonnet "CPI was released", while
    `latest_release_with_prior` handed back the latest period's value
    UNCHANGED from what it had already seen -- an event that did not happen,
    described as though it had, for 5.4% of the macro discovery budget. Not a
    lookahead (the revision genuinely is dated that day), but not an
    information event either.

    The default stays False because the other two callers want revisions
    included: `recent_releases_summary` reports what became public, and
    `macro_calendar` builds the release calendar, where a revision is a real
    publication. Only the replay's event trigger asks "did something NEW come
    out today", and only it passes True."""
    df = _load_vintage(name)
    in_window = (df["realtime_start"] >= start) & (df["realtime_start"] <= end)
    if new_periods_only:
        # A period's first publication, not its latest -- the market reacted to
        # the number as originally printed, same convention as surprise_series.
        first_pub = df.sort_values("realtime_start").groupby("date")["realtime_start"].min()
        in_window &= df["realtime_start"].isin(set(first_pub))
    return pd.DatetimeIndex(sorted(df["realtime_start"][in_window].unique()))


def latest_release_with_prior(name: str, as_of: pd.Timestamp | None = None) -> dict | None:
    """The most recent real release as of `as_of` (default: right now),
    plus the release immediately before it, for a simple, honest
    surprise measure (current vs. the last known value -- there's no
    consensus-estimate data source in this project, so "surprise" here
    means "how much did the real number move from what was already
    known," not "vs. analyst expectations," and is described that way
    wherever it's used)."""
    releases = vintage_releases_as_of(name, as_of)
    if len(releases) < 1:
        return None
    value_col = [c for c in releases.columns if c not in ("date", "realtime_start")][0]
    latest = releases.iloc[-1]
    prior = releases.iloc[-2] if len(releases) >= 2 else None
    return {
        "period": latest["date"], "realtime_start": latest["realtime_start"], "value": float(latest[value_col]),
        "prior_value": float(prior[value_col]) if prior is not None else None,
    }


def recent_releases_summary(as_of: pd.Timestamp | None = None, lookback_days: int = 10) -> str:
    """Plain-text digest of every MACRO_SERIES release in the last
    `lookback_days` before `as_of` (default: right now) -- for giving
    Sonnet real, dated macro context alongside a shock event, without
    it needing to know how to read the vintage files itself."""
    as_of = pd.Timestamp.now() if as_of is None else as_of
    start = as_of - pd.Timedelta(days=lookback_days)
    lines = []
    for key, label in MACRO_SERIES.items():
        for d in release_dates(key, start, as_of):
            release = latest_release_with_prior(key, d)
            if release is None:
                continue
            change = f", change from prior release: {release['value'] - release['prior_value']:+.3f}" if release["prior_value"] is not None else " (first known release)"
            lines.append(f"{label}, published {d.date()}, for period {release['period'].date()}: value={release['value']}{change}")
    return "\n".join(lines) if lines else "No macro releases in the lookback window."
