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


def release_dates(name: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Every distinct real-world date this series actually published a
    NEW number, within [start, end] -- these are the real, dated events
    the replay reacts to, not an approximation."""
    df = _load_vintage(name)
    dates = df["realtime_start"][(df["realtime_start"] >= start) & (df["realtime_start"] <= end)]
    return pd.DatetimeIndex(sorted(dates.unique()))


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
