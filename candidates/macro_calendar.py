"""FOMC + CPI + initial-jobless-claims release calendar.

Every date here is a REAL publication date. FOMC decision dates come
from the Federal Reserve's own press-release schedule (hardcoded below);
CPI and jobless-claims dates are read from the ALFRED vintage files this
project already downloads (`candidates/macro_vintage.py`), where each
row's `realtime_start` IS the day that figure first became public.

Why that matters, measured rather than assumed. CPI days used to be
APPROXIMATED as "the 13th of the month, rolled off weekends", on the
reasoning that BLS publishes somewhere in the 10th-15th range. Checked
against the real release dates already sitting in `data/macro/fred_vintage/cpi.csv`:

    exact match      21%
    off by 1 day     33%
    off by >= 2 days 46%   (worst case 20 days)
    mean abs error   2.16 days

CPI is 108 of this calendar's ~176 pre-fix event days, so the MAJORITY
of macro events were being studied on the wrong day. For an event study
run at 3- and 7-day horizons that is not a rounding error: it smears a
real post-release reaction across the baseline and attenuates the
measured effect toward zero, which is indistinguishable from "no pattern
exists". Every null result this project produced before this fix was
measured through that smearing.

Initial jobless claims are NEW to the calendar here. The vintages were
already downloaded and already graded into `jobless_claims_surprise`,
but `macro_release_days()` unioned only FOMC and CPI -- so the weekly
release that carries the surprise indicator was not itself a macro day.
Adding it takes the base event rate from ~19/yr to ~70/yr, the single
largest sample-size gain available to this project, and the sample size
is what its statistical power is starved of (see
docs/case_study/methodology-decisions.md).

Everything here works at calendar-day granularity, matching the daily-bar
timeframe this project trades on (`is_macro_day` in `definitions.py` only
needs to know WHICH day, not what hour) -- no intraday/timezone precision
is computed because nothing downstream consumes it yet; that would only
become load-bearing if a candidate started trading on an intraday
timeframe.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

FOMC_DECISION_DATES = [
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
]

# ALFRED reports the earliest vintage it holds as starting on the download's
# own start date, so that one date is a boundary artifact (839 CPI rows carry
# it) rather than a real release. Excluded from every release calendar below.
_VINTAGE_BOUNDARY = pd.Timestamp("2017-01-01")


def fomc_days() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in FOMC_DECISION_DATES))


@lru_cache(maxsize=8)
def _release_dates(series: str) -> pd.DatetimeIndex:
    """Real publication dates for an ALFRED series: the distinct
    `realtime_start` values in its vintage file.

    Cached because `macro_release_days()` is called once per candidate
    per battery run and the jobless-claims vintage file is ~11MB.
    Returns an EMPTY index if the vintage file is missing, so a partial
    data checkout degrades to the calendar it can actually support
    rather than crashing the whole battery."""
    from candidates.macro_vintage import VINTAGE_DIR

    path = VINTAGE_DIR / f"{series}.csv"
    if not path.exists():
        return pd.DatetimeIndex([])
    rs = pd.to_datetime(pd.read_csv(path, usecols=["realtime_start"])["realtime_start"])
    return pd.DatetimeIndex(sorted(set(rs[rs > _VINTAGE_BOUNDARY])))


def cpi_days() -> pd.DatetimeIndex:
    """Real BLS CPI publication dates (see module docstring for what the
    old 13th-of-the-month approximation cost)."""
    return _release_dates("cpi")


def jobless_claims_days() -> pd.DatetimeIndex:
    """Real weekly initial-jobless-claims publication dates."""
    return _release_dates("initial_jobless_claims")


def macro_release_days() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(set(fomc_days()) | set(cpi_days()) | set(jobless_claims_days())))
