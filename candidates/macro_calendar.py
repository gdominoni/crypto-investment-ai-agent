"""FOMC + CPI release calendar. FOMC decision dates are real (Federal
Reserve press-release schedule); CPI dates are approximated as the 13th
of each month (BLS publishes CPI on a weekday in the 10th-15th range,
not scraped here). Everything here works at calendar-day granularity,
matching the daily-bar timeframe this project trades on (`is_macro_day`
in `definitions.py` only needs to know WHICH day, not what hour) -- no
intraday/timezone precision is computed because nothing downstream
consumes it yet; that would only become load-bearing if a candidate
started trading on an intraday timeframe.
"""
from __future__ import annotations

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


def fomc_days() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in FOMC_DECISION_DATES))


def cpi_days(start: str = "2018-01-01", end: str = "2026-12-31", release_day: int = 13) -> pd.DatetimeIndex:
    months = pd.date_range(start, end, freq="MS")
    days = []
    for m in months:
        d = m.replace(day=release_day)
        while d.weekday() >= 5:
            d += pd.Timedelta(days=1)
        days.append(d)
    return pd.DatetimeIndex(sorted(days))


def macro_release_days() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(set(fomc_days()) | set(cpi_days())))
