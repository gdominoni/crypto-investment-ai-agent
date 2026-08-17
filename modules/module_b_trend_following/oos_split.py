"""Splits a full backtest date range into In-Sample (optimization) and
Out-of-Sample (holdout validation) periods.

Per the project's operational rules: the most recent 12 months are always
reserved as OOS and excluded from strategy optimization/selection -- they
exist only to check a selected strategy on data it never influenced.
"""

from dataclasses import dataclass
from datetime import date, timedelta

_DAYS_PER_MONTH = 30  # approximation; fine at the monthly granularity this is used at


@dataclass(frozen=True)
class TimeRangeSplit:
    is_start: date
    is_end: date
    oos_start: date
    oos_end: date

    @property
    def is_timerange(self) -> str:
        return f"{self.is_start:%Y%m%d}-{self.is_end:%Y%m%d}"

    @property
    def oos_timerange(self) -> str:
        return f"{self.oos_start:%Y%m%d}-{self.oos_end:%Y%m%d}"


def split_is_oos(start: date, end: date, oos_months: int = 12) -> TimeRangeSplit:
    oos_start = end - timedelta(days=oos_months * _DAYS_PER_MONTH)
    if oos_start <= start:
        raise ValueError(f"Data range {start}..{end} is too short to hold out {oos_months} months as OOS")
    return TimeRangeSplit(is_start=start, is_end=oos_start, oos_start=oos_start, oos_end=end)
