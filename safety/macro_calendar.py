"""Curated, hardcoded 2026 high-impact macro event calendar.

Deliberately static rather than fetched from a live API: FOMC decision
dates are published by the Fed roughly a year in advance and CPI/NFP
release dates follow BLS's fixed annual schedule, so hardcoding verified
dates is *more* deterministic than depending on a feed -- and there is no
reliable free API for FOMC dates specifically.

Sources (verified 2026-08-17):
- FOMC: federalreserve.gov/monetarypolicy/fomccalendars.htm
- CPI:  bls.gov/schedule/ (via usinflationcalculator.com schedule page)
- NFP:  bls.gov/schedule/ (Employment Situation)

MAINTENANCE: this file must be refreshed with next year's schedule before
January -- there is no automated update. Re-verify against the sources
above before extending past 2026 or trusting a date near a schedule
revision (BLS has rescheduled releases around government shutdowns before).
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class MacroEvent:
    label: str
    event_type: str  # "fomc_decision" | "cpi" | "nfp"
    timestamp_utc: datetime


def _et_to_utc(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=_ET).astimezone(timezone.utc)


# FOMC statement release: 2:00 PM ET on the second day of each meeting.
_FOMC_2026_DATES = [(1, 28), (3, 18), (4, 29), (6, 17), (7, 29), (9, 16), (10, 28), (12, 9)]

# CPI release: 8:30 AM ET.
_CPI_2026_DATES = [
    (1, 13), (2, 13), (3, 11), (4, 10), (5, 12), (6, 10),
    (7, 14), (8, 12), (9, 11), (10, 14), (11, 10), (12, 10),
]

# Employment Situation (NFP + unemployment rate) release: 8:30 AM ET.
_NFP_2026_DATES = [
    (1, 9), (2, 11), (3, 6), (4, 3), (5, 8), (6, 5),
    (7, 2), (8, 7), (9, 4), (10, 2), (11, 6), (12, 4),
]

MACRO_CALENDAR_2026: list[MacroEvent] = (
    [MacroEvent("FOMC rate decision", "fomc_decision", _et_to_utc(2026, m, d, 14, 0)) for m, d in _FOMC_2026_DATES]
    + [MacroEvent("CPI release", "cpi", _et_to_utc(2026, m, d, 8, 30)) for m, d in _CPI_2026_DATES]
    + [MacroEvent("Employment Situation (NFP)", "nfp", _et_to_utc(2026, m, d, 8, 30)) for m, d in _NFP_2026_DATES]
)
