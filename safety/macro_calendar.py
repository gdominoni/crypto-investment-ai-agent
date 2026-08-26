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


MAX_REASONABLE_FOMC_GAP_DAYS = 70  # FOMC meets ~8x/year; longest real inter-meeting
                                    # gap (the Dec-Jan turn) is ~7 weeks -- 70 is a safe
                                    # ceiling with margin, not a precisely-derived constant.


def days_to_next_fomc(as_of: datetime, calendar: list[MacroEvent]) -> float | None:
    """For macro_feature_matrix.py's `days_to_fomc` column -- reuses this
    calendar rather than a separate hardcoded copy (added 2026-08-18).

    KNOWN GAP, not silently worked around: `calendar` (MACRO_CALENDAR_2026)
    only has 2026 dates. Extending it back to 2021-08-20 (Module C's full
    Master Feature Table range) needs real, source-verified historical
    FOMC dates (federalreserve.gov's meeting calendar archive), the same
    standard already applied to the 2026 dates above -- not fabricated
    from memory. Deferred as a data-sourcing task, distinct from writing
    this function.

    A real bug lived here until 2026-08-19, caught only by running this
    against real data (a synthetic test happened to only ever query dates
    already inside its own synthetic calendar's range, so it never
    exercised this path): naively taking "the nearest future event >=
    as_of" finds a 2026 meeting for ANY as_of before 2026, including
    2021 -- returning something like "1622 days to FOMC" for an August
    2021 timestamp, which is technically an answer but a meaningless one,
    since the calendar has no real coverage of 2021-2025 at all. Fixed by
    capping how far away a "next event" is allowed to be before it's
    treated as no-coverage: real FOMC meetings never run more than ~10
    weeks apart, so anything the search finds farther out than that is a
    sign the calendar simply has no events near `as_of`, not a real
    answer to "when's the next meeting."

    :param as_of: timestamp to count forward from (UTC).
    :param calendar: list of MacroEvent to search (e.g. MACRO_CALENDAR_2026).
    :return: days (float, fractional) to the next FOMC decision at or after
        `as_of`, or None if no such event exists within
        MAX_REASONABLE_FOMC_GAP_DAYS of `as_of`.
    """
    fomc_events = [e.timestamp_utc for e in calendar if e.event_type == "fomc_decision" and e.timestamp_utc >= as_of]
    if not fomc_events:
        return None
    days = (min(fomc_events) - as_of).total_seconds() / 86400
    if days > MAX_REASONABLE_FOMC_GAP_DAYS:
        return None
    return days
