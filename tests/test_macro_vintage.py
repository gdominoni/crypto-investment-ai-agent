

def test_release_dates_can_exclude_revision_only_days():
    """The annual seasonal-adjustment pass re-publishes years of history at
    once, and ALFRED stamps it with a real `realtime_start`. Left in, the
    replay fires a macro-release event on 1 January -- when no US agency
    publishes anything -- and hands Sonnet the previous period's value
    unchanged. 37 of 687 firings over 2018-2026, almost all on 1 January."""
    import pandas as pd
    from candidates.macro_vintage import MACRO_SERIES, release_dates

    start, end = pd.Timestamp("2018-01-01"), pd.Timestamp("2026-08-31")
    for key in MACRO_SERIES:
        every = release_dates(key, start, end)
        new_only = release_dates(key, start, end, new_periods_only=True)
        assert set(new_only) <= set(every)
        # No US statistical release lands on New Year's Day; any such date is a
        # vintage-bookkeeping artifact, which is exactly what this filters.
        assert not [d for d in new_only if (d.month, d.day) == (1, 1)]
    assert len(release_dates("cpi", start, end, new_periods_only=True)) < \
           len(release_dates("cpi", start, end))
