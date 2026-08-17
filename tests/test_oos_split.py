"""Tests for modules/module_b_trend_following/oos_split.py."""

from datetime import date

import pytest

from modules.module_b_trend_following.oos_split import split_is_oos


def test_splits_correctly():
    split = split_is_oos(date(2023, 1, 1), date(2026, 1, 1), oos_months=12)
    assert split.is_start == date(2023, 1, 1)
    assert split.oos_end == date(2026, 1, 1)
    assert split.is_end == split.oos_start


def test_raises_when_range_too_short():
    with pytest.raises(ValueError):
        split_is_oos(date(2025, 6, 1), date(2026, 1, 1), oos_months=12)


def test_timerange_string_format():
    split = split_is_oos(date(2023, 1, 1), date(2026, 1, 1), oos_months=12)
    assert split.is_timerange == f"{split.is_start:%Y%m%d}-{split.is_end:%Y%m%d}"
    assert "-" in split.oos_timerange
