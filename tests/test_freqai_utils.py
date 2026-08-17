"""Tests for modules/module_c_volatility_ml/user_data/strategies/freqai_utils.py.

Loaded directly by file path rather than as a package import: it lives
inside a Freqtrade user_data/strategies/ tree (so Freqtrade can find it as
a sibling import at runtime), not under modules/module_c_volatility_ml/
as a normal Python package, and it deliberately has zero freqtrade
dependency so it can be tested without Docker.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "modules"
    / "module_c_volatility_ml"
    / "user_data"
    / "strategies"
    / "freqai_utils.py"
)
_spec = importlib.util.spec_from_file_location("freqai_utils", _MODULE_PATH)
freqai_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(freqai_utils)


def test_flatten_yfinance_columns_handles_multiindex():
    df = pd.DataFrame({("Close", "^VIX"): [1.0], ("High", "^VIX"): [2.0]})
    flat = freqai_utils.flatten_yfinance_columns(df)
    assert list(flat.columns) == ["Close", "High"]


def test_flatten_yfinance_columns_is_noop_for_plain_columns():
    df = pd.DataFrame({"Close": [1.0], "High": [2.0]})
    flat = freqai_utils.flatten_yfinance_columns(df)
    assert list(flat.columns) == ["Close", "High"]


@pytest.fixture
def synthetic_vix_dir(tmp_path):
    macro_dir = tmp_path / "yfinance"
    macro_dir.mkdir()
    df = pd.DataFrame(
        {
            ("Close", "^VIX"): [15.0, 16.5, 14.2],
            ("High", "^VIX"): [15.5, 17.0, 14.8],
        },
        index=pd.DatetimeIndex(["2026-01-01", "2026-01-02", "2026-01-03"], name="Date"),
    )
    df.to_parquet(macro_dir / "vix.parquet")
    return tmp_path


def test_load_vix_returns_date_and_close_only(synthetic_vix_dir):
    result = freqai_utils.load_vix(synthetic_vix_dir)
    assert list(result.columns) == ["date", "vix_close"]
    assert len(result) == 3
    assert result["vix_close"].tolist() == [15.0, 16.5, 14.2]


def test_load_vix_is_sorted_by_date(synthetic_vix_dir):
    result = freqai_utils.load_vix(synthetic_vix_dir)
    assert result["date"].is_monotonic_increasing
