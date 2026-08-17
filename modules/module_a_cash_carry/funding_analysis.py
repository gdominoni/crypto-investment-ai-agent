"""Analyzes historical Binance funding-rate data to evaluate whether Module
A's delta-neutral cash & carry strategy (long spot + short perpetual) has
been -- and currently is -- worth running for a given pair.

This is Module A's equivalent of Module B's Freqtrade backtesting: heavy,
local, one-off analysis over historical data that informs whether/how a
live strategy should be deployed. Hummingbot itself has no historical
backtesting mode for this strategy type, so this analysis has to happen
outside it, using the funding-rate history already pulled in Phase 2
(data_ingestion/market_data/binance_fetcher.py).

Simplification, stated explicitly: fee cost is amortized as a single
open+close round trip over the whole annualized yield, i.e. this models
"open once, hold for a year, close once" -- not the more active
open/close behavior the live Hummingbot strategy may actually exhibit
(it trades on basis convergence, not funding directly; see README.md).
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

FUNDING_DIR = Path(__file__).resolve().parents[2] / "data" / "market" / "binance" / "funding"

FUNDING_PERIODS_PER_YEAR = 365 * 3  # Binance perpetuals settle funding every 8 hours

# Estimated cost of opening AND closing a delta-neutral position once
# (one spot leg + one futures leg, each way) at standard (non-VIP) taker
# fees. Deliberately conservative/worst-case -- maker fees would be lower.
_SPOT_TAKER_FEE_PCT = 0.0010  # 0.10% per spot leg
_FUTURES_TAKER_FEE_PCT = 0.0004  # 0.04% per futures leg
ROUND_TRIP_COST_PCT = 2 * (_SPOT_TAKER_FEE_PCT + _FUTURES_TAKER_FEE_PCT)

# Safety margin applied to the round-trip cost when suggesting a Hummingbot
# min_opening_arbitrage_pct -- a simple, transparent heuristic, not a
# fitted model of the basis/funding relationship (see README.md).
_OPENING_THRESHOLD_SAFETY_MULTIPLIER = 1.5


@dataclass(frozen=True)
class FundingYieldReport:
    symbol: str
    periods: int
    mean_funding_rate: float  # per 8h period
    positive_funding_pct: float  # fraction of periods with rate > 0
    annualized_yield_gross_pct: float
    annualized_yield_net_of_fees_pct: float
    sortino_ratio: float  # annualized, computed on the raw per-period funding rate series
    is_currently_attractive: bool  # last N days' annualized yield still clears the cost
    suggested_min_opening_arbitrage_pct: float


def _annualized_sortino_ratio(returns: pd.Series, periods_per_year: int) -> float:
    """Sortino on the funding-rate series itself, treating each 8h funding
    payment as a "return". Used so Module A can be ranked on the exact same
    Win Rate -> Sortino -> Net Profit hierarchy as Modules B and C
    (orchestrator/capital_allocator.py), despite being a fundamentally
    different (yield-harvesting, not directional) strategy.
    """
    mean_return = returns.mean()
    downside = returns.clip(upper=0)
    downside_deviation = (downside**2).mean() ** 0.5
    if downside_deviation == 0:
        return 0.0
    period_sortino = mean_return / downside_deviation
    return float(period_sortino * (periods_per_year**0.5))


def load_funding_history(symbol: str) -> pd.DataFrame:
    path = FUNDING_DIR / f"{symbol}_funding.csv"
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def analyze_funding_yield(symbol: str, recent_window_days: int = 30) -> FundingYieldReport:
    df = load_funding_history(symbol)
    return _compute_report(df, symbol, recent_window_days)


def _compute_report(df: pd.DataFrame, symbol: str, recent_window_days: int) -> FundingYieldReport:
    mean_rate = df["fundingRate"].mean()
    positive_pct = (df["fundingRate"] > 0).mean()
    annualized_gross = mean_rate * FUNDING_PERIODS_PER_YEAR
    annualized_net = annualized_gross - ROUND_TRIP_COST_PCT

    recent_periods = recent_window_days * 3
    recent = df.tail(recent_periods)
    recent_annualized_gross = recent["fundingRate"].mean() * FUNDING_PERIODS_PER_YEAR if len(recent) else 0.0
    is_attractive = bool(recent_annualized_gross - ROUND_TRIP_COST_PCT > 0)

    sortino = _annualized_sortino_ratio(df["fundingRate"], FUNDING_PERIODS_PER_YEAR)

    return FundingYieldReport(
        symbol=symbol,
        periods=len(df),
        mean_funding_rate=mean_rate,
        positive_funding_pct=positive_pct,
        annualized_yield_gross_pct=annualized_gross * 100,
        annualized_yield_net_of_fees_pct=annualized_net * 100,
        sortino_ratio=sortino,
        is_currently_attractive=is_attractive,
        suggested_min_opening_arbitrage_pct=ROUND_TRIP_COST_PCT * _OPENING_THRESHOLD_SAFETY_MULTIPLIER * 100,
    )


if __name__ == "__main__":
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        report = analyze_funding_yield(symbol)
        print(f"{report.symbol}: {report.periods} funding periods")
        print(f"  positive funding: {report.positive_funding_pct:.1%}")
        print(f"  annualized yield (gross): {report.annualized_yield_gross_pct:.2f}%")
        print(f"  annualized yield (net of round-trip fees): {report.annualized_yield_net_of_fees_pct:.2f}%")
        print(f"  sortino ratio: {report.sortino_ratio:.2f}")
        print(f"  currently attractive (last 30d): {report.is_currently_attractive}")
        print(f"  suggested min_opening_arbitrage_pct: {report.suggested_min_opening_arbitrage_pct:.3f}")
