"""Gathers the full current system status -- execution mode, capital
allocation, and circuit-breaker state -- as one call. This is the single
source of truth used by both the CLI (run_allocation.py) and the Telegram
bot's /status command, so the two never drift into showing different
numbers for the same underlying state.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from modules.module_a_cash_carry.funding_analysis import analyze_funding_yield
from modules.module_b_trend_following.candidate_ranking import load_backtest_result
from orchestrator.capital_allocator import AllocationDecision, ModuleKPI, allocate_capital
from orchestrator.kpi_adapters import from_backtest_result, from_funding_yield_report
from safety.circuit_breaker import CircuitBreakerDecision, evaluate_circuit_breaker
from safety.execution_mode import get_mode
from safety.macro_calendar import MACRO_CALENDAR_2026

REPO_ROOT = Path(__file__).resolve().parent.parent
BTC_1H_PATH = REPO_ROOT / "data" / "market" / "binance" / "spot" / "BTCUSDT_1h.parquet"


@dataclass(frozen=True)
class SystemStatus:
    execution_mode: str
    generated_at: datetime
    module_kpis: list[ModuleKPI]
    allocation: AllocationDecision
    circuit_breaker: CircuitBreakerDecision | None  # None if no local OHLC data available to check


def _latest_backtest_zip(module_dir: str) -> Path | None:
    results_dir = REPO_ROOT / "modules" / module_dir / "user_data" / "backtest_results"
    if not results_dir.exists():
        return None
    zips = sorted(results_dir.glob("backtest-result-*.zip"))
    return zips[-1] if zips else None


def _module_kpis() -> list[ModuleKPI]:
    funding_reports = [analyze_funding_yield(s) for s in ["BTCUSDT", "ETHUSDT"]]
    best_funding = max(funding_reports, key=lambda r: r.annualized_yield_net_of_fees_pct)
    kpis = [from_funding_yield_report(best_funding, module_name="module_a_cash_carry")]

    b_zip = _latest_backtest_zip("module_b_trend_following")
    if b_zip:
        kpis.append(
            from_backtest_result(load_backtest_result(b_zip, "TrendEmaAdx"), module_name="module_b_trend_following")
        )

    c_zip = _latest_backtest_zip("module_c_volatility_ml")
    if c_zip:
        kpis.append(
            from_backtest_result(
                load_backtest_result(c_zip, "VolatilityGateSignal"), module_name="module_c_volatility_ml"
            )
        )

    return kpis


def _circuit_breaker_check() -> CircuitBreakerDecision | None:
    if not BTC_1H_PATH.exists():
        return None
    ohlc = pd.read_parquet(BTC_1H_PATH)
    return evaluate_circuit_breaker(datetime.now(timezone.utc), ohlc, MACRO_CALENDAR_2026)


def gather_system_status() -> SystemStatus:
    kpis = _module_kpis()
    return SystemStatus(
        execution_mode=get_mode(),
        generated_at=datetime.now(timezone.utc),
        module_kpis=kpis,
        allocation=allocate_capital(kpis),
        circuit_breaker=_circuit_breaker_check(),
    )
