"""Dynamic capital allocation across Modules A, B, and C.

Every module is reduced to the same shared `ModuleKPI` shape (see
kpi_adapters.py for how each module's own result type maps onto it), then
ranked with the project's fixed hierarchy: Win Rate (desc) -> Sortino
Ratio (desc) -> Net Profit (desc) -- the same hierarchy used for backtest
candidate selection (Module B's candidate_ranking.py) and specified for
live/dry-run monitoring.

Weighting is rank-based, not proportional to raw Sortino magnitude --
deliberately. Module A's Sortino is computed on funding-rate "returns"
(tiny, low-variance, fixed-yield-style numbers), which annualizes to
Sortino values in the hundreds -- structurally incomparable to Module B/C's
trade-level Sortino (typically -2 to +2). Weighting directly by that raw
number would let a unit-scale artifact dominate the allocation, not
genuine risk-adjusted merit. Rank position sidesteps the comparability
problem entirely: it only asks "who's better, by the spec's own ordering,"
never "how many times better," which the raw numbers can't safely answer
across such different strategy types.
"""

from dataclasses import dataclass

from safety.limits import MAX_MODULE_ALLOCATION_PCT, MIN_CASH_RESERVE_PCT


@dataclass(frozen=True)
class ModuleKPI:
    module_name: str
    win_rate: float  # 0-1
    sortino_ratio: float
    net_profit_pct: float  # % of allocated capital, after fees
    sample_size: int
    meets_significance_threshold: bool


@dataclass(frozen=True)
class AllocationDecision:
    weights: dict[str, float]  # module_name -> fraction of total capital; sums to <= 1.0
    cash_reserve_pct: float
    excluded_modules: dict[str, str]  # module_name -> human-readable exclusion reason


def _exclusion_reason(kpi: ModuleKPI) -> str | None:
    if not kpi.meets_significance_threshold:
        return "insufficient sample size"
    if kpi.net_profit_pct <= 0:
        return "non-positive net profit"
    if kpi.sortino_ratio <= 0:
        return "non-positive Sortino ratio"
    return None


def allocate_capital(kpis: list[ModuleKPI]) -> AllocationDecision:
    ranked = sorted(kpis, key=lambda k: (k.win_rate, k.sortino_ratio, k.net_profit_pct), reverse=True)

    excluded: dict[str, str] = {}
    eligible: list[ModuleKPI] = []
    for kpi in ranked:
        reason = _exclusion_reason(kpi)
        if reason is None:
            eligible.append(kpi)
        else:
            excluded[kpi.module_name] = reason

    if not eligible:
        return AllocationDecision(weights={}, cash_reserve_pct=1.0, excluded_modules=excluded)

    # Rank-based weight: 1st of N eligible modules scores N, last scores 1.
    n = len(eligible)
    rank_scores = [n - i for i in range(n)]
    total_score = sum(rank_scores)
    deployable_pct = 1.0 - MIN_CASH_RESERVE_PCT

    weights: dict[str, float] = {}
    for kpi, score in zip(eligible, rank_scores):
        raw_weight = (score / total_score) * deployable_pct
        weights[kpi.module_name] = min(raw_weight, MAX_MODULE_ALLOCATION_PCT)

    # Capital freed up by the per-module cap goes back to cash, not
    # redistributed to other modules -- redistributing would let a capped
    # module's excess silently flow elsewhere without re-running eligibility,
    # which is a correctness trap not worth an extra layer of logic to avoid.
    cash_reserve_pct = 1.0 - sum(weights.values())

    return AllocationDecision(weights=weights, cash_reserve_pct=cash_reserve_pct, excluded_modules=excluded)
