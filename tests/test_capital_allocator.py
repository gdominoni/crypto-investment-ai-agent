"""Tests for orchestrator/capital_allocator.py."""

from orchestrator.capital_allocator import ModuleKPI, allocate_capital
from safety.limits import MAX_MODULE_ALLOCATION_PCT


def _kpi(name, win_rate=0.6, sortino=1.0, net_profit=5.0, sample_size=200, significant=True):
    return ModuleKPI(
        module_name=name,
        win_rate=win_rate,
        sortino_ratio=sortino,
        net_profit_pct=net_profit,
        sample_size=sample_size,
        meets_significance_threshold=significant,
    )


def test_all_ineligible_reserves_all_cash():
    decision = allocate_capital([_kpi("a", significant=False)])
    assert decision.weights == {}
    assert decision.cash_reserve_pct == 1.0
    assert decision.excluded_modules["a"] == "insufficient sample size"


def test_excludes_negative_net_profit():
    decision = allocate_capital([_kpi("a", net_profit=-1.0)])
    assert "a" not in decision.weights
    assert decision.excluded_modules["a"] == "non-positive net profit"


def test_excludes_negative_sortino():
    decision = allocate_capital([_kpi("a", sortino=-0.5)])
    assert "a" not in decision.weights
    assert decision.excluded_modules["a"] == "non-positive Sortino ratio"


def test_single_eligible_module_is_capped_not_given_all_deployable_capital():
    # With only one eligible module, its raw rank-based share would be the
    # full deployable pool (1.0 - MIN_CASH_RESERVE_PCT = 0.9) -- but the
    # per-module cap (0.6) binds first, so it should be clipped there, with
    # the rest returned to cash rather than over-concentrated in one module.
    decision = allocate_capital([_kpi("a")])
    assert decision.weights["a"] == MAX_MODULE_ALLOCATION_PCT
    assert decision.cash_reserve_pct == round(1.0 - MAX_MODULE_ALLOCATION_PCT, 10)


def test_higher_ranked_module_gets_more_weight():
    # "a" wins on win_rate (primary key), so should out-weight "b" even
    # though "b" has a higher Sortino -- ranking order drives weight, not
    # any single metric in isolation.
    better = _kpi("a", win_rate=0.8, sortino=0.5)
    worse = _kpi("b", win_rate=0.5, sortino=5.0)
    decision = allocate_capital([worse, better])
    assert decision.weights["a"] > decision.weights["b"]


def test_no_module_exceeds_max_allocation_cap_across_various_distributions():
    for kpis in (
        [_kpi("a", win_rate=0.9), _kpi("b", win_rate=0.1)],
        [_kpi("a", win_rate=0.9), _kpi("b", win_rate=0.6), _kpi("c", win_rate=0.3)],
    ):
        decision = allocate_capital(kpis)
        assert all(w <= MAX_MODULE_ALLOCATION_PCT + 1e-9 for w in decision.weights.values())


def test_weights_never_exceed_total_capital():
    kpis = [_kpi("a", win_rate=0.9), _kpi("b", win_rate=0.6), _kpi("c", win_rate=0.3)]
    decision = allocate_capital(kpis)
    assert sum(decision.weights.values()) + decision.cash_reserve_pct == 1.0
