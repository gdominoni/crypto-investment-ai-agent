"""Sanity tests for modules/module_b_trend_following/coarse_grid.py's grid
definitions -- catches typos/gaps in the hand-curated grids (a missing
timeframe entry, an inconsistent parameter key within a family) before
they'd otherwise only surface as a silent KeyError deep into a ~20-minute
sweep.
"""

from modules.module_b_trend_following.coarse_grid import ENTRY_GRIDS, EXIT_GRIDS, STRATEGIES, TIMEFRAMES


def test_every_strategy_has_a_grid_for_every_timeframe():
    for strategy in STRATEGIES:
        for timeframe in TIMEFRAMES:
            assert timeframe in ENTRY_GRIDS[strategy], f"{strategy} missing {timeframe} entry grid"
            assert len(ENTRY_GRIDS[strategy][timeframe]) >= 3


def test_every_timeframe_has_an_exit_grid_with_a_null_option():
    for timeframe in TIMEFRAMES:
        assert timeframe in EXIT_GRIDS
        assert len(EXIT_GRIDS[timeframe]) >= 3
        assert any(p.take_profit is None for p in EXIT_GRIDS[timeframe]), f"{timeframe} exit grid has no null option"


def test_entry_presets_within_a_family_share_the_same_parameter_keys():
    for strategy in STRATEGIES:
        for timeframe in TIMEFRAMES:
            presets = ENTRY_GRIDS[strategy][timeframe]
            key_sets = [set(p.keys()) for p in presets]
            assert all(ks == key_sets[0] for ks in key_sets), f"{strategy}/{timeframe} presets have mismatched keys"


def test_exit_presets_scale_up_with_timeframe():
    # A sanity check on the *shape* of the tailoring, not exact values --
    # the widest (non-null) stoploss should get less tight moving from
    # 15m -> 1h -> 4h, matching each timeframe's typical volatility.
    def widest_stoploss(timeframe: str) -> float:
        return min(p.stoploss for p in EXIT_GRIDS[timeframe] if p.stoploss != -0.99)

    assert widest_stoploss("15m") > widest_stoploss("1h") > widest_stoploss("4h")


def test_exit_preset_names_are_unique_per_timeframe():
    for timeframe in TIMEFRAMES:
        names = [p.name for p in EXIT_GRIDS[timeframe]]
        assert len(names) == len(set(names))
