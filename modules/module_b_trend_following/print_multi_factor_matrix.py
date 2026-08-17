"""Prints (does NOT run) the full multi-factor confluence combination
matrix for review, per the human director's explicit instruction not to
execute any backtests until the grid itself is approved.

Run locally with:
    python -m modules.module_b_trend_following.print_multi_factor_matrix
"""

from modules.module_b_trend_following.multi_factor_grid import (
    EXIT_GRIDS,
    FAMILY1_GRIDS,
    FAMILY2_GRIDS,
    FAMILY3_GRIDS,
    TIMEFRAMES,
)


def main() -> None:
    total = 0
    for timeframe in TIMEFRAMES:
        f1 = FAMILY1_GRIDS[timeframe]
        f2 = FAMILY2_GRIDS[timeframe]
        f3 = FAMILY3_GRIDS[timeframe]
        exits = EXIT_GRIDS[timeframe]
        combo_count = len(f1) * len(f2) * len(f3) * len(exits)
        total += combo_count * 2  # x2 for IS + OOS

        print(f"\n{'='*80}\n{timeframe}  ({len(f1)} x {len(f2)} x {len(f3)} entry combos x {len(exits)} exits = {combo_count} configs, x2 IS/OOS = {combo_count*2} backtests)\n{'='*80}")

        idx = 0
        for f1_preset in f1:
            for f2_preset in f2:
                for f3_preset in f3:
                    for exit_preset in exits:
                        idx += 1
                        print(f"\n[{timeframe} #{idx}]")
                        print(f"  Family1 (filter, not-overbought): {f1_preset}")
                        print(f"  Family2 (trigger, breakout):      {f2_preset}")
                        print(f"  Family3 (filter, volume):         {f3_preset}")
                        print(f"  Exit:                             {exit_preset}")

    print(f"\n{'='*80}")
    print(f"TOTAL across all timeframes: {total} backtests (entry combos x exits x timeframes x 2 periods)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
