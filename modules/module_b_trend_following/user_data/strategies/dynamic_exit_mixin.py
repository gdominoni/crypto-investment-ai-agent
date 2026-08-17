"""Shared dynamic-exit constants for Module B's Phase 9 coarse-grid
strategy families (mean reversion, volatility breakout, volume-driven).

Freqtrade requires `stoploss` to be a real negative float and
`minimal_roi` to be a real dict -- there's no native "disabled" state.
The coarse grid spec requires a "null" SL/TP option, to test exits driven
purely by signal/indicator reversal, so "null" is emulated as a value so
permissive it never triggers in practice: NULL_STOPLOSS (-99%) and
NULL_ROI (10,000% required profit).

All three family strategies import these same two constants rather than
each approximating "disabled" slightly differently -- the DRY point isn't
much code, it's that "null" means the exact same thing everywhere it's
used, so the coarse-grid results are comparable across families.
"""

NULL_STOPLOSS = -0.99
NULL_ROI = {"0": 100.0}
