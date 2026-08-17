"""Module B, Phase 9 redesign: multi-factor confluence, combining all
three strategy families on every candle instead of testing them in
isolation (see modules/module_b_trend_following/README.md for why the
isolated-family approach was superseded, not just extended).

NOT YET RUN OR VERIFIED as of writing -- this is the specification for
review (multi_factor_grid.py + this file), per the explicit instruction
to hold off on any backtest until the grid is approved. No sanity-check
backtest has been run against this file, unlike every other strategy
file in this project up to this point -- that verification step still
needs to happen once the design itself is approved.

Design: asymmetric roles, not a flat AND of all three families' full
original conditions (which would be so restrictive it would rarely fire).
- Family 2 (volatility breakout) is the TIMING TRIGGER: a squeeze release
  is a naturally rare, punctual event, well suited to being the thing
  that actually opens a trade.
- Family 1 (RSI-based) is a CONFIRMING FILTER: RSI must not already be
  overbought at the moment of the trigger, guarding against chasing an
  already-extended move.
- Family 3 (volume) is a CONFIRMING FILTER: real volume + positive money
  flow must be present at the trigger, guarding against a low-volume
  fakeout/chop breakout -- directly targeting "breakouts buy into chop."

Family 1 has two variants (selected per preset via "variant"):
- "classic": RSI < a fixed overbought threshold.
- "rsi_bb": Bollinger Bands applied to the RSI series itself (not price)
  -- RSI < its own upper RSI-BB band. Adaptive to how volatile RSI has
  recently been, rather than a fixed level regardless of regime.

Exit is intentionally NOT a mirror of the entry's strict confluence --
protecting a position should be faster/looser than the bar for opening
one. Exits on ANY of: breakout momentum turning negative, price closing
back below VWAP, or the RSI-based signal (matching whichever Family 1
variant is active) crossing back into overbought territory. This is
deliberately weighted toward indicator/trend-reversal signals rather
than relying on SL/TP, per Phase 1's finding that the pure indicator-
driven ("null" SL/TP) exit outperformed every fixed-percentage exit
preset on average.
"""

import pandas_ta as pta
from pandas import DataFrame

from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter, IStrategy

from dynamic_exit_mixin import NULL_ROI, NULL_STOPLOSS


class MultiFactorConfluence(IStrategy):
    INTERFACE_VERSION = 3
    can_short = False

    minimal_roi = NULL_ROI
    stoploss = NULL_STOPLOSS
    trailing_stop = False

    startup_candle_count = 200

    # Family 1 (filter)
    f1_variant = CategoricalParameter(["classic", "rsi_bb"], default="classic", space="buy", optimize=True)
    rsi_period = IntParameter(7, 21, default=14, space="buy", optimize=True)
    rsi_overbought = IntParameter(60, 85, default=70, space="buy", optimize=True)
    rsi_bb_period = IntParameter(10, 30, default=20, space="buy", optimize=True)
    rsi_bb_std = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space="buy", optimize=True)

    # Family 2 (trigger)
    bb_period = IntParameter(10, 30, default=20, space="buy", optimize=True)
    bb_std = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space="buy", optimize=True)
    kc_period = IntParameter(10, 30, default=20, space="buy", optimize=True)
    kc_atr_mult = DecimalParameter(1.0, 3.0, default=1.5, decimals=1, space="buy", optimize=True)
    momentum_period = IntParameter(10, 40, default=20, space="buy", optimize=True)

    # Family 3 (filter)
    vwap_period = IntParameter(20, 200, default=24, space="buy", optimize=True)
    volume_surge_period = IntParameter(10, 30, default=20, space="buy", optimize=True)
    volume_surge_multiple = DecimalParameter(1.2, 3.0, default=1.5, decimals=1, space="buy", optimize=True)
    cmf_period = IntParameter(10, 30, default=20, space="buy", optimize=True)
    cmf_threshold = DecimalParameter(0.0, 0.2, default=0.05, decimals=2, space="buy", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Family 1: RSI + RSI's own Bollinger Bands.
        dataframe["rsi"] = pta.rsi(dataframe["close"], length=self.rsi_period.value)
        rsi_bb = pta.bbands(dataframe["rsi"], length=self.rsi_bb_period.value, std=self.rsi_bb_std.value)
        dataframe["rsi_bb_upper"] = rsi_bb.iloc[:, 2]

        # Family 2: Keltner Channels + TTM Squeeze momentum.
        mom_smooth = max(3, self.momentum_period.value // 3)
        sq = pta.squeeze(
            dataframe["high"], dataframe["low"], dataframe["close"],
            bb_length=self.bb_period.value, bb_std=self.bb_std.value,
            kc_length=self.kc_period.value, kc_scalar=self.kc_atr_mult.value,
            mom_length=self.momentum_period.value, mom_smooth=mom_smooth,
        )
        dataframe["sqz_momentum"] = sq.iloc[:, 0]
        dataframe["sqz_off"] = sq["SQZ_OFF"]

        # Family 3: rolling VWAP + volume surge + CMF.
        price_volume = dataframe["close"] * dataframe["volume"]
        dataframe["vwap"] = price_volume.rolling(self.vwap_period.value).sum() / dataframe["volume"].rolling(
            self.vwap_period.value
        ).sum()
        avg_volume = dataframe["volume"].rolling(self.volume_surge_period.value).mean()
        dataframe["volume_surge"] = dataframe["volume"] > (avg_volume * self.volume_surge_multiple.value)
        dataframe["cmf"] = pta.cmf(
            dataframe["high"], dataframe["low"], dataframe["close"], dataframe["volume"],
            length=self.cmf_period.value,
        )
        return dataframe

    def _family1_not_overbought(self, dataframe: DataFrame):
        if self.f1_variant.value == "classic":
            return dataframe["rsi"] < self.rsi_overbought.value
        return dataframe["rsi"] < dataframe["rsi_bb_upper"]

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        trigger = (dataframe["sqz_off"] == 1) & (dataframe["sqz_off"].shift(1) == 0) & (dataframe["sqz_momentum"] > 0)
        family1_filter = self._family1_not_overbought(dataframe)
        family3_filter = (
            (dataframe["close"] > dataframe["vwap"])
            & dataframe["volume_surge"]
            & (dataframe["cmf"] > self.cmf_threshold.value)
        )
        dataframe.loc[
            (trigger & family1_filter & family3_filter & (dataframe["volume"] > 0)),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        momentum_fading = dataframe["sqz_momentum"] < 0
        trend_reversed = dataframe["close"] < dataframe["vwap"]
        rsi_exhausted = ~self._family1_not_overbought(dataframe)
        dataframe.loc[
            ((momentum_fading | trend_reversed | rsi_exhausted) & (dataframe["volume"] > 0)),
            "exit_long",
        ] = 1
        return dataframe
