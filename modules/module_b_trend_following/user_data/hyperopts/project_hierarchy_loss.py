"""Custom hyperopt loss matching the project's fixed ranking hierarchy --
Win Rate (desc) -> Sortino Ratio (desc) -> Net Profit (desc), the exact
same hierarchy candidate_ranking.py applies to backtest results and the
capital allocator applies across modules. Freqtrade's built-in loss
functions (SharpeHyperOptLoss, SortinoHyperOptLoss, CalmarHyperOptLoss,
etc.) each optimize a single metric; none reflect this project's specific
lexicographic priority. Using one of them would mean Module B gets tuned
against a different standard than the one it's judged by everywhere else
in this project -- this loss function closes that gap.

Reuses freqtrade.data.metrics.calculate_sortino (the same function behind
the backtest report's own "Sortino (closed trades)" figure) rather than
recomputing Sortino a third way, so hyperopt's notion of "better" stays
consistent with what candidate_ranking.py reads from the resulting
backtest zip afterward.
"""

import math
from datetime import datetime

from pandas import DataFrame

from freqtrade.data.metrics import calculate_sortino
from freqtrade.optimize.hyperopt_loss.hyperopt_loss_interface import IHyperOptLoss

# Composite score weights: win_rate dominates by scale (matching its
# status as the primary rank key), sortino is the secondary tiebreak,
# net_profit_pct refines further -- mirrors candidate_ranking.py's sort
# key (win_rate, sortino, net_profit), just collapsed into one scalar
# hyperopt's optimizer can search over.
_WIN_RATE_SCALE = 1_000_000
_SORTINO_SCALE = 1_000


def _dynamic_min_trade_count(backtest_days: int, confidence: float = 0.95, margin_of_error: float = 0.10, min_trades_per_week: float = 1.0) -> int:
    """Duplicated from candidate_ranking.py's dynamic_min_trade_count --
    can't import it here, since this file runs inside the Freqtrade Docker
    container, which only has user_data/ mounted, not the full repo (same
    boundary as data_ingestion/macro_data/loaders.py vs freqai_utils.py).
    Keep in sync by hand if the formula in candidate_ranking.py changes.
    """
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)
    statistical_floor = math.ceil((z**2) * 0.25 / (margin_of_error**2))
    activity_floor = math.ceil((backtest_days / 7) * min_trades_per_week)
    return max(statistical_floor, activity_floor)


class ProjectHierarchyLoss(IHyperOptLoss):
    @staticmethod
    def hyperopt_loss_function(
        *,
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        starting_balance: float,
        **kwargs,
    ) -> float:
        backtest_days = (max_date - min_date).days
        min_trades = _dynamic_min_trade_count(backtest_days)
        if trade_count < min_trades:
            # Fails the project's own statistical-significance bar -- same
            # worst-case treatment as zero trades. Without this, hyperopt
            # readily finds a single lucky trade with a 100% win rate and
            # rates it above any real, larger sample with a merely-good win
            # rate, since win_rate dominates the composite score by scale.
            # Caught by actually inspecting a first hyperopt run's "best"
            # result (1 trade) rather than trusting the objective value.
            return 100.0

        win_rate = (results["profit_abs"] > 0).mean()
        sortino_ratio = calculate_sortino(results, min_date, max_date, starting_balance)
        net_profit_pct = results["profit_abs"].sum() / starting_balance * 100

        composite_score = win_rate * _WIN_RATE_SCALE + sortino_ratio * _SORTINO_SCALE + net_profit_pct
        return -composite_score  # hyperopt minimizes; higher composite = better, so negate
