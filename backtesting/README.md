# Backtesting (Local-Only)

Heavy compute: historical data downloads, IS/OOS backtesting sweeps, hyperparameter optimization. Runs on the local PC, never on the 24/7 VPS — this is the "local offloading" half of the cost strategy.

Rules enforced here:
- The most recent 12 months of data are always held out as out-of-sample (OOS) validation, excluded from optimization.
- A dynamically computed minimum trade-count threshold filters out statistically insignificant candidates before ranking.
- Surviving candidates are ranked: Win Rate (desc) → Sortino Ratio (desc) → Net Profit after fees (desc).

Status: not yet built (Phase 4 onward).
