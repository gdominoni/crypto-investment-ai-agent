# Orchestrator

Cross-module coordination layer: the dynamic capital allocator (built, Phase 7), and — still to come — the Telegram bot (Haiku-powered NLP/status formatting) and the live/dry-run switch (Phase 8). The system always boots in dry-run, and only an explicit human-issued Telegram confirmation command will be able to move any module to live execution.

## Dynamic capital allocator

`capital_allocator.py` reduces every module to one shared shape, `ModuleKPI` (win rate, Sortino ratio, net profit %, sample size, whether it clears its own significance threshold), then applies the project's fixed ranking hierarchy — **Win Rate → Sortino Ratio → Net Profit**, the same hierarchy used for Module B's backtest candidate selection — to decide both *whether* a module gets capital and *how much*.

**Eligibility first, weight second.** A module gets 0% if it fails any of: insufficient sample size, non-positive net profit, or non-positive Sortino. Only eligible modules are weighted at all — this means a module can look statistically "good" on paper and still receive nothing, if the sample behind that number isn't yet trustworthy.

**Weighting is rank-based, not proportional to raw Sortino — deliberately.** Module A's Sortino is computed on funding-rate "returns" (tiny, low-variance, fixed-yield-style numbers) and annualizes to values in the hundreds — structurally incomparable to Module B/C's trade-level Sortino (typically -2 to +2). Weighting directly by that raw number would let a unit-scale artifact dominate the allocation, not genuine merit. Rank position sidesteps the problem: it only asks "who's better, by the spec's own ordering," never "how many times better," which the raw numbers can't safely answer across such different strategy types. This was caught by actually running the allocator against Module A's real numbers before finalizing the design — Module A's raw Sortino (103–136) would have swamped everything else by orders of magnitude.

**Hardcoded ceilings, from the safety kernel (`safety/limits.py`), not from this file:**
- `MAX_MODULE_ALLOCATION_PCT = 0.60` — no single module ever gets more than 60% of capital, however good it looks, preserving diversification even when one module is genuinely the best option available.
- `MIN_CASH_RESERVE_PCT = 0.10` — at least 10% always stays in USDT, never fully deployed.

Capital freed up by the per-module cap returns to cash, not to other modules — redistributing it would let a capped module's excess silently flow elsewhere without re-running eligibility, which is a correctness trap not worth an extra layer of logic to avoid.

## Real result (2026-08-17)

`python -m orchestrator.run_allocation`, against Module A's live funding analysis and the most recent real backtest result on disk for Modules B and C:

| Module | Win Rate | Sortino | Net Profit | Sample size | Outcome |
|---|---|---|---|---|---|
| A (cash & carry, ETH/USDT) | 84.6% | 103.42 | 7.18% | 3,285 funding periods | **60.0%** (capped) |
| B (trend-following) | 42.5% | -0.98 | -10.79% | 40 trades | excluded — insufficient sample size |
| C (volatility gate) | 50.0% | -100.00 | -0.06% | 2 trades | excluded — insufficient sample size |

Cash reserve: 40.0%. This is an honest, non-cherry-picked result that ties together every prior phase's real findings: Module A's genuinely attractive funding yield (Phase 5) earns it the maximum allowed allocation, while Modules B and C are correctly excluded for the exact reasons already surfaced in Phases 4 and 6 — not enough validated trades to trust yet, not a flaw in the allocator.

Status: ✅ built and verified against real data (Phase 7). Telegram bot and live/dry-run switch are Phase 8.
