# Phase 7: Dynamic Capital Allocator

**Goal:** build the piece that reads all three modules' real performance together and decides how to split capital across them — the part of the spec that says the initial split "is NOT fixed" and must be dynamically derived from each module's own results.

## The prompt

"yes" — a plain continuation after Phase 6. As with the last several phases, the specific design decisions below come from actually building against the project's real, accumulated results (Module A's funding analysis, Module B and C's real backtest zips still on disk) rather than new spec text.

## Decisions made in this phase

1. **A shared `ModuleKPI` shape, with per-module adapters, not a shared base class.** Module A's economics (funding yield) and Module B/C's economics (backtest P&L) are different enough in kind that forcing them into a shared class hierarchy would be the wrong abstraction. Instead, each module keeps its own real result type (`FundingYieldReport`, `BacktestResult`), and `orchestrator/kpi_adapters.py` maps each onto one common `ModuleKPI` shape just for the allocator's purposes -- the mapping is where the interesting judgment calls live (see below), not a type-coercion exercise.
2. **Module A's "win rate" is its positive-funding-period percentage.** Not a metaphor -- structurally, the fraction of 8-hour funding periods where the rate was positive is exactly analogous to a trading strategy's win rate (fraction of profitable trades), which is what let Module A be ranked on the *exact same* hierarchy as B and C rather than needing a separate allocation mechanism.
3. **Added a real Sortino calculation to Module A's funding analysis**, computed on the funding-rate series itself (mean/downside-deviation, annualized) -- both to let Module A be ranked on the same hierarchy, and because this surfaced the scale-mismatch problem described in the decisions log before it could become a bug in the allocator's weighting logic.
4. **Weighting is rank-based, not proportional to raw Sortino.** Full reasoning in [decisions-log.md](decisions-log.md) -- the short version: Module A's Sortino (103-136, from tiny fixed-yield-style funding "returns") and Module B/C's Sortino (-2 to +2, from trade-level P&L) aren't on comparable scales, so weighting by raw magnitude would let a units artifact dominate the decision. Rank position only compares "who's better, by the hierarchy," never "how many times better."
5. **Two new hardcoded safety constants**, added to `safety/limits.py` alongside the Phase 3 constants rather than living in the orchestrator: `MAX_MODULE_ALLOCATION_PCT` (0.60 -- no module ever gets more, however good it looks) and `MIN_CASH_RESERVE_PCT` (0.10 -- never fully deployed). Keeping these in the safety kernel, not `capital_allocator.py`, matches the established pattern of centralizing every hardcoded number in one place the LLM/orchestrator can read but never edit.
6. **Capital freed by the per-module cap returns to cash, not to other eligible modules.** Redistributing it would mean a capped module's "excess" silently flows elsewhere without re-running eligibility on the recipient -- a correctness trap judged not worth an extra layer of rebalancing logic to avoid.

## What got built and verified (against real, current data from all three modules)

- `orchestrator/capital_allocator.py` — `ModuleKPI`, `allocate_capital()`, 7 unit tests (including one that specifically caught a bug in a first draft of the tests themselves, not the allocator -- see below).
- `orchestrator/kpi_adapters.py` — `from_funding_yield_report()`, `from_backtest_result()` (the latter works for both Module B and Module C's results, since both produce ordinary Freqtrade backtest output), 4 unit tests.
- `orchestrator/run_allocation.py` — pulls Module A's live funding analysis and the most recent real backtest result on disk for B and C, runs the actual allocation.

**Real result:** Module A (ETH/USDT) — 84.6% win rate, Sortino 103, net profit 7.18%, 3,285 funding periods — clears every eligibility bar and gets capped at the maximum allowed 60%. Module B (40 trades) and Module C (2 trades) are both excluded for insufficient sample size, consistent with what Phases 4 and 6 already found honestly. Cash reserve: 40%. Full table in [orchestrator/README.md](../../orchestrator/README.md#real-result-2026-08-17).

## A test bug caught before it shipped

The first draft of `test_single_eligible_module_gets_full_deployable_share` asserted a lone eligible module would receive the full deployable pool (90%). Running it failed immediately: the allocator correctly capped it at 60% (`MAX_MODULE_ALLOCATION_PCT`), and the *test's* expectation was wrong, not the code. Fixed by recalculating the expected value against the actual constants rather than adjusting the allocator to match a test written from an assumption. Renamed to `test_single_eligible_module_is_capped_not_given_all_deployable_capital` to describe the behavior actually being verified.

## Still pending

- The allocator produces a decision; nothing yet acts on it. Wiring rebalancing into an actual live/dry-run loop, and reading the Module C volatility signal as a risk *modifier* on top of this allocation (rather than treating it purely as a fourth ranked strategy) are Phase 8 questions.
- No macro-regime input yet, despite the original spec mentioning "dynamically analyze market regimes" -- the allocator currently only looks at each module's own backtest/yield performance, not broader market conditions. A natural Phase 8 extension once there's a concrete regime signal to feed in (Module C's volatility prediction is the obvious candidate).
