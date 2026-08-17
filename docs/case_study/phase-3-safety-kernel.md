# Phase 3: Safety Kernel

**Goal:** build the deterministic circuit breaker and hardcoded risk limits *before* any strategy logic — so Modules A, B, and C are structurally incapable of being built without the safety gate already in place, rather than having safety bolted on afterward.

## The prompt

Another single-word "yes" to continue. As with Phase 2 part 2, the detailed decisions here extend the original spec's safety requirements (deterministic circuit breaker, hardcoded leverage/SL limits the LLM can never override, dry-run-by-default with explicit-confirmation live switch) rather than responding to new spec text.

## Decisions made in this phase

1. **Architectural isolation over access control.** The spec says "the LLM can never disable stop-losses or increase leverage." Rather than relying on prompt instructions or runtime permission checks (which an LLM could, in principle, be tricked around), `safety/` simply never imports the Anthropic SDK — there is no code path from an LLM call into a risk parameter. This is enforced as a real test (`test_safety_isolation.py`) that scans every file under `safety/` for any reference to `anthropic` and fails if found. It's the difference between "we told the model not to" and "the model has no mechanism to."
2. **Real, verified 2026 macro calendar, not a placeholder.** Rather than stub out `macro_calendar.py` with fake dates or defer it, looked up the actual FOMC (federalreserve.gov), CPI, and NFP (bls.gov) 2026 release schedules and hardcoded all 32 verified dates. Chose hardcoding *deliberately* over a live feed: FOMC dates aren't available from any free real-time API, and BLS/Fed publish their schedules a year ahead, so a verified static list is arguably more reliable than a scraped one — the tradeoff is it needs manual refresh before each new year (documented in the file's docstring as a maintenance requirement).
3. **Confirmation-phrase gate for live mode, not NLP intent parsing.** `execution_mode.request_live_mode()` requires an exact string match ("CONFIRM LIVE TRADING"), not an LLM's judgment that the user "seems to want" live trading. The orchestrator (Phase 8) will relay the human's literal Telegram text into this function — Haiku's role is to relay, never to decide. Reverting to dry-run has no gate at all, since moving to the safer state should never be blocked.
4. **Circuit breaker fails safe on insufficient data.** `check_volatility_spike()` returns `False` (no trigger) rather than raising or defaulting to "always trigger" when there isn't enough ATR history yet to establish a baseline. Chose this direction deliberately: on a cold start, the system shouldn't force-liquidate everything just because it hasn't observed enough history to judge normal volatility — that failure mode (spurious liquidation) is worse here than briefly running without volatility protection while ATR history accumulates. Discussed as a tradeoff, not an oversight.
5. **Drawdown check lives separately from the circuit breaker.** `is_max_drawdown_breached()` sits in `risk_limits.py`, not folded into `evaluate_circuit_breaker()`, because it needs portfolio P&L state that only the orchestrator/portfolio layer tracks — the circuit breaker itself only looks at market data (OHLC) and the calendar. Both are "always active" per the spec, just called from different layers.

## Bug caught during testing

`check_volatility_spike()` initially returned `numpy.bool_` instead of a native Python `bool` (a pandas comparison artifact). The logic was correct, but `result is True` identity checks in the test suite failed anyway — caught immediately because the tests were written to check exact boolean identity, not just truthiness. Fixed by wrapping the return value in `bool(...)`. Small, but a reminder that pandas/numpy boolean results need explicit casting before they're used with `is` comparisons anywhere downstream (Telegram message formatting, JSON serialization, etc.).

## What got built and verified

- `safety/limits.py`, `safety/risk_limits.py`, `safety/circuit_breaker.py`, `safety/macro_calendar.py`, `safety/execution_mode.py`.
- 17 pytest tests across 4 test files, all passing — covering volatility-spike detection (including the fail-safe insufficient-history case), macro-blackout window logic, risk-limit clamping, the execution-mode gate, and the anthropic-import isolation check.
- Live sanity check against the real 2026 calendar: 32 events loaded, correctly reports no blackout active "now" (2026-08-17), and correctly identifies the next 3 upcoming events (NFP Sep 4, CPI Sep 11, FOMC Sep 16).

## Still pending

- The safety kernel is a library, not yet wired into anything — Modules A/B/C (Phases 4–6) will import from it, and the orchestrator (Phase 8) will own the actual portfolio-level force-liquidation action when `evaluate_circuit_breaker()` or `is_max_drawdown_breached()` trips.
