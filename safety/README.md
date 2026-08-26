# Safety Kernel

Deterministic, hardcoded risk controls shared by all three modules. **Nothing here is AI-controlled.** The safety package has no code path that imports or calls the Anthropic SDK — enforced by [`tests/test_safety_isolation.py`](../tests/test_safety_isolation.py), which fails CI if any file under `safety/` ever references `anthropic`. This is the concrete, testable form of "the LLM can never disable stop-losses or raise leverage": not a permission check, a structural fact.

**`limits.py`** — hardcoded constants (max leverage, max position size, max drawdown, ATR spike multiplier, macro blackout window). Changing a limit means editing this file and redeploying — no config file, database row, or Telegram command can touch it.

**`risk_limits.py`** — the only sanctioned way any module (or the LLM-driven orchestrator) touches a risk parameter. Every function clamps *toward* `limits.py`'s bounds; none can expand past them: `clamp_leverage()`, `clamp_position_size_pct()`, `is_max_drawdown_breached()`.

**`macro_calendar.py`** — a hardcoded, verified 2026 calendar of FOMC decisions, CPI releases, and NFP/Employment Situation releases (32 events), sourced from federalreserve.gov and bls.gov. Hardcoded deliberately: these dates are published far in advance and there's no reliable free live-feed API for FOMC specifically, so a verified static list is *more* deterministic than a fetched one. **Needs manual refresh before January each year** — see the file's docstring.

**`circuit_breaker.py`** — combines two checks into one deterministic decision:
- `check_volatility_spike()`: current ATR(14) vs. a 90-period rolling baseline; triggers past a hardcoded multiplier. Fails safe to "no trigger" when there isn't enough history to judge what's normal.
- `check_macro_blackout()`: is `now` within the hardcoded before/after window of any calendar event?
- `evaluate_circuit_breaker()` returns a `CircuitBreakerDecision(should_liquidate, reasons)` — the trigger for forcing a module to 100% USDT.

**`execution_mode.py`** — the live/dry-run gate. Always boots in `dry_run` (no state file = dry-run, by construction). `request_live_mode()` only flips to live on an *exact* string match against a hardcoded confirmation phrase — the orchestrator relays the human's literal Telegram text into this function, Haiku never decides on its own whether the phrase was "close enough." Reverting to dry-run is always allowed, no confirmation needed.

Status: ✅ built (Phase 3), 17 tests passing.
