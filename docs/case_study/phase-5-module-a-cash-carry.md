# Phase 5: Module A — Cash & Carry (Hummingbot)

**Goal:** determine whether delta-neutral funding-rate arbitrage is actually worth running (local analysis, before touching Hummingbot at all), then stand up Hummingbot via Docker with a strategy config informed by that analysis.

## The prompt

"proceed with phase 5" — a plain continuation. As with Phases 4's Docker decision, the specific engineering choices below extend the established patterns (local heavy analysis before live infrastructure, Docker over pip, honest reporting of blockers) rather than responding to new spec text.

## Decisions made in this phase

1. **Analyze funding yield locally before building anything live.** Hummingbot has no historical backtesting mode for this strategy type, so the equivalent of Module B's backtest step had to be built from scratch: `funding_analysis.py` reuses the funding-rate history Phase 2 already downloaded to answer "has this actually been profitable, after fees?" *before* any Hummingbot config exists. This mirrors the project's core cost/architecture principle — heavy analytical work happens locally in plain Python, not inside the live-trading framework.
2. **Fee model stated as an explicit simplification, not hidden precision.** Net yield is computed as gross annualized funding minus a single open+close round-trip cost amortized over a year — deliberately simple and clearly documented as such, rather than modeling the more complex, uncertain trading frequency the live Hummingbot strategy might actually exhibit (it trades on basis convergence, a proxy for funding favorability, not funding directly).
3. **Discovered mid-build: Hummingbot's paper-trade simulator doesn't cover perpetual connectors.** This wasn't assumed going in — found by grepping the actual connector source (`connector/exchange/paper_trade/` exists, `connector/derivative/paper_trade/` doesn't) rather than guessing from the strategy template. Resolved by using Binance's own Futures Testnet (`binance_perpetual_testnet`) for the perpetual leg — a different, separate sandbox from Spot Testnet, requiring its own API keys. Logged in the decisions log since it's a genuinely non-obvious asymmetry a reader could easily miss.
4. **Traced the headless startup failure to its actual root cause instead of stopping at the first error.** The first attempt failed with an `EOFError` from a login-screen prompt; instead of treating that as "needs `-it`," passed an explicit `--config-password` flag and re-ran, which surfaced the real blocker two layers deeper: a fresh `conf/` directory has never had a local encryption password set, so there's nothing for headless mode to validate against on a first run. That's a one-time, inherently interactive setup step (choosing the password, connecting exchange credentials) — not something safe to script around by faking Hummingbot's encryption files.
5. **Leverage fixed at 1x in the strategy config**, explicitly tied back to the Phase 3 safety kernel's `MAX_LEVERAGE = 3.0` ceiling in the config's own comments — this strategy harvests a yield spread, it isn't meant to take a leveraged directional position, and the config says so rather than leaving that as an implicit assumption.

## What got built and verified

- `modules/module_a_cash_carry/funding_analysis.py` — ran against 3 years of real Binance funding-rate data (re-fetched fresh via Phase 2's `binance_fetcher.py`, since the earlier smoke-test data had been deleted): **BTC/USDT and ETH/USDT both show ~7% annualized funding yield, net of conservative round-trip fees, and both are still attractive in the most recent 30 days.** Unlike Module B's first candidate, this is a genuinely encouraging result — reported as such, not downplayed to seem more "balanced."
- 4 unit tests against synthetic funding series (positive/negative/net-yield/threshold-scaling), caught the same `numpy.bool_` vs. Python `bool` issue flagged as a general lesson in the Phase 3 case study — confirms that lesson was worth writing down, since it recurred here independently.
- `docker-compose.yml` + `conf/strategies/conf_spot_perpetual_arbitrage_btc.yml`, parameterized directly from the funding analysis (`min_opening_arbitrage_pct` derived from the measured round-trip cost, not a guessed number).
- Attempted a real headless startup against Docker (the same rigor applied to Freqtrade in Phase 4) and traced the failure to its precise, correctly-diagnosed cause rather than stopping at a surface-level error.

## Still pending (genuine human action required)

Unlike every prior phase, this one can't be fully verified end-to-end without the human director:
1. Binance Futures Testnet API keys (testnet.binancefuture.com — separate from the Spot Testnet keys already in `.env`).
2. A one-time **interactive** Hummingbot session (`docker compose run --rm -it hummingbot`) to set the local config password and connect the Futures Testnet credentials — instructions are in [Module A's README](../../modules/module_a_cash_carry/README.md#human-action-needed-before-this-can-actually-run).

After that one-time step, headless paper-trade runs should work using the command already written into `docker-compose.yml`'s comments.
