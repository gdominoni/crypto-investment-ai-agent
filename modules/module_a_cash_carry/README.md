# Module A — Market-Neutral / Cash & Carry

Delta-neutral funding-rate arbitrage: long spot + short perpetual futures on the same asset, harvesting the funding-rate yield while staying market-neutral. Built on **Hummingbot, run via Docker** (same rationale as Module B — see the main [README](../../README.md#2-cost--infrastructure-architecture)).

## Local analysis first: is this trade even worth running?

Hummingbot has no historical backtesting mode for this strategy type, so before touching Hummingbot at all, `funding_analysis.py` answers the question locally, for free, using the funding-rate history already pulled in Phase 2: **has holding this position actually been profitable, after fees, historically and right now?**

Run with `python -m modules.module_a_cash_carry.funding_analysis`. Real result against 3 years of Binance data (2026-08-17):

| Symbol | Positive funding | Annualized yield (gross) | Annualized yield (net of round-trip fees) | Currently attractive |
|---|---|---|---|---|
| BTC/USDT | 84.6% | 7.23% | 6.95% | ✅ yes |
| ETH/USDT | 84.6% | 7.46% | 7.18% | ✅ yes |

Both pairs clear their costs comfortably, both historically and in the most recent 30 days — a genuinely encouraging first result, not a cherry-picked one (contrast with Module B, where the first strategy candidate was rejected). The fee-cost model is a deliberately conservative simplification (a single open+close round trip amortized over a year of holding, at standard non-VIP taker fees) — see the module's docstring for the exact assumption.

This analysis also derives `min_opening_arbitrage_pct` for the Hummingbot strategy config below: round-trip cost × a 1.5x safety margin ≈ 0.42%.

## An important nuance: paper trade doesn't cover both legs

Hummingbot's built-in paper-trade simulator only wraps **spot** exchange connectors — there's no simulated equivalent for perpetual/derivative connectors. So this strategy's two legs run in two different (both zero-real-risk) modes:
- **Spot leg**: `binance_paper_trade` — Hummingbot's own simulator, fully fake fills and balance, no API key needed.
- **Perpetual leg**: `binance_perpetual_testnet` — Binance's own dedicated Futures Testnet sandbox, which is a **separate site and separate account from Spot Testnet**, requiring its own API keys from testnet.binancefuture.com (not testnet.binance.vision).

Both are zero-real-money, consistent with the project's dry-run-by-default rule — they just achieve it through two different mechanisms, which is worth understanding rather than assuming "paper trade" is one uniform thing.

Leverage is fixed at 1x in the config, well inside the hardcoded safety-kernel ceiling (`safety/limits.py`: `MAX_LEVERAGE = 3.0`) — this strategy harvests yield, it doesn't take a leveraged directional bet.

## What's built

- `funding_analysis.py` — the yield analysis above, unit tested (4 tests, synthetic data).
- `docker-compose.yml` — runs the official `hummingbot/hummingbot:latest` image.
- `conf/strategies/conf_spot_perpetual_arbitrage_btc.yml` — the `spot_perpetual_arbitrage` strategy config (Hummingbot's built-in cash & carry strategy), parameterized from the funding analysis above.

## Human action needed before this can actually run

Traced Hummingbot's headless startup all the way to its real blocker: a brand-new `conf/` directory has never had a local encryption password set, so headless mode has nothing to validate against on first run — and entering that password (and connecting exchange credentials) for the first time is an inherently interactive step that can't be safely scripted around. This is a one-time setup, not a recurring one:

1. Get Binance Futures Testnet API keys from **testnet.binancefuture.com** (separate from your existing Spot Testnet keys) and add them to `.env` as `BINANCE_FUTURES_TESTNET_API_KEY` / `_SECRET`. Also set `HUMMINGBOT_CONFIG_PASSWORD` in `.env` to any password you choose (it encrypts local files only, it's not an exchange credential).
2. Run the one-time interactive setup:
   ```
   cd modules/module_a_cash_carry
   docker compose --env-file ../../.env run --rm -it hummingbot
   ```
   When prompted, set the password to match `HUMMINGBOT_CONFIG_PASSWORD`, then inside the Hummingbot CLI run `connect binance_perpetual_testnet` and paste in the Futures Testnet key/secret. Exit with `exit`.
3. After that one-time setup, this will run headless:
   ```
   docker compose --env-file ../../.env run --rm hummingbot \
     python bin/hummingbot_quickstart.py --headless \
     --config-file-name conf_spot_perpetual_arbitrage_btc.yml \
     --config-password "$HUMMINGBOT_CONFIG_PASSWORD"
   ```

Status: 🚧 funding analysis complete and genuinely attractive; Docker/strategy config built; blocked on the one-time interactive credential setup above (Phase 5).
