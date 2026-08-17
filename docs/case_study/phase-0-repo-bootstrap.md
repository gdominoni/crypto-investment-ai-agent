# Phase 0: Repo Bootstrap

**Goal:** stand up the GitHub repository, local git, folder structure, and documentation skeleton before any trading logic is written.

## The initiating prompt

The project began with a single detailed specification message from the human director, assigning the AI the role of "Senior AI & Software Architect and Project Manager" and laying out:
- The 3-module architecture (Cash & Carry, Trend-Following, Volatility/ML Gate) with dynamic capital allocation.
- The cost strategy (cheap/free VPS for the live daemon, local PC for heavy compute, Haiku-only on the server).
- Backtest rules (12-month out-of-sample holdout, dynamic minimum-trade-count filter, Win Rate → Sortino → Net Profit ranking).
- Non-negotiable safety rules (deterministic circuit breaker, hardcoded leverage/SL limits the LLM can never override, dry-run-by-default).
- A request for three concrete deliverables: a requirements list, a phased execution plan, and an explicit "Human Action #1" — the first batch of things only a human can do (sign-ups, key generation, confirmations).

This "spec-first" prompting style — a long, structured brief up front rather than an incremental back-and-forth — is why the roadmap below could be drafted in one pass instead of being discovered phase by phase.

## Decisions made in this phase

1. **Repo name vs. local folder name.** The spec asked for the GitHub repo to exactly match the local folder name. The local folder was `Crypto investment AI Agent` (with spaces), which GitHub repo names don't allow. Flagged to the human, who confirmed the kebab-case slug `crypto-investment-ai-agent`.
2. **Model naming correction.** The spec referenced `claude-3-5-haiku`. That naming scheme is outdated; the current equivalent lightweight model is `claude-haiku-4-5`, with `claude-sonnet-5` as the local/dev-time model. Roadmap updated to use current model IDs.
3. **CryptoPanic → CryptoCompare News API + RSS.** CryptoPanic's free API tier was discontinued after this project's initial spec was written. Replaced with CryptoCompare's free News API as the primary structured source, with a small RSS feed set (CoinDesk, Cointelegraph, The Block, Decrypt) as redundancy. The downstream interface (headlines/summaries in → Haiku-produced JSON sentiment out) is unchanged, so this substitution doesn't affect any later phase. Full rationale in [decisions-log.md](decisions-log.md).
4. **Requirements split into `requirements-server.txt` / `requirements-local.txt`**, with the server file kept intentionally minimal (no ML/backtesting libraries) to keep the 24/7 VPS footprint small, matching the "local offloading" cost strategy from the spec.

## Human actions completed in this phase

- Installed and authenticated the GitHub CLI (`gh auth login`).
- Confirmed the repo name.
- Chose Binance (Spot + Futures testnet) as the primary exchange.
- Chose Oracle Cloud Free Tier as the hosting target, with Hetzner as a fallback.
- Created the Telegram bot via BotFather.
- Created an Anthropic API key.

## What got built

- Folder structure separating safety-critical code (`safety/`), the three strategy modules (`modules/`), the cross-module orchestrator (`orchestrator/`), data ingestion (`data_ingestion/`), and local-only backtesting (`backtesting/`).
- `.gitignore` covering secrets, virtual environments, databases, and framework runtime directories (Freqtrade/Hummingbot generate large local state that shouldn't be versioned).
- `.env.example` documenting every required secret without exposing real values.
- Main `README.md` as the living case-study document for non-technical readers.
