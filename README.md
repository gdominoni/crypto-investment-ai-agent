# Crypto Investment AI Agent

A 24/7 autonomous, cost-optimized AI agent system for crypto trading, controlled via Telegram, built as a public **"Vibe Coding" case study**: every phase of this repo was designed and implemented through a conversation with an AI coding assistant (Claude), with a human directing priorities and handling anything requiring real-world action (accounts, keys, servers). This README is written for **non-technical readers** — if you've never coded before, you should be able to follow along and understand *why* every decision was made.

Status: 🚧 early build — currently in **Phase 0 (repo & project scaffold)**. Full roadmap below.

---

## 1. Project Vision

The agent manages a crypto portfolio split across three independent, isolated strategy modules, each targeting a different market condition:

| Module | Strategy | Market condition it targets |
|---|---|---|
| **A — Market-Neutral / Cash & Carry** | Delta-neutral spot + short futures, harvesting funding-rate yield | Sideways / any regime (designed to be regime-agnostic) |
| **B — Dynamic Trend-Following** | Systematic trend strategies on futures/spot, self-optimizing | Trending markets, up or down |
| **C — Volatility Gate / ML** | Probabilistic ML signals, gated by a hardcoded circuit breaker | Regime transitions / high uncertainty |

Capital is **not** split evenly or statically — the agent continuously analyzes market regime, funding yields, and drawdown history to recommend how much capital each module should hold, and rebalances accordingly.

### Financial architecture principles

1. **Safety before returns.** A deterministic circuit breaker (hardcoded, not AI-controlled) can force the entire portfolio to 100% USDT if volatility spikes or a high-impact macro event (FOMC, CPI, NFP) is imminent. No AI component — including the LLM — can ever disable a stop-loss or raise leverage past hardcoded limits.
2. **Everything starts in paper trading.** All three modules default to dry-run/simulated execution. Moving to real money requires an explicit, human-issued Telegram confirmation command.
3. **Overfitting is the enemy.** Strategy selection always reserves the most recent 12 months of data as an out-of-sample (OOS) test the strategy never saw during optimization.
4. **Compute cost is a first-class constraint** (see below) — the live server should cost close to $0/month to run indefinitely.

---

## 2. Cost & Infrastructure Architecture

The system is deliberately split across two environments to keep 24/7 operating costs near zero:

| | Where it runs | What runs there | Why |
|---|---|---|---|
| **Live daemon** | Cloud VPS (Oracle Cloud Free Tier, Hetzner as fallback) | Telegram bot, order execution, monitoring, lightweight NLP | Needs to be always-on; kept minimal so it fits a free/cheap instance |
| **Heavy compute** | Local PC | Historical data downloads, backtesting sweeps, hyperparameter optimization, ML model training | Expensive to run 24/7 for no benefit — these are periodic, bursty jobs |

**AI model tiering (cost optimization):**
- The **server** exclusively uses **Claude Haiku** (currently `claude-haiku-4-5`) for Telegram interaction, JSON status formatting, and news sentiment extraction — cheap enough to run continuously.
- **Sonnet-class models** are only ever used **locally**, during development and backtesting analysis — never called by the always-on daemon. This keeps the 24/7 API bill negligible regardless of how often the bot posts updates.

**Data ingestion is zero-cost:**
- Market data: Binance REST API (spot + futures testnet initially) via `ccxt`.
- Macro data: `yfinance` (SPX, Gold) and the FRED API (CPI, rates, unemployment) — both free.
- News: CryptoCompare News API (free, structured JSON) + a curated RSS feed set (CoinDesk, Cointelegraph, The Block, Decrypt) as redundancy. *(Originally planned around CryptoPanic — replaced after CryptoPanic moved its API behind a paid plan; see [Technical Decisions Log](docs/case_study/decisions-log.md).)*

---

## 3. Repository Structure

```
crypto-investment-ai-agent/
├── safety/                      # Deterministic circuit breaker + hardcoded risk limits (shared by all modules)
├── modules/
│   ├── module_a_cash_carry/     # Delta-neutral funding-rate arbitrage (Hummingbot)
│   ├── module_b_trend_following/# Trend-following strategies (Freqtrade)
│   └── module_c_volatility_ml/  # ML volatility/regime gate
├── orchestrator/                # Telegram bot, capital allocator, cross-module KPI ranking
├── data_ingestion/
│   ├── market_data/             # Exchange OHLCV + funding rates
│   ├── macro_data/              # yfinance / FRED
│   └── news_sentiment/          # CryptoCompare News API + RSS → Haiku sentiment extraction
├── backtesting/                 # Local-only: IS/OOS sweeps, hyperopt, statistical filters
├── scripts/                     # One-off / operational scripts
├── docs/case_study/             # This project's "vibe coding" log: prompts, decisions, rationale
├── tests/
├── requirements-server.txt      # Minimal deps for the 24/7 VPS
├── requirements-local.txt       # Full deps for local backtesting/ML (superset of server)
└── .env.example                 # Template for required secrets (never commit the real .env)
```

---

## 4. Development Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Repo scaffold, `.gitignore`, README, folder structure | ✅ in progress |
| 1 | Credentials & accounts (GitHub, Telegram, Anthropic, Binance testnet, FRED) | ✅ done |
| 2 | Local data ingestion: market data, macro data, news sentiment | ✅ done |
| 3 | Safety kernel: deterministic circuit breaker + hardcoded risk limits | ✅ done |
| 4 | Module B (Freqtrade): strategies, IS/OOS backtesting, hyperopt, ranking | ⏳ next |
| 5 | Module A (Hummingbot): funding-rate scanner, delta-neutral paper trading | ⏳ |
| 6 | Module C: ML volatility/regime model, gated by the safety kernel | ⏳ |
| 7 | Dynamic capital allocator across A/B/C | ⏳ |
| 8 | Telegram Orchestrator (Haiku-powered), explicit live-mode confirmation flow | ⏳ |
| 9 | VPS deployment: Docker Compose, supervision, monitoring | ⏳ |
| 10 | Multi-week all-module dry-run soak test | ⏳ |
| 11 | Go/no-go review before any live capital is ever committed | ⏳ |

---

## 5. Replicating This Project From Scratch

*(Filled in incrementally as each phase lands — this section will eventually be a complete, standalone walkthrough for a non-technical reader.)*

1. Prerequisites: a GitHub account, a Telegram account, an Anthropic API key, a Binance account (testnet keys are free), a free Oracle Cloud or Hetzner account.
2. Clone this repo.
3. Copy `.env.example` to `.env` and fill in your own keys (never commit this file).
4. Create a local virtual environment and install dependencies:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-local.txt
   ```
5. Run the test suite to confirm the safety kernel is working: `pytest tests/`
6. *(Server deployment steps — added when Phase 9 is built.)*

---

## 6. Safety & Guardrails (Inviolable)

These are hardcoded in [`safety/`](safety/) and cannot be altered by the LLM or by any Telegram command short of editing and redeploying the code itself:

- Deterministic circuit breaker: forces 100% USDT on volatility spikes or ahead of high-impact macro releases.
- Hardcoded ceilings on leverage and position size that no AI component can raise.
- Stop-loss / take-profit enforcement independent of any LLM output.
- Every module defaults to dry-run; live execution requires an explicit human-issued Telegram confirmation.

---

## 7. Case Study Log

The full "vibe coding" process — exact prompts used, technical decisions and their rationale, and dead ends — is documented in [`docs/case_study/`](docs/case_study/):
- [Phase 0: Repo Bootstrap](docs/case_study/phase-0-repo-bootstrap.md)
- [Phase 2 (part 1): Macro & Cross-Asset Data Ingestion](docs/case_study/phase-2-macro-data.md)
- [Phase 2 (part 2): Exchange Market Data & News Sentiment](docs/case_study/phase-2-market-and-news-data.md)
- [Phase 3: Safety Kernel](docs/case_study/phase-3-safety-kernel.md)
- [Technical Decisions Log](docs/case_study/decisions-log.md)
