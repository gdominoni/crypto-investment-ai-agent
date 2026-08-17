# Phase 2 (part 2): Exchange Market Data & News Sentiment

**Goal:** finish Phase 2 — Binance market data (spot, futures, funding rates) and the news sentiment pipeline (CryptoCompare + RSS → Haiku).

## The prompt

A single-word "yes" to continue building the remaining two legs of Phase 2 after the macro data ingestion (part 1) and FRED key confirmation. No new spec details were given here — decisions below were made by extending the same patterns already established (curated config, local-only, smoke-tested before commit).

## Decisions made in this phase

1. **Historical market data comes from Binance mainnet, not testnet**, even though live paper-trading execution (Phase 4/5) will connect to testnet. Testnet order books/history are sparse and unreliable; OHLCV and funding-rate history are public endpoints on mainnet requiring no API key at all, so there's no security tradeoff in using mainnet purely for historical data.
2. **Symbol list restricted to BTC/USDT and ETH/USDT** — the same "small curated set, expand deliberately" pattern used for the macro features, for the same reason (liquidity/data quality for the two modules that need it most: Module A cares about funding yield + depth, Module B cares about slippage).
3. **Funding-rate history fetched alongside OHLCV**, not deferred to Module A's build phase — it's cheap to pull now and Module A's entire strategy (Phase 5) depends on historical funding yield analysis, so having it available from Phase 2 onward avoids a dependency later.
4. **Haiku JSON parsing needed a markdown-fence strip.** The first live smoke test of `haiku_sentiment.py` failed `json.loads` — Haiku wrapped its output in ` ```json ... ``` ` fences despite the system prompt explicitly saying not to. Fixed by stripping fences before parsing rather than fighting the model with more prompt instructions, since LLM output formatting isn't fully deterministic even with a chosen instruction. This is worth remembering for every future Haiku JSON-extraction call in the project (orchestrator status formatting, etc.) — always parse defensively, never assume a clean fenceless response.
5. **RSS feed URLs verified live**, not assumed. The Block's actual feed is `theblock.co/rss.xml`, not the `/rss` guess.

## What got built and verified (all smoke-tested against live endpoints)

- `data_ingestion/market_data/binance_fetcher.py` — spot OHLCV, futures OHLCV, and funding-rate history, tested with a 7-day window across both symbols and all three timeframes.
- `data_ingestion/news_sentiment/cryptocompare_fetcher.py` — live pull confirmed (~15+ articles/request).
- `data_ingestion/news_sentiment/rss_fetcher.py` — all 4 feeds confirmed live (CoinDesk, Cointelegraph, Decrypt, The Block).
- `data_ingestion/news_sentiment/aggregator.py` — combined + deduplicated, 128 unique articles from a single live pull.
- `data_ingestion/news_sentiment/haiku_sentiment.py` — full pipeline tested end-to-end with a live Haiku call on 5 real headlines; returned valid, well-formed sentiment JSON after the markdown-fence fix.
- FRED fetcher (Phase 2 part 1) re-verified live now that `FRED_API_KEY` was added to `.env` — all 4 series pulled successfully.

## Phase 2 status: complete

All three legs (market data, macro data, news sentiment) are built and verified against live sources. Next phase is the safety kernel (Phase 3) — the deterministic circuit breaker and hardcoded risk limits every module will import, built before any strategy logic so nothing downstream can be built without it in place.
