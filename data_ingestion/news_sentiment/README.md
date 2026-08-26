# News Sentiment Ingestion

Pulls headlines/summaries from the CryptoCompare News API (primary) and a curated RSS feed set (secondary/redundancy), deduplicates them, and passes them to Claude Haiku for structured JSON sentiment extraction.

Originally planned around CryptoPanic; switched after CryptoPanic's free API tier was discontinued. See [../../docs/case_study/decisions-log.md](../../docs/case_study/decisions-log.md).

**Pipeline:**
1. `cryptocompare_fetcher.py` — structured JSON news from CryptoCompare.
2. `rss_fetcher.py` — CoinDesk, Cointelegraph, Decrypt, The Block (see `config.py`).
3. `aggregator.py` — merges both, deduplicates by normalized title.
4. `haiku_sentiment.py` — sends headlines to Claude Haiku (`claude-haiku-4-5`), gets back:
   ```json
   {"headline": "...", "asset": "BTC", "sentiment": "bullish", "magnitude": 3, "event_type": "regulatory"}
   ```
   This is the only place in the ingestion layer that calls the Anthropic API, and it's deliberately the cheap model — the server daemon will call this continuously in production (see README Section 2, cost tiering). Sonnet is never used in this path.

Run locally with:
```
python -m data_ingestion.news_sentiment.aggregator     # raw dedup'd headlines
python -m data_ingestion.news_sentiment.haiku_sentiment  # + Haiku sentiment
```

Status: ✅ built (Phase 2).
