# News Sentiment Ingestion

Pulls headlines/summaries from the CryptoCompare News API (primary) and a curated RSS feed set (secondary/redundancy), deduplicates them, and passes them to Claude Haiku for structured JSON sentiment extraction (`{asset, sentiment, magnitude, event_type}`).

Originally planned around CryptoPanic; switched after CryptoPanic's free API tier was discontinued. See [../../docs/case_study/decisions-log.md](../../docs/case_study/decisions-log.md).

Status: not yet built (Phase 2).
