# News Ingestion

- **`cryptocompare_fetcher.py`** — fetches headlines/summaries from the CryptoCompare News API. The only file in this folder used by the live pipeline.
- **`config.py`** — the CryptoCompare endpoint URL.

Sentiment extraction itself (Claude Haiku, asset/sentiment/magnitude/event-type scoring) happens downstream, in `llm_pipeline/haiku_sonnet_pipeline.py::haiku_scout` — not in this folder.

Run locally with:
```
python -m data_ingestion.news_sentiment.cryptocompare_fetcher
```

Status: ✅ used in production.
