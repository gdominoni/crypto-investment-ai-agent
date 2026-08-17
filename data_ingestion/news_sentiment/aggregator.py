"""Combines CryptoCompare and RSS news, deduplicated by normalized title.

Simple exact-match dedup on a normalized title -- the two source sets
overlap but aren't identical, and this is enough to drop the obvious
repeats without pulling in a fuzzy-matching dependency.
"""

import re

from data_ingestion.news_sentiment.cryptocompare_fetcher import fetch_cryptocompare_news
from data_ingestion.news_sentiment.rss_fetcher import fetch_rss_news


def _normalize(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def fetch_all_news() -> list[dict]:
    combined = fetch_cryptocompare_news() + fetch_rss_news()
    seen = set()
    deduped = []
    for item in combined:
        key = _normalize(item["headline"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


if __name__ == "__main__":
    news = fetch_all_news()
    print(f"{len(news)} unique articles")
    for item in news[:10]:
        print(f"[{item['source']}] {item['headline']}")
