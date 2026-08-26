"""Fetches headlines from a curated RSS feed set -- redundancy for the
CryptoCompare News API (see config.RSS_FEEDS).

Local-only. Run as:
    python -m data_ingestion.news_sentiment.rss_fetcher
"""

from datetime import datetime, timezone

import feedparser

from data_ingestion.news_sentiment.config import RSS_FEEDS


def fetch_rss_news(limit_per_feed: int = 20) -> list[dict]:
    items = []
    for source, url in RSS_FEEDS.items():
        parsed = feedparser.parse(url)
        for entry in parsed.entries[:limit_per_feed]:
            published = entry.get("published_parsed")
            items.append(
                {
                    "headline": entry.get("title", ""),
                    "summary": entry.get("summary", "")[:500],
                    "source": source,
                    "url": entry.get("link", ""),
                    "published_at": (
                        datetime(*published[:6], tzinfo=timezone.utc).isoformat() if published else None
                    ),
                }
            )
    return items


if __name__ == "__main__":
    news = fetch_rss_news(limit_per_feed=3)
    for item in news:
        print(f"[{item['source']}] {item['headline']}")
