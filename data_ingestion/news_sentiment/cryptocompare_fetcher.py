"""Fetches aggregated crypto news from the CryptoCompare News API.

Free tier; CRYPTOCOMPARE_API_KEY raises the rate limit but isn't required
for basic use. Local-only. Run as:
    python -m data_ingestion.news_sentiment.cryptocompare_fetcher
"""

import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from data_ingestion.news_sentiment.config import CRYPTOCOMPARE_NEWS_URL


def fetch_cryptocompare_news(limit: int | None = None) -> list[dict]:
    load_dotenv()
    api_key = os.environ.get("CRYPTOCOMPARE_API_KEY")
    headers = {"authorization": f"Apikey {api_key}"} if api_key else {}

    response = requests.get(CRYPTOCOMPARE_NEWS_URL, params={"lang": "EN"}, headers=headers, timeout=15)
    response.raise_for_status()
    articles = response.json().get("Data", [])

    items = [
        {
            "headline": a["title"],
            "summary": a.get("body", "")[:500],
            "source": a.get("source_info", {}).get("name") or a.get("source", ""),
            "url": a["url"],
            "published_at": datetime.fromtimestamp(a["published_on"], tz=timezone.utc).isoformat(),
        }
        for a in articles
    ]
    return items[:limit] if limit else items


if __name__ == "__main__":
    news = fetch_cryptocompare_news(limit=5)
    for item in news:
        print(f"[{item['source']}] {item['headline']}")
