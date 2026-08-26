"""Extracts structured sentiment from news headlines using Claude Haiku.

This is the only place in Phase 2 that calls the Anthropic API, and it's
deliberately the cheap model: the server daemon will run this continuously
in production, so cost tiering matters most here (see README Section 2).
Sonnet is never called from this code path.

Local smoke test. Run as:
    python -m data_ingestion.news_sentiment.haiku_sentiment
"""

import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv

MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = """You are a crypto market news sentiment extractor. Given a batch of \
headlines, return a JSON array where each item has exactly these fields:
- "headline": the original headline (verbatim, truncated to 100 chars)
- "asset": the primary crypto asset affected (e.g. "BTC", "ETH", "MARKET" if broad/unclear)
- "sentiment": one of "bullish", "bearish", "neutral"
- "magnitude": integer 1-5, how market-moving this is likely to be (5 = major, 1 = negligible)
- "event_type": one of "regulatory", "macro", "hack_exploit", "adoption", "market_structure", "other"

Return ONLY the JSON array, no prose, no markdown fences."""


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def extract_sentiment(articles: list[dict]) -> list[dict]:
    load_dotenv()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    headlines_block = "\n".join(f"- {a['headline']}" for a in articles)
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": headlines_block}],
    )
    return json.loads(_strip_markdown_fences(response.content[0].text))


if __name__ == "__main__":
    from data_ingestion.news_sentiment.aggregator import fetch_all_news

    sample = fetch_all_news()[:5]
    results = extract_sentiment(sample)
    for r in results:
        print(r)
