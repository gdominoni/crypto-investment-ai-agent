"""News source config (Phase 2).

CryptoCompare is primary (free, structured JSON, aggregates ~50+ outlets).
RSS feeds are a redundancy layer in case CryptoCompare misses a story or
has an outage. See docs/case_study/decisions-log.md for why this replaced
CryptoPanic.
"""

CRYPTOCOMPARE_NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/"

RSS_FEEDS = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
    "decrypt": "https://decrypt.co/feed",
    "theblock": "https://www.theblock.co/rss.xml",
}
