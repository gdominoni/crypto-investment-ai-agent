"""News source config. CryptoCompare is the only source the live pipeline
uses (free, structured JSON, aggregates ~50+ outlets) -- replaced
CryptoPanic after its free API tier was discontinued.
"""

CRYPTOCOMPARE_NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/"
