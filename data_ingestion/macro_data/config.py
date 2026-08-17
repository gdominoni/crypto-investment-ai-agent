"""Single source of truth for macro/cross-asset feature series (Phase 2 spec).

Deliberately restricted to a small, curated set. Module C (FreqAI) trains on
these features directly, so every series added here is a series that can
overfit a small crypto dataset -- the list only grows with a specific,
justified signal in mind.
"""

# yfinance tickers: ticker -> internal series name
YFINANCE_TICKERS = {
    "^GSPC": "sp500",          # S&P 500 -- risk-on/tech correlation proxy
    "^IXIC": "nasdaq_composite",  # Nasdaq Composite -- risk-on/tech correlation proxy
    "^VIX": "vix",             # CBOE Volatility Index -- fear gauge / risk-off detection
    "GC=F": "gold",            # Gold futures -- macro hedge / risk-off flows
    "CL=F": "crude_oil",       # WTI Crude futures -- macro/inflation proxy
}

# FRED series: series_id -> internal series name
FRED_SERIES = {
    "FEDFUNDS": "fed_funds_rate",   # Monetary policy stance
    "UNRATE": "unemployment_rate",  # Labor market
    "CPIAUCSL": "cpi",              # Inflation
    "INDPRO": "industrial_production",  # Real economy
}
