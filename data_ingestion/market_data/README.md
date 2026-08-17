# Market Data Ingestion

Exchange OHLCV and funding-rate data via `ccxt`, Binance primary. Zero-cost — direct public REST endpoints, no API key and no paid data vendor.

Curated symbol/timeframe list lives in [`config.py`](config.py): `BTC/USDT` and `ETH/USDT` across `1h`/`4h`/`1d`, 3 years of history by default.

**`binance_fetcher.py`** pulls three things:
- Spot OHLCV → `data/market/binance/spot/`
- Perpetual futures OHLCV (Binance USDⓈ-M) → `data/market/binance/futures/`
- Funding-rate history (needed for Module A's cash & carry yield calculation) → `data/market/binance/funding/`

Historical data is always pulled from **mainnet** (testnet history is sparse/unreliable), even though live paper-trading execution in Phase 4/5 will connect to Binance testnet per `BINANCE_TESTNET` in `.env` — these are separate concerns (backtesting data vs. execution venue).

Run locally with:
```
python -m data_ingestion.market_data.binance_fetcher
```

Status: ✅ built (Phase 2).
