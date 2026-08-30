# Open Work

Deliberately-deferred items, with enough detail to pick each up cold.
Ordered by what unblocks the most. Every claim here was measured, not
estimated from intuition -- see `methodology-decisions.md` for the runs.

---

## 1. News/sentiment backfill (GDELT) -- unblocks the project's own title

**Why it's blocking.** This project claims to test whether *market
sentiment* combined with market conditions produces repeatable patterns.
It currently cannot test that at all, and going live does not fix it:

- The whitelist (`llm_pipeline/novel_condition_tester.py::SUPPORTED_INDICATORS`)
  has **no sentiment term**, because acceptance is always decided by a
  backtest and there is no historical sentiment series to backtest against.
- Measured: of 92 conditions Sonnet proposed in the replay, **34% contain
  no event term at all**. Of **771 live tests** opened for Sonnet-discovered
  candidates, **zero** were news-linked -- all came from the mechanical
  hourly scan, which reads price/OHLC/funding only.
- Haiku's sentiment therefore acts as a *search prior* (it decides which
  chart hypothesis gets proposed) and then disappears before anything is
  measured or tracked.

**Verified dead end.** CryptoCompare's news API is live-only: a default
call returns ~50 articles all from today, and `lTs` backward paging
returns empty immediately. **3,527 days of history missing.**

**The plan.**
1. Backfill a daily, per-coin news series 2017 -> present. **GDELT 2.0**
   is the only free source plausibly covering the full window (2015+,
   includes tone scoring). Needs a spike first: verify crypto coverage
   for the smaller coins (DOGE, ADA) in 2017-18, which is the likeliest
   failure point. CryptoPanic is the paid fallback.
2. Score it **with Haiku, batched by day** -- not per article. ~3,527
   daily calls at ~1,500 in / ~300 out tokens is roughly **$11**, call it
   **$35** with retries. Per-article scoring would be ~$250 for no gain.
3. **Score history with the same scorer live uses.** If history is scored
   by GDELT's tone metric and live by Haiku, that is a train/serve skew --
   the backtest validates one variable and live tracks another. Exactly
   the family of bug already found twice in this codebase.
4. Emit `data/news/{COIN}_sentiment_daily.parquet`, then add whitelist
   indicators (`news_sentiment_zscore_30d`, `news_volume_zscore_30d`,
   and ideally `sentiment_price_divergence`). Register them in
   `EVENT_INDICATORS` so `reduced_spec()` treats them as the treatment,
   and in `DAILY_NATIVE_INDICATORS` (daily concept, no intraday version).
5. Re-run the replay. Every candidate's verdict has already changed twice
   this session; a fresh run is needed regardless.

**Effort / cost.** 3-5 days engineering, ~$35 API, plus ~$15-25 for the
replay re-run.

**Note on expectations.** The naive hypothesis ("negative news -> price
falls") is likely priced in and, worse, largely redundant with
`shock_zscore`, which already captures "something violent happened". The
version worth testing is **divergence**: sentiment strongly negative but
price hasn't moved; or price crashed with no news (a liquidation cascade,
mechanically different from a news-driven selloff). Those are only
expressible once sentiment is a term *separate* from price.

---

## 2. Re-run the historical replay end to end

State is at 2023-08-02 and was produced under the pre-audit methodology.
Every verdict has since changed twice (the 2026-08-29 audit, then the
capability work). ~$15-25 in API calls, hours of wall clock.

Do this **after** item 1, not before -- otherwise it gets paid for twice.

---

## 3. Smaller, known, and deliberately left

- **`daily_range_pct` ignores `scale` entirely.** Harmless now that it's
  in `DAILY_NATIVE_INDICATORS` (so the live scan computes it on the daily
  frame), but the lambda's signature still implies a scaling it doesn't do.
- **`TRIGGER_NUMERIC_DEFINITIONS` is hand-synced** to `compute_triggers()`.
  A duplicated number is a number that can drift; deriving it from the
  source would remove the hazard.
- **Freqtrade-era trade-database queries.** Mostly resolved: `telegram/
  kpi_queries.py` and the `/results` command that opened it were deleted
  (they could only ever answer "no trades"), and
  `context_builder.build_technical_snapshot` no longer tells Sonnet the
  missing database is a setup problem. What remains is its open-position
  SQL, which is unreachable while no order is ever placed -- harmless,
  but it is the last piece of the old model still in the LLM's context
  path. `execution/signal_store.py` is NOT in this category: despite the
  name, `load_battery_state()` is live and imported by `context_builder`.
- **The horizon search is itself a multiple comparison** (5 horizons per
  fold). Currently handled correctly by selecting on train and evaluating
  out-of-sample, which is the right discipline -- noted here only so a
  future reader doesn't mistake it for an oversight.
