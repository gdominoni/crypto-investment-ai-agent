# Open Work

Deliberately-deferred items, with enough detail to pick each up cold.
Ordered by what unblocks the most. Every claim here was measured, not
estimated from intuition -- see `methodology-decisions.md` for the runs.

---

## 1. News/sentiment backfill (GDELT) -- MEASURED AND NOT RECOMMENDED

> **Result, 2026-08-30: do not build this.** The go/no-go test described at the
> end of this section was run (285 conditions, offline, free). A sentiment feed
> of realistic quality is **not separable from a feed containing no information
> at all**:
>
>     rho    meaning                     accepted   vs noise (Fisher)
>     0.30   oracle, not achievable         23/41    p<0.0001  DISTINGUISHABLE
>     0.15   exceptional feed               20/41    p<0.0001  DISTINGUISHABLE
>     0.08   very good feed                  5/41    p=0.216   indistinguishable
>     0.04   realistic news sentiment        3/41    p=0.500   indistinguishable
>     0.00   pure noise (the floor)          2/41    --
>
> Broad news sentiment typically achieves single-digit correlation with
> next-period returns, i.e. the rho=0.04-0.08 rows -- which here produce the
> same number of acceptances as pure noise. The 3-5 days of engineering and
> ~$35 of API spend would buy nothing measurable.
>
> **What to do instead**, in order: raise events per candidate (the levers that
> worked -- real release dates, more release types, lower shock threshold), and
> if sentiment is still wanted, use a NARROW high-signal source rather than
> broad news -- exchange listing/delisting announcements, regulatory filings,
> protocol incidents. Those plausibly sit near rho=0.15, which IS separable.
>
> The plan below is kept in full because it is still the right plan *if* a
> high-signal source is found; only the choice of source changes.

**Original rationale (retained -- the diagnosis was correct, the remedy was not)**

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

**Go/no-go test, run BEFORE any of the above** (`forecast/sentiment_power.py`,
scored by `forecast/analyse_sentiment_power.py`). The plan above is the largest
remaining item in this project, so it should not be started on the assumption
that a sentiment feed would be usable. That assumption is testable without any
news data at all.

The test models sentiment as a CONTINUOUS score present on every day -- which
is what a real feed gives you, mostly low with a right tail -- rather than the
rare binary event an earlier control used:

    score_t = rho * z(forward_return_t) + sqrt(1 - rho^2) * noise_t

so `rho` IS the correlation between the feed and the future return: one
interpretable number for "how informative is this feed". It is swept at
rho = 0.30 (oracle), 0.15 (exceptional), 0.08 (very good), 0.04 (realistic)
and 0.00 (the false-positive floor, which must stay empty), crossed with three
trigger thresholds (>=1.0/1.5/2.0 sigma, giving ~16%/7%/2% of days -- so
sample size is a parameter, not a fixed accident) and with real macro terms in
the state grammar so "sentiment AND a CPI surprise" is expressible.

Published work on news sentiment and next-period returns generally reports
single-digit correlations, so a real GDELT-derived feed plausibly lands near
rho = 0.05. That is why 0.04 and 0.08 bracket the decision:

  * detected at rho <= 0.08 -> a realistic feed clears the bar; build it.
  * detected only at rho >= 0.15 -> only an exceptional feed pays, which
    broad general-news sentiment is unlikely to be. Effort would be better
    spent on sample size, or on a narrower high-signal source (exchange
    listings, regulatory filings, protocol incidents) than on broad news.
  * detected nowhere -> the constraint is the method at these sample sizes,
    not the data, and the backfill cannot rescue it.

This is recorded whatever the project ends up doing with GDELT: the point is
that the decision was made against a measurement rather than an assumption,
and a future reader can re-run the test in about twenty minutes, offline and
free, rather than re-arguing it.

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

- ~~**`daily_range_pct` ignores `scale` entirely.**~~ DONE 2026-08-30 --
  documented as a deliberate no-op alongside `shock_zscore`'s, including
  what would break if it were removed from `DAILY_NATIVE_INDICATORS`.
- ~~**`TRIGGER_NUMERIC_DEFINITIONS` is hand-synced**~~ DONE 2026-08-30 --
  the four thresholds are named once in `definitions.py` and the
  human-facing description is built from them, with tests guarding both
  directions.
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
