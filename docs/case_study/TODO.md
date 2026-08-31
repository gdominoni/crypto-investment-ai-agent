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

## 1b. Haiku-vs-Sonnet substitutability -- BUILT, NOT YET RUN

Deferred until the trigger rework settles: the comparison samples real trigger
days, so re-running it before the triggers are final would measure the old
system.

`forecast/model_comparison.py` + `analyse_model_comparison.py` are written and
tested. Sonnet is taken as the reference; the question is whether Haiku proposes
the SAME conditions, at one third the price (measured $0.0153/call vs $0.0051,
~$18 vs $6 over a full replay).

Two design points worth not re-deriving:

  * Agreement is measured BEHAVIOURALLY, not textually: `behavioural_agreement`
    is the Jaccard overlap of the (coin, day) pairs two specs fire on. Two
    models never emit the same JSON; matching text would score formatting.
  * Every event is judged THREE times -- Sonnet twice, Haiku once. `temperature=0`
    is rejected by the API, so Sonnet disagrees with ITSELF run to run, and that
    self-agreement is the ceiling any second model could reach. Scoring Haiku
    against 1.0 would condemn it for variance the reference model has too.

Sample size is counted in PAIRS, not events: an event yields a comparable pair
only when all three judgments produce a validated condition, and the measured
rate is 55.7% per call (118 usable specs from 212 real calls). Simulated power
against a 0.20 overlap gap: 4 pairs 0%, 6 pairs 63%, 12 pairs 93%. At 4 pairs
the one-sided Wilcoxon cannot reach p<0.05 at all (smallest attainable p is
1/2^4 = 0.0625).

    python3 -m forecast.model_comparison --pairs 6 --max-spend 1.50   # ~$0.38-1.24

**Worth having regardless of the Haiku verdict:** the ceiling itself. If Sonnet
self-agrees only weakly, the proposal step is far less determinate than a single
run suggests -- a fact about the methodology, not about model choice.

---

## 2. Re-run the historical replay end to end

State is at 2023-08-02 and was produced under the pre-audit methodology.
Every verdict has since changed twice (the 2026-08-29 audit, then the
capability work). ~$15-25 in API calls, hours of wall clock.

Do this **after** item 1, not before -- otherwise it gets paid for twice.

---

## 2b. Outcome-triggered discovery (shock / MACD reversal / macro direction change)

**The problem it solves.** Sonnet is currently asked to judge every macro
release. Measured: **76% of those days are followed by no move larger than one
standard deviation** -- three calls in four evaluate a day where nothing
measurable happened. The alternative is to trigger on an OUTCOME and ask what
preceded it.

**Why this is legitimate and not circular.** You cannot trigger on "a pattern
appeared" -- identifying the pattern is the goal. Triggering on a price outcome
and searching backwards conditions the SEARCH on the outcome, but not the TEST:
`pattern_significance` then runs over every occurrence of the proposed
condition, including the ones that preceded nothing. The bias affects which
hypotheses get proposed, which FDR already accounts for.

**Costed, and it is not free** (calibrated at $0.0137/call):

    ATTUALE  every macro release       713 calls    $9.77
    ATTUALE  volatility shock          943         $12.92
    ATTUALE  total                    1656         $22.69

    NEW      volatility shock          943         $12.92
    NEW      MACD reversal            1520         $20.82
    NEW      macro reversal or |z|>=1   381          $5.22
    NEW      total                    2844         $38.96

**+72%.** MACD is the culprit at 169 crossings/year across 7 coins -- nearly one
every other day per coin. It needs a stricter confirmation (histogram above a
threshold, or crossing plus volume) before it is usable. Note the macro filter
CUTS that branch from $9.77 to $5.22: restricting to real surprises works.

**Open design problem, unresolved.** For backwards search Sonnet must be told
something happened -- but telling it "price rose 15%" means the DIRECTION of the
proposed condition is chosen knowing the outcome. The test stays honest (it runs
on all occurrences) but the hypothesis is born contaminated. Showing that a
reversal occurred without its sign, and letting the model propose both
directions, is the obvious candidate fix and is untested.

---

## 2c. Macro REGIME indicators (persistent state, not change)

**The gap.** Every macro indicator in the whitelist says "something changed".
None says "something PERSISTS". A hypothesis like "inflation stays high,
unemployment stays low, rates stay low, and futures go oversold" is currently
inexpressible -- and not for want of data: `dff_rate`, `fed_funds_rate`,
`industrial_production`, `unemployment_rate`, `us02y_yield`, `us10y_yield` are
all downloaded and none is exposed as a testable indicator.

**The criticality that decides feasibility, measured.** Regime variables are
persistent by construction, so their occurrences clump into single periods and
run straight into the 60% year-concentration gate:

    regime at the median          dominant year 23%   passes
    regime at the 75th pct        dominant year 46%   passes, barely
    regime at the 90th pct        dominant year 90%   BLOCKED

Inflation was high in 2021-2024 and almost nowhere else. **The intersection of
three regimes is a single historical period**, which is the worst case: exactly
the scenario that motivates the feature is the one the concentration check
refuses. Only broad regimes are testable, or the year check needs the same
declared-exemption treatment `coins` already has for single-coin hypotheses.

**The strategic objection, which applies to both items above.** The grammar
sweep found 0 of 296 with the current vocabulary. Adding regime indicators
multiplies the expressible hypothesis space, and under Benjamini-Hochberg more
hypotheses means a stricter threshold for every one of them. Statistical power
is this project's binding constraint; widening the search worsens it. Test
regimes offline in `forecast/grammar_sweep.py` FIRST -- if they produce nothing
there, the question is answered for free and no replay is needed.

---

## 2d. Macro direction-change trigger -- MEASURED AND NOT RECOMMENDED

> **Result, 2026-08-31: the data cannot support this, and what it does show
> does not favour it.**

**The proposal.** Trigger Sonnet not on every macro release, nor on the size of
a surprise, but when a macro series CHANGES DIRECTION -- inflation stops falling
and starts rising -- and analyse the medium/long-term trend that follows,
whether or not a shock accompanies it. Directionally a good instinct: a regime
turn is a real economic event in a way that a scheduled publication is not.

**Why the earlier trigger measurement did not answer it.** `trigger_value.py`
tested horizons of 1-14 days. A medium/long-term thesis is not measurable there,
so that null says nothing about this proposal. Re-measured at 21-180 days, with
direction defined as a sign flip in each series' ~3-month trend of first prints:

    inversion of        coin-days   p@21   p@30   p@60   p@90  p@120  p@180
    CPI + Fed Funds           102     --     --     --     --     --     --
    Jobless Claims            503  0.375  0.466  0.229  0.052  0.070  0.851
    all series                605  0.319  0.326  0.139  0.051  0.061  0.707

**Three problems, jointly fatal.**

1. **The series that matter almost never turn.** CPI changed direction **4 times
   in seven years**; Fed Funds 11. Fifteen events together -- too few to test at
   all (the `--`). Every bit of the 0.051 comes from **jobless claims**, the
   noisy weekly series, not from the macro regime turns the proposal is about.

2. **The observations are not independent, and correcting for it is brutal.**
   At a 90-day horizon the 88 event dates collapse to **23 non-overlapping
   episodes**; at 180 days, to 12. The test treats 605 coin-days as independent,
   and the seven coins are strongly correlated with each other on top of that.
   A p of 0.051 computed on a sample that size is not a p of 0.051.

3. **The gradient collapses.** 0.051 at 90 days, 0.707 at 180. An isolated peak
   across six horizons tested is the shape of noise, not of an effect.

**The underlying reason is not fixable by engineering.** Macro regimes turn a
handful of times per decade. This is the same wall item 2c already measured for
regime INDICATORS -- "the intersection of three regimes is a single historical
period". No better trigger design gets around a sample size set by economic
history; it would take decades more data, or different markets.

**A separate, real finding surfaced by this.** `ConditionSpec.horizons` defaults
to `(1, 3, 7, 14, 21)` -- **nothing beyond 21 days can be tested at all**. Even
had the trigger worked, the medium/long-term hypothesis it exists to serve is
currently inexpressible downstream. Extending the horizons is possible but makes
problem 2 worse, since longer windows mean fewer independent episodes from the
same history. Recorded here so a future reader does not mistake the ceiling for
an oversight.

**What remains standing:** the volatility-shock trigger, the only one measured to
select days that differ from ordinary days (p=0.000 at 1 day, 0.009 at 3).

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
