# Crypto Pattern-Discovery Agent: Can Market Sentiment and Real-Time Conditions Reveal Statistically Real, Repeatable Patterns in Crypto Prices?

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![Claude](https://img.shields.io/badge/Claude-Haiku%20%2B%20Sonnet-6B4FBB) ![Telegram](https://img.shields.io/badge/Telegram-human--in--the--loop-26A5E4) ![Freqtrade](https://img.shields.io/badge/Freqtrade-hyperopt%20cross--check-orange) ![Status](https://img.shields.io/badge/Status-Case%20Study-blue)

## What This Project Tests

Whether an LLM-driven architecture — reading real market news and recognizing specific, recurring market conditions as they happen — can identify genuine, statistically real patterns in crypto prices. Not a paper-trading guess: every candidate pattern is tested directly against that coin's own historical baseline with a real significance test, and once identified, tracked through real, dated occurrences going forward. In plain terms: after a specific kind of announcement (say, a Federal Reserve rate decision), under a specific kind of market condition (say, Bitcoin's price swinging unusually wide that day and closing lower) — is there a clear, repeatable, *statistically significant* reaction in that coin's price over the following days, one an AI system could actually recognize and track as it happens?

**This project never opens a funded position.** It is a knowledge-discovery investigation of patterns, not an investment strategy — every "trade" described below is an observational live test: a real, dated occurrence of a tracked condition, held for a fixed horizon, resolved by measuring what actually happened next. No capital is ever at risk, at any point, in any phase. Everything below (the historical methodology, the statistical significance test, the Haiku/Sonnet judgment layer, the human gate on testing genuinely new patterns) exists to answer the question above honestly and live, rather than assume the answer from a backtest.

## Executive Summary

A prior, static rule-based research phase (Phase 1 below) tested six categories of deterministic market triggers — scheduled macro releases, futures-market crowding, trend-efficiency continuation — across seven-plus years of crypto data, under full walk-forward validation, and found no fully persistent, unconditional edge. This project's response to that finding is the adaptive system described below: a continuously re-validated statistical baseline, a bootstrap significance test that asks whether a pattern is *real* (not merely profitable-looking in one backtest), an LLM judgment layer (Claude Haiku → Claude Sonnet) that discovers and proposes genuinely new conditions, and a human supervisor at exactly one decision point — observed live rather than assumed from a backtest.

### The System, Live on Telegram — Real Screenshots

Every message below is a real, unedited screenshot from this project's own Telegram bot — nothing staged, nothing mocked up for this README. They show the interface as it actually behaves; the *verdicts* visible in them predate the [2026-08-29 statistical audit](docs/case_study/methodology-decisions.md) and no longer hold — see Phase 3 for the current, corrected battery.

<table>
<tr>
<td width="33%" align="center"><img src="docs/case_study/assets/telegram_novel_condition_proposal.png" alt="Sonnet proposing a novel condition, with Test It / Don't Test It buttons" width="280"><br><sub>Sonnet proposes a novel condition — human approves with a button, never free text</sub></td>
<td width="33%" align="center"><img src="docs/case_study/assets/telegram_live_test_resolved.png" alt="A live test resolved, with real forward return, best and worst point reached" width="280"><br><sub>A live test resolves — real forward return, best/worst point reached, no TP/SL</sub></td>
<td width="33%" align="center"><img src="docs/case_study/assets/telegram_prune_decision.png" alt="A keep-or-drop decision after 2+ years untested, with Sonnet's advisory opinion" width="280"><br><sub>2+ years untested — Sonnet's advisory opinion, human decides Keep or Drop</sub></td>
</tr>
<tr>
<td width="33%" align="center"><img src="docs/case_study/assets/telegram_replay_summary.png" alt="/replay_summary grouping every tracked candidate by status" width="280"><br><sub><code>/replay_summary</code> — every tracked candidate, grouped by status, recomputed fresh <i>(verdicts shown are pre-audit)</i></sub></td>
<td width="33%" align="center"><img src="docs/case_study/assets/telegram_replay_details.png" alt="/replay_details showing the full numeric breakdown for one candidate" width="280"><br><sub><code>/replay_details</code> — every number behind one candidate's verdict <i>(pre-audit figures)</i></sub></td>
<td width="33%" align="center"><img src="docs/case_study/assets/telegram_help_pinned.png" alt="The pinned /help reference listing every standard command" width="280"><br><sub>The pinned <code>/help</code> reference — every command, always one scroll away</sub></td>
</tr>
</table>

## Target Objectives

1. Determine whether adaptive re-validation and LLM-mediated discovery surface real, statistically significant patterns the static baseline missed.
2. Do so with a fully causality-safe, leakage-free methodology throughout, and a significance test independent of any specific trade structure.
3. Keep every classification traceable to a specific, real number — on demand, in plain language, never invented.

The human gate sits on exactly one kind of decision: whether a genuinely new, LLM-proposed condition is worth testing at all. Once a condition is being tracked — whether from the original static battery or a human-approved test — everything downstream (the statistical classification, whether it opens a live test, whether it earns "validated" status) is fully deterministic and code-driven, with no further human input needed. Gating every individual classification on a human would test human-plus-LLM judgment, not the LLM's own, defeating the point of objective 1.

Every non-obvious methodology or design decision — why 50, why 60%, why a fixed horizon instead of a TP/SL ladder, why no funded position is ever opened — is logged with its own stated reasoning in [`docs/case_study/methodology-decisions.md`](docs/case_study/methodology-decisions.md). File-by-file guide to the whole codebase: [`PROJECT_MAP.md`](PROJECT_MAP.md). Want to run this yourself, step by step, no prior knowledge of the code assumed? See [`HOW_TO_RUN.md`](HOW_TO_RUN.md).

---

## Phase 1: Historical Research & Statistical Methodology

Before any live component was built, this project ran a systematic historical study of whether deterministic market triggers predict real, repeatable price moves in crypto — the foundation everything downstream is built on, and honest about what it did and didn't find.

**What was tested.** A battery of trigger categories tied to scheduled financial releases and market-structure signals — FOMC / CPI / initial-jobless-claims macro-event reactions, futures/perpetual funding-rate crowding, and Kaufman Efficiency-Ratio trend continuation — across BTC, ETH, BNB, XRP, DOGE, ADA, and LTC, using daily and hourly data spanning 2017 through the present, kept current by a live data refresh rather than a frozen snapshot.

**Why the methodology is built the way it is** — six requirements, each chosen for a specific, verifiable reason rather than as boilerplate rigor. Click any one for the reasoning behind it:

<details>
<summary><b>A pattern must be statistically real, not just profitable-looking, to be accepted.</b></summary>

The actual acceptance gate is a bootstrap significance test (`candidates/methodology.py::pattern_significance`, 2,000 resamples): it compares a trigger's forward returns against that *same coin's own* unconditional baseline over the identical calendar stretch, at whichever holding horizon (1, 3, 7, 14, or 21 days) a walk-forward search finds most reliable on training data alone. **The test is directional and pre-specified** — the horizon is chosen by *signed* mean return and the p-value is a fixed upper tail in the direction the candidate actually trades, so a pattern that is real but points the *opposite* way can never pass (it previously could; see the audit entry in the decisions log). Because the baseline's overlapping windows are serially dependent, the resampling is a **moving-block** bootstrap rather than i.i.d. — calibrated against a true-null harness, where the i.i.d. version rejected 43% of the time at a nominal 5%. Win rate, strict win rate, and a TP/SL-conditioned Sortino ratio are still computed and shown, purely for reference — they describe how well a barrier-based trade structure would have captured the pattern historically, but they no longer decide accepted/watch/rejected. This project's purpose is finding real patterns, not fitting the most flattering trade structure around noise. A candidate also needs a favorable risk path during the hold (the average best-case move exceeding the average worst-case move) and no single coin or calendar year carrying more than 60% of the positive result — a real, general pattern, not a fluke of one period or one coin.

</details>

<details>
<summary><b>Strict causality lag, enforced structurally, not by convention.</b></summary>

Several trigger definitions are only knowable once the period they describe is fully over — "the day closes bearish" cannot be evaluated before the day closes. Entry is therefore always the *next* period's open following the period a trigger condition reads, with no code path that allows an earlier fill. This matters because the alternative — filling at a price from before that information existed — silently inflates a backtest with information no live system could ever have had.

</details>

<details>
<summary><b>Purged walk-forward validation with mandatory concentration checks.</b></summary>

Every candidate's out-of-sample result — and the significance test above — is checked for concentration: is the pooled result actually carried by one coin or one calendar year? A result that doesn't diversify across both is held at `watch`, regardless of how strong its headline number looks. A single strong year or a single strong coin is not evidence of a general, repeatable pattern.

</details>

<details>
<summary><b>Win rate is always reported with a strict counterpart and a Sortino ratio, never alone — but never as the acceptance gate either.</b></summary>

A headline win rate computed only on decisively-resolved trades can look strong while a large population of inconclusive, loss-leaning timeouts sits outside that denominator. Reporting the strict version (timeouts counted as non-wins) alongside Sortino removes that blind spot for anyone reading the reference numbers — precision that matters for the informational TP/SL comparison, even though none of these three numbers gates acceptance anymore.

</details>

<details>
<summary><b>Extreme historical shocks are statistically isolated from the static battery's fitting, not blended in.</b></summary>

A coin's short-term realized volatility is z-scored against its own longer trailing distribution; events where that z-score crosses an extreme threshold are excluded from the pattern test the static battery runs, so a handful of crash days can't distort the read on ordinary conditions. These excluded events aren't discarded — they're the population the live shock-detection pathway (Phase 2) is built and tested on instead.

</details>

<details>
<summary><b>"Accepted" and "validated" are two different, deliberately non-interchangeable claims.</b></summary>

`accepted` means a candidate has cleared the statistical bar above — a real pattern, right now, in the historical record. `validated` is earned separately and later: a candidate that has actually lived through 50 occurrences of the kind that count toward its own origin, while still `accepted` at that checkpoint (see Phase 3 for exactly what counts, and why it differs for a static vs. an LLM-discovered candidate). The first is a backtest's verdict; the second is a live track record. This project never lets the first stand in for the second.

</details>

<h3 align="center">🔍 Technical & Quantitative Methodology Details</h3>

<details>
<summary><b>Fees, Concentration Thresholds & Shock Handling</b></summary>

**Reference trading fees.** The informational TP/SL comparison line shown alongside every verdict nets a fixed 0.20% round-trip transaction cost — chosen to describe *directional* edge honestly if it were traded with a barrier structure, not to model exact execution cost of a real order that's never actually placed.

**Why 60%.** The concentration threshold (no single coin or year may carry more than 60% of a candidate's positive return) isn't arbitrary — it's set to catch the specific failure mode that invalidated several nominally-positive candidates in this project's own prior research: one coin, or one calendar year, dressed up as a general result. The full reasoning, along with every other numeric threshold in this project, is logged in [`docs/case_study/methodology-decisions.md`](docs/case_study/methodology-decisions.md).

**Variance non-stationary shift.** Realized volatility in crypto is not stationary — a handful of historic shocks (a 2018-style crash, a 2020-style crash, a 2022-style unwind) can dominate a naive average and produce a distorted read on ordinary conditions. This project's fix: a rolling z-score of short-term realized volatility against its own trailing baseline (`candidates/methodology.py::shock_zscore_series`, causally safe by construction — every value uses only bars up to and including itself) flags events crossing an extreme threshold (empirically ~2% of days, not the ~0.13% a normal distribution would predict — crypto's return distribution is heavily right-skewed) as `shock`-regime; the static battery's pattern test uses `normal`-regime events only.

**Live shock/crash recognition.** Real-time market data is scanned for the same statistical shock signature Phase 1 uses to exclude extreme events (`llm_pipeline/shock_detector.py`) — when a coin's current short-term volatility crosses that same extreme threshold, Sonnet is escalated directly, independent of any news headline. Every scan refreshes local OHLCV from Binance first, rather than reading a static snapshot. Sonnet's only available response to a confirmed shock is to propose testing it — using the same whitelisted-indicator, walk-forward, concentration-checked pipeline as any other novel condition, run specifically on that coin's own historical shock-regime events (the ones excluded from the static battery above). If a human approves and it validates statistically, the resulting condition is tracked going forward and its own triggering occurrence becomes a live test — a direct, honest test of whether the LLM layer can recognize and react to a crash or a surge as it's actually happening, not just process routine headlines.

</details>

<h3 align="center">The honest finding.</h3>

Applying this methodology across the full candidate battery, static deterministic rule sets did not produce a statistically persistent, cross-coin, cross-year edge on their own. That's the pessimistic baseline this project's live architecture is built to test against — not a result to argue away, and not one this project re-litigates by re-running the same static candidates hoping for a different answer.

**And the adaptive layer has not overturned it.** As of the 2026-08-29 audit, 98 candidates have been tested against real history — 92 of them proposed by Sonnet, not hand-picked — and **none currently clears the bar** (Phase 3). An earlier version of this README reported two accepted and one validated; those were artifacts of a significance test that ignored direction and used a bootstrap rejecting at 43% under a true null. Correcting the statistics removed the finding rather than the other way round, which is the outcome this section always said it was prepared for.

<h3 align="center">The Dynamic Agent Thesis.</h3>

Given that a fixed rule set doesn't hold up on its own, this project's central bet is architectural rather than statistical: a system that (1) treats every historical finding as perishable, re-validating the full candidate battery on a weekly cadence against live data rather than fitting once and trusting it indefinitely; (2) escalates genuinely novel market conditions — by definition, the ones a fixed rule set cannot anticipate — to an LLM judgment layer and a human decision-maker, instead of silently misclassifying them; and (3) requires a real, tracked live occurrence, not just a favorable backtest, before ever calling a pattern "validated." This doesn't guarantee a different live outcome than the static study found — it's a mechanistically different hypothesis (adaptive discovery plus continuous re-validation, versus one rule set fit once), and this project exists to test it for real rather than assume it inherits the static result.

---

## Phase 2: System Architecture

Two cooperating components, run continuously by a single process (`scheduler/live_daemon.py` — see Phase 3):

| Module | Role |
|---|---|
| **B — Live Test Engine** | Opens and resolves observational live tests the instant a tracked trigger's condition fires (mechanical, hourly detection, no LLM involved, run hourly by `scheduler/live_daemon.py`). Never a funded position, never a TP/SL ladder — holds for the horizon `pattern_significance` found significant, resolves by measuring the real forward return, best-case, and worst-case excursion. A separate, purely informational cross-check does simulate a TP/SL-bounded position for comparison — see below. (`execution/live_testing.py`) |
| **C — Signal Layer** | The candidate battery (Phase 3) plus the Haiku Scout → Sonnet Strategist judgment pipeline. Cadence: hourly for headline/shock scanning, weekly for the full battery refresh — both run on that schedule by `scheduler/live_daemon.py`. (`candidates/`, `llm_pipeline/`) |

**A separate, purely informational cross-check does use a TP/SL-bounded position — never the live tests above.** A periodic, local-only Freqtrade hyperopt run re-derives TP/SL multipliers for each tracked candidate by backtesting a simulated trade that *is* bounded by a take-profit/stop-loss ladder, using a genuinely different search method (Bayesian optimization) and a different, industry-standard backtesting engine. Shown alongside the reference numbers for context only — it never gates `accepted`/`watch`/`rejected`, never opens or closes anything live, and the live test engine above continues to hold for a fixed horizon regardless of what it finds. Full detail in Phase 3.

<h3 align="center">Haiku Scout → Sonnet Strategist</h3>

The adaptive layer at the center of the Dynamic Agent Thesis — narrowed, deliberately, to exactly two jobs:

- **Haiku (runs continuously, cheap):** reads news, extracts asset/sentiment/magnitude/event type, and screens out routine, already-classified noise — run hourly by `scheduler/live_daemon.py`. Only genuinely escalation-worthy headlines reach Sonnet — Sonnet's attention is expensive and reserved for real judgment calls (`llm_pipeline/haiku_sonnet_pipeline.py`).
- **Sonnet (escalated, judgment):** given an escalation, a real indicator snapshot for the relevant coin(s), the last 10 days of real macro releases, and the current candidate battery status — never invented, never a hand-maintained file — does exactly one of two things: **proposes testing a genuinely new condition**, or **does nothing** (routine, already-covered territory). That's the entire decision space. Sonnet never opens a live test itself, never decides accepted/watch/rejected, and never answers a direct human question without the real numbers behind it (§4).
- **A proposal needs a human "yes" before anything is tested — via buttons, not free text.** Every proposal Sonnet makes arrives with two buttons: **Test It** / **Don't Test It**. This is deliberate: a fixed, small set of valid answers is *always* presented as buttons in this project, never left to free-text matching that can silently do nothing for a natural phrasing that doesn't hit an exact string. Pressing **Test It** runs the real methodology engine directly (`test_novel_condition` — the same walk-forward, significance-tested, concentration-checked pipeline the static battery uses) — Sonnet's job ends at proposing the spec; the actual classification is deterministic and Sonnet has no further say in the outcome.
- **Novel-condition proposals are built as a whitelist, never as code execution.** A proposal can only compose from a fixed registry of fifteen indicators (`llm_pipeline/novel_condition_tester.py::SUPPORTED_INDICATORS` — RSI, ATR%, Bollinger %B, Donchian position, funding/volume z-scores, trend efficiency, the shock z-score, and three point-in-time-correct macro surprises: CPI, Fed funds rate, jobless claims) plus a comparison, a threshold, and an optional `within_days` lookback that makes an ORDERED hypothesis expressible ("crash, THEN news" is a different claim from "crash and news the same day"), rendered back to a human in plain English (e.g. *"14-day RSI below 30 AND shock z-score above 3.0"*, never a raw variable name) — chosen specifically so there is no path from an LLM proposal to arbitrary executed code. **Every proposal must contain a news/macro event clause — a necessary condition, enforced in code, not requested in a prompt.** A condition built only from price/volume/funding indicators is a chart pattern and does not test this project's question; `ConditionSpec` rejects it before it can ever be tested, and a label may not name a macro event its clauses lack. A market shock does not satisfy this on its own — a violent price move is a market event, not news. Tested or not, the condition is written to a persistent registry re-tested every week alongside the static battery, so nothing is trusted forever from one test, and a rejected condition isn't silently re-proposed without genuinely new evidence.
- **A newly-discovered condition's own triggering occurrence can be honestly backdated.** Because no funded position is ever placed, there's nothing stopping a real, retroactive read of when a just-discovered condition first became true (`execution/live_testing.py::find_backdated_entry` scans the last 14 days of already-recorded hourly history) — anchoring the live test to the real hour it happened, not the moment a human happened to approve it. This would never be legitimate for a real funded order; it's exactly the honest thing to do for an observational record.
- **Real-time shock/crash detection is a distinct escalation path, market-data-driven rather than news-driven.** `llm_pipeline/shock_detector.py` scans live volatility for the same statistical extreme Phase 1 excludes from the static battery, and escalates directly to Sonnet when found — full detail in Phase 1's collapsible technical section above.

<p align="center">
  <img src="docs/case_study/assets/architecture_diagram.svg" alt="System architecture: statistical baseline feeding and checked against the adaptive Haiku/Sonnet layer, both driving live testing and Telegram" width="900">
</p>

---

## Phase 3: Continuous Weekly Re-Validation & Candidate Battery Status

Every candidate signal carries a live status, re-derived — not assumed — on a fixed schedule:

| Status | Meaning |
|---|---|
| `accepted` | Clears the statistical significance test, the risk-path check, and the concentration check. Its own trigger opens a live test automatically the moment it next fires. |
| `watch` | A real pattern signal that fails a robustness check (concentration, or an unfavorable risk path), or too little data yet for the significance test itself to say either way. |
| `rejected` | No statistically significant pattern found, or too small a sample even to test. Logged so it isn't silently re-proposed without new evidence. |
| `insufficient_data` | Fewer historical occurrences than the minimum needed to run the test at all. |

**`validated` is a separate, later claim — earned only by living occurrences, not by a backtest alone. What counts toward the 50 depends on how the candidate was found.** For the static battery (C1/C2/C6, derived by directly mining this project's own historical data — a real look-then-test risk), only real, resolved live-test occurrences count: a candidate crosses the bar the first time it survives 50 of them while still `accepted` at that checkpoint. For an LLM-discovered condition (§2), Sonnet never sees this project's own backtest results before proposing one — only a live market snapshot — so the contamination risk is far weaker and diffuse rather than direct; these use a rolling window of the most recent 50 occurrences, backtest and live mixed, which only ever accelerates the *first* checkpoint (topping up thin early live evidence with the freshest backtest occurrences) — every checkpoint after that, and every dynamic candidate that's accumulated 50 real live occurrences on its own, follows the exact same live-only rule the static battery always has. Either way, the label is re-earned (or lost) fresh at every following multiple of 50, never a one-time badge — full reasoning in [`docs/case_study/methodology-decisions.md`](docs/case_study/methodology-decisions.md). A separate, purely calendar-based safety net (2+ years tracked, never once accepted) catches a trigger too rare to ever reach even its first checkpoint on its own, offering the same keep-or-drop decision to a human either way.

### Current Candidate Battery Status

`candidates/run_battery.py`, 7 coins, daily bars, walk-forward-selected horizon, bootstrap significance test against each coin's own baseline — recomputed fresh, not a frozen snapshot:

| Candidate | Direction | Status | N | p-value | Excess return | Risk Path (MFE/MAE) |
|---|---|---|---|---|---|---|
| c1 (funding crowding) | long | watch | 325 | 0.783 | −1.79% | 0.76 |
| c1 (funding crowding) | short | watch | 166 | 0.886 | −0.26% | 0.88 |
| c2 (post-macro reaction) | long | watch | 62 | 0.255 | +2.50% | 2.59 |
| c2 (post-macro reaction) | short | watch | 86 | 0.864 | −1.50% | 1.00 |
| c6 (efficiency trend) | long | rejected | 289 | 0.075 | +9.49% | 2.53 |
| c6 (efficiency trend) | short | watch | 184 | 0.764 | −3.36% | 0.89 |

**On the acceptance thresholds, and how they were set.** Every gate in this system was audited on 2026-08-30 against a *positive control*: synthetic signals planted by deliberate lookahead on days that genuinely are followed by strong returns, plus a pure-noise arm that must stay silent. Ground truth makes the only useful question answerable directly — when a signal really is there, which gate kills it? The answer was unambiguous: of 294 known-good conditions with a valid test and adequate data, **92% died at the significance threshold**; the concentration check killed four and the MFE/MAE gate killed none. That redirected the tuning entirely — the gates that *felt* strict were not the problem, statistical power was. Three changes followed, each with its measurement in [`methodology-decisions.md`](docs/case_study/methodology-decisions.md): the per-test significance level moved to 0.10 (detection of real effects 8.3% → 27.4%, false positives 0.0% at both levels), the minimum-events gate from 50 to 20, and a genuine bug was found in horizon selection — it scored raw mean forward return, which grows with horizon from pure market drift, so it was systematically picking the longest horizon on offer rather than the one where the effect lived.

**Nothing is currently `accepted`.** These numbers replace an earlier version of this table that showed all six candidates as significant (p < 0.05) and two as `accepted`. A [statistical audit on 2026-08-29](docs/case_study/methodology-decisions.md) found the significance test was measuring the wrong thing in two compounding ways, both since fixed:

- **It ignored direction.** The holding horizon was chosen by the strongest effect *in either direction*, and the p-value's tail was picked after seeing which way the effect went. Four of six candidates were "significant" with a **negative** excess return — a real pattern, pointing the opposite way from the direction the candidate actually trades. The gate never checked the sign.
- **Its bootstrap was miscalibrated.** Overlapping forward-return windows were resampled independently, which understates the null's variance. Measured under a true null, where there is nothing to find, the old test rejected **43% of the time** against a nominal 5%. That alone explains six of six looking significant.

The test is now directional and uses a block bootstrap calibrated to an 8.7% false-positive rate (stated, not rounded to 5%). Under it, four candidates are held at `watch` on concentration, and the strongest survivor — c6_long, +9.49% excess return — misses significance at p=0.075. That is the honest reading, and it is consistent with Phase 1's own finding rather than an exception to it.

### The Re-Validation Loop That Keeps This Table Alive

`scheduler/weekly_revalidation.py` first refreshes local OHLCV/funding data from Binance, then re-runs the full battery's significance test and concentration check against that current data, diffs every candidate's status against the previous run, and notifies a human only when something actually changed — run automatically once a week by `scheduler/live_daemon.py`, no external cron job needed. Separately, any time Haiku/Sonnet flag a genuinely novel condition, a human can approve testing it on the spot instead of waiting for the next re-validation run — if it clears the bar, it joins the same weekly regime from then on. One candidate's own bad data or bug can't take the rest of the battery down with it — each is isolated and retried the following run — and any failure that does reach the top level sends a Telegram alert rather than failing silently.

**A second, fully independent opinion — never load-bearing.** A periodic, local-only Freqtrade hyperopt run re-derives TP/SL multipliers for each tracked candidate using a genuinely different search method (Bayesian optimization over a continuous space vs. this project's own grid search) and a different, industry-standard backtesting engine — shown alongside the reference numbers above, purely informational, never gating acceptance and never touching live execution. Deliberately kept off any live host (see [`PROJECT_MAP.md`](PROJECT_MAP.md)'s "Cost Optimization" section) — run it yourself with `python3 -m execution.hyperopt_runner`.

### Historical Replay: Results So Far

The table above is production's — real, but young, with no live tests resolved yet. The historical replay (`replay/`, see [`HOW_TO_RUN.md`](HOW_TO_RUN.md) to run it yourself) exists to answer this project's own question against years of real history rather than waiting on real time to pass.

**The replay has been reset and is being re-run from 2017.** The previous run's results are withdrawn, for a reason worth stating plainly: it tested 92 Sonnet-proposed conditions, and **80 of them contained no news or macro event at all** — 31 were pure chart patterns, 49 carried only a market shock. The code never enforced the event term that this project's own question requires, so the large majority of what it measured did not answer that question. Fourteen labels made it worse by *naming* a macro event their clauses never contained.

An event clause is now a necessary condition, enforced in `ConditionSpec` rather than requested in a prompt. The re-run is therefore the first one that tests the stated hypothesis at all.

What the withdrawn run did establish, and which still holds, is a **null result under corrected statistics**: re-graded against the full 2017–2026 history with a directional test, a calibrated block bootstrap and Benjamini–Hochberg FDR, **0 of 98 candidates cleared every gate**, and 21 of 47 testable candidates had a positive excess return — 45%, a coin flip. That is the honest baseline the new run starts from.

## Phase 4: Interactive Telegram Interface

Two structurally separate interaction modes, kept apart deliberately: **free-text conversation never generates a financial number itself** — every figure a message cites is pulled from a real computation, never invented by the language model. **Structured commands and buttons never touch the language model at all** — a fixed, small set of valid answers is always presented as buttons, never left to free-text guessing.

The conversations below are illustrative mockups of the real message format this project's bot produces — the specific figures shown are examples, not a claim about the current battery's exact state, which is reported plainly in Phase 3 and on demand via `/summary`.

### 4.1 A new condition, proposed and approved

```
🤖 Agent: Sonnet Strategist Alert

          Headline: Exchange X halts withdrawals amid liquidity concerns
          Asset: BTC | Magnitude: 4/5

          Assessment: This looks like a genuine liquidity shock, not
          routine noise.

          This needs your input. This looks like a condition we haven't
          tested before.

          Proposed test: "rsi_shock_combo"
          (14-day RSI (momentum, 0-100 scale) below 30.0 AND shock
          z-score (how extreme today's price move is vs. this coin's
          own history) above 3.0 → long)

          Test It runs a real walk-forward backtest of this condition
          before it's tracked as a live test (no real money is ever
          placed on it). Don't Test It dismisses this proposal.

          [ Test It ]  [ Don't Test It ]

You:     [taps "Test It"]

🤖 Agent: Historical backtest -- rsi_shock_combo

          (14-day RSI (momentum, 0-100 scale) below 30.0 AND shock
          z-score above 3.0 → long)

          Pattern signal: excess return +2.10% vs. this coin's own
          baseline over the same period, p=0.031 (significant at the
          5% level, N=62).
          Risk profile: MFE/MAE ratio=2.05 (favorable -- reward tends
          to exceed the risk taken to get there).

          Verdict: ACCEPTED

          This is a historical screening result, not a live track
          record -- the real test is ongoing: added to the battery now,
          re-tested every Sunday alongside the static candidates.
```

### 4.2 A live test, opened and resolved

```
🤖 Agent: 2026-08-27

          Live test opened -- LONG BTCUSDT

          (candidate c1_long: funding rate persistently and heavily
          negative, a classic crowded-short setup)

          Held for 7d, no TP/SL.
```

```
🤖 Agent: 2026-09-03

          Live test resolved -- LONG BTCUSDT

          (candidate c1_long: funding rate persistently and heavily
          negative, a classic crowded-short setup, held 7d, opened
          2026-08-27)

          Forward return: +3.10%
          Best point reached: +4.20%
          Worst point reached: -1.80%
```

**A VALIDATED candidate's own aggregate is, by design, resistant to a short losing streak — a separate, fast alert covers the gap that creates.** Measured directly: a well-established candidate can absorb 20-30 consecutive worst-case losses before its own significance test would ever flip status — correct behavior against noise, but too slow to flag a genuine regime change (a market-structure shift, a new rule, an arbitraged-away inefficiency) on its own. So immediately after any live test resolves, if a VALIDATED candidate's last 2+ resolved live tests came back negative in a row, it fires a purely informational alert — the last several occurrences with their return/MFE/MAE, never a status change — long before the aggregate statistics would ever say anything.

### 4.3 Natural-language conversation

```
You:     How is the market going today?

🤖 Agent: BTC +1.8% (24h), ETH +2.1%, broad risk-on tone. No macro
          releases in the last 10 days. Nothing currently accepted;
          5 on watch for concentration or risk-path reasons, 1
          rejected on significance. No live tests open.
```

### 4.4 Structured commands — no LLM involved

`/summary`, `/details`, and `/help` never touch the language model — a command recomputes the real battery fresh (or, for `/help`, prints a static reference) and renders it directly.

```
You:     /help

🤖 Agent: [pinned] Standard commands, and the local-only hyperopt
          cross-check CLI reference (see Phase 3).

You:     /summary

🤖 Agent: Still under test

          Watch (4)
          a real pattern signal, but fails a robustness check
          (concentration or an unfavorable risk profile), or too
          little data for the risk check yet
            c1_long -- N=325, p=0.010, MFE/MAE=0.74
            c1_short -- N=166, p=0.005, MFE/MAE=0.38
            ...

          [second message] Already discarded

          Nothing here right now.

You:     /details c1_long

🤖 Agent: c1_long
          What triggers it: 30-day funding-rate z-score below -2.0 --
          extreme relative to that coin's own trailing 30-day funding
          history, not a fixed absolute rate.
          Status: watch -- a real pattern signal, but fails a
          robustness check (direction: long)
          Held for: 21d (empirically-derived, re-checked weekly)

          • Historical occurrences (N): 325
          • Statistical significance: significant (p=0.010), excess
            return vs. this coin's own baseline: -1.67%
          • Risk path (mean favorable / mean adverse excursion): 0.74
            (favorable if > 1.0)
          • Coin concentration: 64% of total positive return comes
            from a single coin (LTCUSDT) -- flagged above 60%
          • Year concentration: 100% of total positive return comes
            from a single year (2026) -- flagged above 60%

          Why not accepted: a statistically significant pattern, but
          100% of it comes from a single year -- too concentrated to
          trust as general.
```

`/summary` is deliberately terse — a status line like "Watch, N=325, p=0.010" answers "what's the verdict" but not "what does 'elevated concentration' actually mean in numbers." `/details <name>` (and `/replay_details <name>` for the historical replay) exists for exactly that: the trigger's own exact numeric definition, plus every number behind its current classification, one candidate at a time — kept as a separate command rather than folded into every `/summary` line or every Sonnet proposal, which would make either too long to scan at a glance.

---

## Repository Structure & Build Plan

```
crypto-sentiment-trading-agent/
├── candidates/                  # Phase 1's methodology + battery: methodology.py, definitions.py, run_battery.py
├── execution/                   # Live test engine (live_testing.py), local-only hyperopt cross-check (hyperopt_runner.py)
├── llm_pipeline/                # Haiku Scout, Sonnet Strategist, live context builder, novel-condition tester, shock_detector.py
├── telegram/                    # Both interaction modes from Phase 4
├── scheduler/                   # live_daemon.py (the one command that runs everything), weekly_revalidation.py
├── data_ingestion/               # market_data/binance_fetcher.py (keeps data/ current, from-scratch backfill capable)
├── data/                        # Historical + periodically-refreshed market/macro data
├── replay/                      # Historical walk-forward simulation used to validate the system and build an initial live-test track record before going live -- see PROJECT_MAP.md
├── docs/case_study/             # methodology-decisions.md, this project's build log
├── .github/workflows/           # tests.yml -- the tests below run automatically on every push
└── tests/                       # candidates/methodology.py, status_history.py, novel_condition_tester.py, run_battery.py
```

**Running this live is one command:** `python3 -m scheduler.live_daemon`. This project is built to be operated as an agent, not maintained as infrastructure — one process owns the Telegram bot, the hourly scans, and the weekly re-validation, so there's no separate cron job to configure or forget. It picks up right where it left off after a restart.

**Safety & guardrails:** this project never opens a funded position, at any point, in any phase — the single guarantee everything else is built around, not a configuration flag that could be toggled off. Every "trade" described above is an observational live test: a real, dated occurrence tracked and measured, never capital at risk. The human gate sits on exactly one decision — whether a genuinely new, LLM-proposed condition is worth testing at all (§2, §4.1); everything downstream is deterministic and code-driven, with no further human input needed.

**Observation & reporting:** the candidate battery re-validates weekly against live data; a live test's outcome, once resolved, is never retroactively re-tuned. `/summary` and the full decision log are the report at any point — including if the honest result is "no better than the static baseline Phase 1 already found."

File-by-file guide to what every module does, including cost-optimization details: [`PROJECT_MAP.md`](PROJECT_MAP.md). Every non-obvious methodology or design decision, with its own reasoning: [`docs/case_study/methodology-decisions.md`](docs/case_study/methodology-decisions.md).

---

## About the Author

**Giovanni Dominoni** — Riga, Latvia
[giovanni.dominoni@gmail.com](mailto:giovanni.dominoni@gmail.com) · [LinkedIn](https://www.linkedin.com/in/giovannidominoni/)
