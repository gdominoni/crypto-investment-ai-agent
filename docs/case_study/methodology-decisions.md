# Methodology Decisions Log

A running record of every non-obvious methodology choice in this project, why it was made, and whether it rests on a statistical justification or is a stated compromise (and why the compromise was accepted). Companion to [PROJECT_MAP.md](../../PROJECT_MAP.md) (where it lives in code) and [README.md](../../README.md) (what the system does) — this file exists so a reader can find out *why* a specific number or design choice is what it is, in one place, instead of archaeology through commit history.

Each entry is dated and never silently rewritten — if a decision is later reversed, a new entry says so and links back to the one it supersedes.

---

## `accepted` vs `validated` — two different claims, deliberately separate words

**Decision.** `accepted` means a candidate cleared the historical/backtest statistics. `validated` is reserved exclusively for a candidate that has actually lived through its own tracking window (see "N=50 replaces the 2-year milestone" below) while still `accepted`. The two words are never used interchangeably anywhere in code, prompts, or Telegram messages.

**Why.** The two originally used the same word, which let a candidate that had merely passed a historical backtest be described the same way as one with an actual live track record — a real source of confusion, not just a naming nitpick: it's the difference between "this pattern looks real in hindsight" and "this pattern kept looking real going forward, tested prospectively." Same discipline as the earlier finding that a component's 80.7% win rate was entry-price leakage — precise language about what's actually been demonstrated is not optional here.

**Type.** Definitional / statistical rigor, not a compromise.

---

## `classify_status`'s gate: pattern significance, not P&L

**Decision.** A candidate is `accepted` if `pattern_significance` finds a statistically significant, out-of-sample directional effect (vs. the coin's own unconditional baseline over the same period) with a favorable risk path (mean MFE > mean MAE at the horizon the effect was found at), and isn't carried by a single coin or period. Sortino, win_rate, and strict_win_rate — the TP/SL-conditioned backtest numbers — are still computed and still reported, but **no longer gate acceptance.**

**Why.** This project's stated purpose is discovering whether a real, reproducible relationship exists between a market condition and subsequent price behavior — not optimizing a specific TP/SL barrier structure. The two questions are related but different: a real, small, reliable pattern can fail a P&L gate simply because the barriers are too wide to register it (a high-timeout-fraction candidate); conversely, a barrier structure can look profitable by fitting the same noise it was graded against. Concretely observed on real data during this rework: a shock-based condition with `win_rate=42.5%` (would have failed the old win-rate gate permanently) turned out to have a statistically significant pattern (`p=0.027`) with a favorable risk profile (`MFE/MAE=2.01`) — exactly the case the old gate was structurally blind to.

**Type.** Statistical rigor. Direct consequence of the project's own stated goal, not an arbitrary preference.

---

## `pattern_significance`: how "does a pattern exist" is actually tested

**Decision.** For each walk-forward fold (expanding window, yearly), the holding horizon is chosen on the **train** set only (whichever of `(1, 3, 7, 14, 21)` days shows the strongest `|mean forward return|`), then the effect is measured **only on the held-out test fold** at that horizon — same discipline already used for TP/SL multiplier selection, extended to horizon selection, specifically so the horizon can never be picked and graded on the same data. The test-fold sample's mean forward return is compared against the coin's own unconditional forward-return distribution over the *same calendar stretch* (not the whole multi-year history — that would let a triggered sample from an unusually volatile year get compared against a calmer baseline). Significance is assessed via bootstrap (2,000 resamples), not a textbook t-test, because financial returns violate the assumptions a t-test needs (fat tails, and overlapping-window autocorrelation when trigger events cluster in time).

**Why.** Any single-horizon comparison chosen after seeing the full sample is picking-and-testing on the same data — the exact trap a prior, predecessor research effort's own methodology fell into with data leakage, before this project's own causality-safe rebuild. The train/test split for horizon selection closes that gap the same way it's already closed for TP/SL.

**Type.** Statistical rigor.

---

## N > 50 replaces N > 100 as the sample-size gate

**Decision.** `MethodologyConfig.min_report_events = 50` (was 100, briefly, before that).

**Why.** A compromise, stated plainly: N > 100 combined with the other new gates (Sortino > 1, win_rate > 50%, strict_win_rate ≥ 45%, MFE/MAE > 1, statistical significance, no concentration) risked accepting *nothing at all* given the amount of real history available — five gates stacked that tightly is a lot to clear simultaneously. Lowering the sample-size floor to 50 doesn't weaken any of the other, more important gates (particularly statistical significance, which is the one that actually answers "does a pattern exist") — it just stops sample size itself from being the bottleneck. If nothing clears the bar even at N=50, that is itself a real, reportable finding, not a problem to engineer away further.

**Type.** Compromise (practical yield vs. rigor), explicitly not a loosening of the pattern-existence test itself.

---

## N=50 live tests replaces the 2-year calendar milestone

**Decision.** The one-time "has this candidate been validated" report now fires the first time a candidate accumulates 50 resolved **live** tests (not backtest events — see "Live testing" below), not after 2 elapsed calendar years. The existing 2-year "tracked this long and never once accepted" keep-or-drop decision is unaffected and stays as the safety net for candidates whose trigger is too rare to ever reach N=50 in a reasonable span.

**Why.** A fixed calendar span is arbitrary relative to how often a given trigger actually fires — a common condition could reach N=50 in weeks, a rare one might take years past any fixed calendar cutoff, or never get there at all. Gating the milestone on the same sample size the statistics themselves require (N=50, the acceptance floor above) ties the checkpoint to actual statistical readiness instead of an unrelated calendar convention.

**Type.** Statistical rigor (replaces an arbitrary calendar convention with the same threshold the statistics already require).

---

## Live testing: no TP/SL execution, hold for the horizon, measure forward return + MFE/MAE

**Decision.** Once a candidate is `accepted`, a live occurrence of its trigger opens a **live test**, not a funded position with a TP/SL exit. It's held for exactly the horizon `pattern_significance` found significant at (no barrier check in between), then resolved by measuring the realized forward return, MFE, and MAE — the identical measure `pattern_significance` itself uses. `barrier_prices`, `bucket_for_elapsed`, and the TP/SL grid search are no longer part of how a live/replay test opens or resolves.

**Why.** If acceptance is decided by "does a fixed-horizon forward return differ from baseline," then executing with a *different* structure (a Sortino-optimized TP/SL ladder) live would test a derived strategy, not the actual pattern that was accepted — the live occurrence would no longer measure the same concept the backtest measured. TP/SL/Sortino remain useful, reported information (how well a barrier-based structure would have captured this pattern) but are no longer the thing being tested live.

**Type.** Statistical rigor / conceptual consistency, not a compromise. (This originally left open how, or whether, this applies to production's own execution — resolved by the later "This project never opens a funded position" decision below: the same live-test model, no exception.)

---

## Simulation start date: earliest available data (2017), not an arbitrary offset

**Decision.** The historical replay's simulated clock starts at `min(coin.index.min() for coin in COINS)` — 2017-08-26 in the current data (BTC/ETH) — rather than a fixed "N years before today" offset.

**Why.** `pattern_significance` needs as many yearly walk-forward folds as it can get, both for a robust horizon choice and to give each tracked trigger a real chance of reaching its own N=50 live-test milestone within the simulated run.

**Compromise this creates.** The first several simulated years mostly return `insufficient_data` — there aren't yet enough yearly folds (`min_train_periods = 3`) for `pattern_significance`/`walk_forward` to run at all. This is expected, not a bug, but it does mean roughly the first 3-4 simulated years are statistically quiet. Starting from 2017 instead of a 3-year window also roughly **triples** the number of simulated days the replay has to walk through (~110 chunks of 30 days vs. ~36), which is a real increase in wall-clock time and API calls to complete a full run — accepted deliberately for the sake of a longer, more defensible walk-forward history.

**Type.** Compromise, stated plainly, in service of the statistical goal above.

---

## Manually-seeded horizon for the earliest tracked candidates

**Decision.** For a candidate tracked from the very start of the simulated history (the six static, "academic" C1-C6 triggers), there is no prior data at all to run `pattern_significance`'s train/test horizon selection against for the first several years. Until enough yearly folds accumulate for the empirical selection to run, live tests for these candidates are held for a single, **uniform, neutral placeholder horizon** (the middle of the search space, 7 days) — not a different "logical-sounding" horizon hand-picked per trigger.

**Why the uniform default, specifically.** Picking a different horizon per trigger based on domain intuition ("C2 is a reversal pattern, effects concentrate in the first week") would be exactly the kind of unfalsifiable, story-shaped reasoning this project has spent this whole rework moving away from — it would look, to any careful reader, indistinguishable from choosing the answer and writing the justification afterward. A single neutral default, applied uniformly and labeled explicitly as a placeholder, makes the compromise legible instead of disguising it as domain expertise.

**Transition.** The moment `pattern_significance` can actually run for a candidate (enough yearly folds exist) and returns its own empirically-derived horizon, that value takes over for all subsequent live tests — and the switch itself is reported on Telegram (from-placeholder to from-data, with the N and year it happened), not applied silently.

**Type.** Compromise, explicitly labeled, with a defined, documented, and notified exit condition — not a permanent manual override.

---

## Trigger detection: hourly; backtest and live-test resolution: daily

**Decision.** The historical/statistical backtest (`pattern_significance`, `walk_forward`, the weekly battery refresh) runs entirely on daily bars — unchanged. Separately, once past the simulated present, **detecting** whether a trigger condition has fired scans hourly bars within each simulated day (day-window indicators like `rsi_14d` are evaluated on a rolling window re-expressed in hours, e.g. 14 days → a 336-hour rolling window on the hourly series), so a condition that only crosses its threshold intraday and reverts by end of day isn't missed entirely by a once-a-day check. Once detected, opening and resolving the live test still uses the existing daily-bar machinery (entry snaps to the day's next daily bar; the horizon is still counted in days).

**Why the split.** Running the full backtest (which re-runs repeatedly, across years, per candidate, with a 2,000-sample bootstrap each time) on hourly bars would multiply compute cost by roughly 24x on top of the ~3x already introduced by starting from 2017 — a genuinely large cost for a self-funded case study, not a funded production system. Detection precision matters for correctly recognizing whether a condition was ever true at all; a few hours of drift in exactly *when* a multi-day pattern's holding period starts does not meaningfully change what it measures.

**Type.** Compromise (explicitly a cost tradeoff for a case study, stated here rather than left undocumented), scoped narrowly to detection only — the actual statistical claims (backtest, and the horizon/measurement used to resolve a live test) remain entirely daily and unaffected.

---

## Shock threshold: `z ≥ 3.0`, justified empirically, not by a "3-sigma" claim

**Decision.** `shock_zscore_series`'s threshold stays at 3.0 (a z-score of 5-day realized volatility against its own trailing 252-day distribution). The value itself is unchanged; its justification is rewritten to be honest about what it actually measures.

**Why the original justification was wrong.** "z ≥ 3.0" reads as "a 3-sigma event," which under a normal distribution should occur ~0.13% of the time. It does not: measured on the real pooled data across all 7 coins, `z ≥ 3.0` actually occurs **1.97%** of the time — about 15x more often than the normal-distribution framing implies, because realized-volatility z-scores are strongly right-skewed (empirical skew 1.8-3.3 per coin), not normal.

**What was actually checked before keeping 3.0.** A bootstrap test (same method as `pattern_significance`) comparing the 7-day forward return of the "above threshold" population against "below threshold," at candidate thresholds from z=1.5 to z=4.5: the effect (a positive excess return following elevated volatility) is present and similarly sized across the whole range; there is no sharp natural "elbow." What does change is statistical reliability as the sample shrinks: significant at p<0.05 through z=4.0, no longer significant by z=4.5 (p=0.068, N=126). 3.0 sits comfortably inside the range that stays both statistically reliable and reasonably extreme (~2% of observations), not at either edge of it.

**Type.** Statistical rigor for the justification; the specific value (3.0) is a defensible choice within a validated range, not a uniquely-derived optimum — there is no single "correct" answer the data hands over on its own, and this file says so rather than implying otherwise.

---

## Sonnet's role narrowed: no more `propose_trade` / `watch` / `exit_now`

**Decision.** Sonnet's live judgment now does exactly two things: (1) decide whether to ask a human to test a genuinely new condition (`propose_novel_test`), and (2) answer natural-language questions about system state. It no longer decides to open a trade (`propose_trade` — superseded by the mechanical, unattended trigger-detection scan described above, which fires identically for every occurrence of an accepted candidate without needing a per-event LLM judgment), no longer has a `watch` action (verified to have had zero behavioral difference from `no_action` — same downstream consequence, different message wording only), and no longer has an unused, never-implemented `exit_now` action (removed rather than built, since a discretionary live exit would reintroduce exactly the kind of unattributable LLM judgment this project has spent this rework removing from entry and TP/SL sizing).

**Why.** Each of these was either genuinely redundant with a more reliable mechanical process, or a piece of surface area that had no real behavior behind it. Removing dead/redundant paths is itself a form of rigor — fewer places where a claim about what the system does can silently drift from what it actually does.

**Type.** Simplification / consistency, not a compromise.

---

## Shock: a fixed statistical rule, never an LLM judgment call

**Decision.** Whether a given day/coin counts as a volatility "shock" stays a fixed, deterministic threshold (`shock_zscore ≥ 3.0`, see above) — never a qualitative judgment from Haiku/Sonnet. Sonnet's role is to interpret and react to an already-detected shock (or an already-published macro release, or a headline), including proposing that a *specific combination* of what it's shown (an indicator reading, a recent release, a headline) looks like a distinct, testable pattern — never to decide what magnitude of price move counts as extreme in the first place.

**Why.** Two reasons, both load-bearing: (1) reproducibility — the backtest needs a regime label it can compute identically over nine years of history, cheaply and deterministically; an LLM call per bar, per coin, per year is both prohibitively expensive and not reproducible run to run; (2) consistency with every other boundary already enforced this way in this project — TP/SL sizing, the indicator whitelist, and now trade execution are all explicitly *never* left to unattended LLM discretion. A qualitatively-judged shock threshold would be the same category of exception in the one place it was never allowed anywhere else.

**Type.** Design principle, applied consistently — not a compromise.

---

## This project never opens a funded position -- production gets the same live-test model as the replay

**Decision.** `execution/live_testing.py` + `execution/live_test_state.py` port the replay's exact live-test model to real, unsandboxed data: no TP/SL, no Freqtrade order, a real-dated occurrence held for the horizon `pattern_significance` found significant, resolved by measuring the real forward return/MFE/MAE. `candidates/status_history.py` gained the same N=50 milestone tracking replay/status_history.py already had, so "validated" is reachable in production too, on the same real evidence bar.

**Why.** This is a pattern-discovery investigation, not an investment strategy -- there is no funded position to protect or size, so there is nothing stopping the same observational discipline the replay uses from running on real, current data instead of simulated history.

**A genuinely new capability this makes possible: backdating a newly-discovered condition's own triggering occurrence.** By the time Sonnet proposes a new compound condition and a human approves it, real wall-clock time has already passed since the underlying condition first became true -- unlike the replay (which can look up any historical bar on demand), production has nothing tracking an unregistered condition before it's discovered. But since no funded position is ever placed, there is nothing physically stopping an honest, real-data retroactive read: `execution.live_testing.find_backdated_entry` scans the already-recorded hourly price history (kept fresh by `data_ingestion/market_data/binance_fetcher.py`, same data every other real-time check in this project uses) for the earliest hour the condition was already true, and backdates entry to that point instead of the discovery moment -- this would NEVER be legitimate for a real funded order (no exchange lets you buy at a historical price), but is exactly the honest thing to do for an observational record.

**A real bug this caught.** The first version of `find_backdated_entry` sliced the hourly series down to the lookback window *before* computing rolling-window indicators, starving multi-hundred-hour windows (e.g. a 720-hour funding z-score) of their lookback and silently returning nothing but NaN/False -- the exact same bug already caught once in `replay/engine.py`'s mechanical scan. Fixed the same way: compute on the full series, slice the result afterward. Caught by actually running the function against real data, not by reading the code.

**Type.** Statistical rigor / conceptual consistency, direct consequence of the "no funded position" decision above.

---

## Freqtrade hyperopt cross-check -- a second, independent optimizer, purely informational

**Decision.** `execution/hyperopt_runner.py` + `execution/freqtrade_bridge.py` + `execution/freqtrade_userdir/strategies/hyperopt_candidate_strategy.py` run Freqtrade's own Bayesian hyperopt engine, periodically and only ever locally, against real data already used everywhere else in this project, to independently re-derive the TP/SL multipliers for each tracked candidate's real anchor set. The result (best tp_mult/sl_mult + the resulting N/win-rate/Sortino/total-profit) is stored in `execution/hyperopt_results.json` and surfaced as one line in the 50-live-test milestone report -- never gates acceptance, never feeds live execution.

**Why.** This project's own walk-forward grid search (`candidates/methodology.py::walk_forward`) already computes an equivalent "if traded with a barrier structure" figure -- the "For reference, trading this with a TP/SL structure..." line shown in every message. Freqtrade's hyperopt is a genuinely *independent* second opinion: different search machinery (Bayesian optimization over a continuous parameter space vs. this project's own 25-point grid), a different, third-party, industry-standard backtesting engine, reusing the exact same real price history -- not a redundant re-implementation, a cross-check using different tooling arriving at (or interestingly failing to arrive at) similar numbers. For a project meant to demonstrate methodological rigor, an independent validation of one's own numbers is worth more than another internally-consistent chart.

**Real, non-obvious issues hit and fixed while building this (not guessed at, actually run against real data):**
- Freqtrade hardcodes `tickers_have_price=False` for Binance specifically -- config validation fails without `use_order_book: true` on both `entry_pricing` and `exit_pricing`, even in backtest/hyperopt mode where no order book is actually consulted.
- The `freqtrade[hyperopt]` extra (scikit-optimize/Optuna, filelock, etc.) is a separate install from the base `freqtrade` package.
- Best-epoch results are read via `freqtrade.optimize.hyperopt_tools.HyperoptTools.load_filtered_results` against the `.fthypt` file named in `hyperopt_results/.last_result.json` -- not scraped from console output.

**Cost.** A real, recurring local compute cost (Bayesian hyperopt runs hundreds of backtest evaluations per candidate) -- accepted explicitly because it's periodic, local-only, and never blocks the live/replay hot path (confirmed: `execution/freqtrade_bridge.py` imports `freqtrade` lazily, inside function bodies, specifically so importing `replay/engine.py` or `execution/live_testing.py` never requires the freqtrade package to be installed at all unless a hyperopt run is actually invoked).

**Type.** Enhancement for methodological credibility, not required for the system's own statistical claims (which stand on `pattern_significance` and the walk-forward grid search regardless) -- explicitly scoped as informational-only from the start.

---

## Q&A context bloat -- caught by running the historical replay for real, not by reading the code

**Decision.** `_trades_by_candidate_summary()` / `_trades_by_candidate_and_coin_summary()` (`replay/judgment.py`) and `build_live_test_summary()` (`llm_pipeline/context_builder.py`) now cap themselves at the top 15 rows ranked by `|mean_return|`, with a note pointing to the free, local `/summary`/`/replay_summary` command when truncated. `_all_candidates_status_summary()` (`replay/judgment.py`) collapses the (usually large) `insufficient_data` bucket into one count-plus-name-list line instead of one detailed line per candidate. See PROJECT_MAP.md's "Cost Optimization" section for the full breakdown of what each Sonnet call actually sends and why.

**Why.** This was a real, measured incident, not a hypothetical worth guarding against in the abstract: running the historical replay for real (this case study's own demonstration mechanism) discovered that `answer_market_question()`'s context had grown to ~10,400 tokens by the time the replay had tracked 96 candidates and logged 1,728 live tests -- 74% of that from two functions that, by design, listed *every* candidate and *every* (candidate, coin) pair ever seen, in full detail, on every single call, regardless of whether that call's question had anything to do with most of them. This is exactly the kind of cost/latency regression that's invisible from reading the code in isolation (each function looks reasonable on its own) and only shows up once real, accumulating state is actually exercised over a long run -- which is the whole reason this project insists on running things for real rather than trusting a static review.

**A distinction that matters here: not every Sonnet call has this problem.** The automated per-event judgment calls (`judge_event()`, `sonnet_strategist()`, `sonnet_shock_response()`) were never affected -- their context is built from the current indicator snapshot, the last 10 days of macro releases, and the currently-*accepted* candidate list, none of which are keyed to the total candidate or trade-log count. Only the human-Q&A path (built specifically to answer "give me everything" style questions) had this growth, because it deliberately listed exhaustive detail rather than a summary. The fix keeps that same category of information available (nothing is omitted, only the long tail past 15 rows is deferred to a free local command), it just stops the exhaustive part from being repeated, in full, on every single future call regardless of relevance.

**Type.** Cost/scalability fix, caught live -- not a statistical or methodology change, no effect on any candidate's classification.

---

## `/details` -- the exact numbers behind a status word, kept out of every other message on purpose

**Decision.** `/details <name>` (`telegram/bot.py`, and `/replay_details <name>` for the replay) shows a single named candidate's full numeric breakdown: the trigger's own exact threshold (`candidates/definitions.py::TRIGGER_NUMERIC_DEFINITIONS`, e.g. "funding z-score below -2.0" rather than `TRIGGER_DESCRIPTIONS`'s prose-only "an extreme funding rate"), N, p-value, excess return, MFE/MAE ratio, coin/year concentration as an exact percentage, the reference TP/SL backtest stats, and `explain_non_acceptance()`'s reason if it isn't accepted (`candidates/methodology.py::format_candidate_details()`). None of this detail is added to `/summary`, a Sonnet proposal, or a live-test notification -- those stay a one-line-per-candidate verdict by design (see the Q&A context-bloat entry above for the same instinct applied elsewhere: exhaustive detail belongs in a free, on-demand local command, not repeated in every message regardless of relevance).

**Why.** A real gap, not a hypothetical: a phrase like "high futures concentration" or "not statistically significant" tells a reader *that* a candidate failed a check, not *by how much* -- there was no way to answer "elevated concentration -- how elevated, exactly?" without reading the code. Folding the full numeric breakdown into every summary line or every proposal would fix that but make either too long to scan at a glance (a dynamic-registry `/summary` can already run to dozens of candidates). A separate, on-demand command answers both needs without trading one off against the other.

**A real bug this caught.** `dominant_year` (from `concentration_check()`'s groupby on a `period` column that's assigned as an int but can get upcast to `float64` by an unrelated NaN elsewhere in the same frame) rendered as e.g. `2023.0` instead of `2023` wherever a human read it -- including `explain_non_acceptance()`'s own "Why:" line, already shipped and visible in `/summary` before `/details` existed. Invisible until `/details` was run against real battery output and a live number was actually read end to end, not caught by reading `concentration_check()`'s code in isolation (a plain `int` output would look correct there). Fixed with a small formatting guard (`_format_dominant_year()`) applied in both places.

**Type.** UI/reporting addition plus a real formatting bug fix, caught live -- no effect on any candidate's classification (the underlying `max_year_share` value used for the 60% concentration gate was always numerically correct; only its display leaked the float artifact).

---

## Production never re-synced a candidate's horizon after its first "Test It" -- replay always did

**Decision.** `candidates/run_battery.py::run_all()` now re-derives and re-saves every candidate's horizon (`execution/live_test_state.py::save_horizons()`) on every run a `pattern_significance` result exists for it -- independent of accepted/watch/rejected, mirroring `replay/battery.py`'s own sync exactly. `scheduler/weekly_revalidation.py` also now sends a Telegram notice ("Horizon updated -- ...") whenever a candidate's horizon actually changes, mirroring `replay/engine.py`'s own notice.

**Why.** `pattern_significance()`'s `chosen_horizon` is the last walk-forward fold's own pick -- what an occurrence discovered *now* should be held for -- and it can legitimately shift week to week as more data accumulates. `execution/live_testing.py::_open_live_test()` reads the horizon to hold a new live test for from `execution/live_test_state.py::load_horizons()`, a SEPARATE file from the one `run_all()` writes its own per-candidate `"horizon"` field into (`execution/live_battery_state.json`). Before this fix, nothing in production's weekly cycle ever called `save_horizons()` -- the only call anywhere in production code was in `telegram/bot.py::handle_test_it_confirmation()`, fired once, the moment a human approves a brand-new novel condition. In practice this meant: a static candidate's (C1/C2/C6) horizon was **never** set at all in production, so every live test for one silently used the `PLACEHOLDER_HORIZON_DAYS=7` fallback regardless of what `pattern_significance` actually found; a dynamic candidate's horizon was set once at approval and then frozen forever, never updated by any of the dozens of weekly re-validations that followed.

**A real gap between production and replay, not a hypothetical.** `replay/battery.py` already called `state.save_horizons()` on every single run, for both static and dynamic candidates (lines 77-82 and 116-121) -- the fix makes production match code that was already correct on the replay side, rather than inventing new logic. Caught not by reading the code in isolation but by a direct question about whether the live system genuinely re-optimizes its own horizon over time, followed by grepping every call site of `save_horizons()` in the repository and finding production's weekly cycle simply never among them.

**Type.** Real bug fix, production/replay parity -- a live test's held-for-period was silently wrong (or stuck) for every candidate until this fix; no effect on any backtest classification, since `pattern_significance`'s own computation of `horizon` was always correct, only its propagation to where a NEW live test actually reads it was missing.

---

## What counts toward "validated" differs by how a candidate was discovered

**Decision.** `_effective_milestone_count()` (`execution/live_testing.py`, mirrored in `replay/engine.py`) replaces a single, origin-blind rule with two: static candidates (C1/C2/C6) still require 50 real (or, in the replay, simulated) resolved live tests -- unchanged. Dynamic (Sonnet-proposed) candidates instead use a rolling window of the most recent 50 occurrences, backtest and live mixed: `min(backtest_n, 50 - live_n) + live_n` while `live_n < 50`, otherwise just `live_n`. Since every live occurrence is by definition more recent than every backtest one, this is exactly a "most recent 50, chronologically" window -- it fills from live occurrences first and tops up with the freshest backtest ones only while live evidence alone is still short of 50. The window only decides *when* a checkpoint fires; `pattern_significance`/`classify_status` themselves are untouched, still computed over full available history exactly as before. `candidates/status_history.py::candidates_due_for_milestone()`/`mark_milestone_reported()` needed no changes at all -- they already took a generic count dict, so the origin-dependent logic lives entirely in the two callers.

**Why.** A direct consequence of an earlier finding in this log (the multiple-comparisons / hypothesis-generation-contamination discussion): static candidates were derived by directly mining this project's own historical data (a dedicated prior research phase, see "the three candidates selected... out of the prior research's larger set" in `candidates/definitions.py`) -- a strong, direct look-then-test risk, so only genuinely prospective evidence should count toward calling one validated. Dynamic candidates carry a much weaker, diffuse version of the same risk: `sonnet_strategist()`/`sonnet_shock_response()` never see this project's own backtest results before proposing a condition (verified directly against `llm_pipeline/haiku_sonnet_pipeline.py`'s system prompts -- only a live indicator snapshot and the last 10 days of macro releases), only whatever general market-pattern knowledge Sonnet's own training absorbed. That residual risk can't be measured or corrected for after the fact (unlike the separate, still-unaddressed multiple-comparisons problem noted in this log's other recent entry), but it doesn't warrant making a dynamic candidate wait for 50 real live occurrences of a possibly-rare trigger -- sometimes months or years -- before its first checkpoint, when its backtest evidence is already substantial.

**A real, immediately observable consequence, checked against real data before shipping.** Because `classify_status` already requires backtest `n > 50` before a candidate can ever be `accepted` in the first place, `_effective_milestone_count()` is pinned at exactly 50 for every dynamic candidate from the moment of acceptance until its own `live_n` independently exceeds 50 -- meaning a dynamic candidate's *first* validation checkpoint fires on the very next weekly run after acceptance, not after waiting for real live exposure. Confirmed against the replay's real state: `high_efficiency_breakout_with_volume_confirmation` (backtest N=159, only 1 real live test) was immediately flagged as due for its first checkpoint. Every *subsequent* checkpoint (100, 150, ...) still requires that many real live occurrences -- the rolling window only ever accelerates the first one.

**Type.** Methodology decision, direct consequence of the earlier multiple-comparisons discussion in this log -- differentiates the "validated" gate by discovery process rather than applying one rule uniformly, deliberately, not a compromise.

---

## A well-established candidate's own aggregate is, by design, slow to react to a real regime change -- a fast informational alert covers the gap

**Decision.** `_check_consecutive_failures()` (`execution/live_testing.py`, mirrored in `replay/engine.py`) fires immediately after each live test resolves, only for a candidate that's currently VALIDATED (`milestone_cleared`). If its most recent resolved live tests, counted backward, show `CONSECUTIVE_FAILURE_ALERT_THRESHOLD=2` or more negative forward returns in a row, it sends a Telegram alert showing the last `max(streak, 5)` occurrences with their individual forward return/MFE/MAE, plus the mean return and MFE/MAE ratio over that window. Purely informational -- it never changes any candidate's status; `classify_status`/`pattern_significance` are completely untouched by it.

**Why.** Directly measured, not assumed: reconstructing `c2_long`'s (N=62, marginal p=0.034) and `c6_long`'s (N=289, strong p=0.0005) real out-of-sample return populations and simulating consecutive additions of each candidate's own worst-ever observed loss (not an average loss -- the single worst MAE actually recorded, repeated) showed `c2_long` flips out of significance after only 3 such worst-case failures, while `c6_long` needs roughly 30 before its p-value crosses 0.05. This confirms two things simultaneously: (1) a large, statistically overwhelming sample is *correctly* resistant to short-term noise -- that resistance is the entire point of testing significance over a larger N, not a flaw; (2) precisely because of that resistance, if a well-established candidate's real-world edge stops working for a genuine reason (a market-structure shift, a new regulation on futures funding, the specific inefficiency getting arbitraged away), the aggregate alone could take dozens of real occurrences -- plausibly months -- to reflect it. Because this project never opens a funded position, the cost of that lag is not capital at risk, but it is still a real gap: a human watching the system has no fast signal that something might be going wrong, only the slow-moving aggregate. The alert closes exactly that gap without touching the aggregate's own (correct) behavior.

**Scoped to VALIDATED candidates only, deliberately.** A candidate that's merely `accepted` but not yet validated still has a comparatively small sample (by definition, `n` only just above `min_report_events=50`), so its own aggregate is already reasonably sensitive to new occurrences -- see the same experiment above, where `c2_long` at N=62 flipped after just 3 worst-case failures with no separate alert needed at all. The alert exists specifically for the population where the aggregate's own resistance to noise becomes a genuine blind spot: candidates with enough accumulated history that a short losing streak is invisible to the aggregate for a long time.

**Type.** Methodology decision, additive and purely informational -- verified against real reconstructed data before deciding the threshold was worth building, then verified again end-to-end against the replay's own real trade log (a real losing streak on `c1_short`, `milestone_cleared` forced True to exercise the alert path directly, all state and Telegram sends mocked out for the check -- no real message sent, no real state mutated) before shipping.

---

## `/details` never showed `VALIDATED` -- a real, live-caught gap between two different claims

**Decision.** `format_candidate_details()` now takes a `milestone` parameter (the caller's `all_latest_statuses()[candidate]` entry) and shows a `VALIDATED`/`NOT validated`/"hasn't reached its first checkpoint yet" line whenever milestone info exists for that candidate -- `telegram/bot.py`'s `/details` and `/replay_details` handlers now fetch and pass it.

**Why.** A real, live-caught bug, not a hypothetical: `high_efficiency_breakout_with_volume_confirmation` became genuinely `VALIDATED` (a real checkpoint fired, `milestone_cleared=True` in `replay/state/status_history.json`) during a real replay run -- but `/replay_details` on that exact candidate right afterward showed only `Status: accepted`, with no mention of `validated` anywhere, because `format_candidate_details()` only ever read `row["status"]` (from `run_battery.py`/`run_replay_battery()`'s own return value) and had no access to the SEPARATE milestone-tracking state (`status_history.py`) at all. This is exactly the confusion the "accepted vs validated" entry earlier in this log warns about in the abstract -- here it actually happened, in this project's own most detail-oriented command, the one built specifically to answer "what does this status actually mean, precisely."

**Type.** Real bug fix, caught by a human actually using the feature and noticing the mismatch against what they'd been told moments earlier -- not caught by reading the code (`format_candidate_details()` looked complete and correct in isolation; the missing piece was an input it was never given, not a flaw in its own logic).

---

## `run_bot()`'s unprotected `_get_updates()` -- a documented gap that actually crashed the process

**Decision.** `run_bot()`'s long-poll loop now wraps its own `_get_updates()` call in try/except (10s backoff, then retries) -- previously only the per-update dispatch was protected, mirroring the same fix `scheduler/live_daemon.py` already had for its own polling.

**Why.** This exact gap was already documented in PROJECT_MAP.md's "Partial Failures & Crashes" as "known, not yet handled" -- reasoned to be acceptable because `run_bot()` was meant for isolated testing, with `live_daemon.py` as the real, intended way to go live. It stopped being theoretical the moment `run_bot()` was actually run standalone as a real, ongoing process (deliberately, to answer commands without the daemon's proactive hourly/weekly jobs): a second, unrelated `getUpdates` call made from outside the running loop (Telegram allows only one active long-poll per bot token) caused the *next* poll inside `run_bot()` to receive an HTTP 409 Conflict, unhandled, which killed the entire process silently -- no crash alert, no auto-restart, just a bot that stopped answering until someone noticed and manually restarted it.

**Type.** Real bug fix, caught live -- promotes a previously-accepted, explicitly-scoped gap to fully handled once the assumption behind accepting it ("only ever run via live_daemon.py") stopped holding.

---

## The new VALIDATED tag leaked onto REJECTED candidates -- gated on current status, not just on milestone history existing

**Decision.** `_trigger_summary_line()` and `format_candidate_details()` now only show the `VALIDATED`/`not validated` tag when the candidate's CURRENT status is `accepted`. `milestone_reported`/`milestone_cleared` existing is no longer sufficient on its own.

**Why.** A real bug, caught immediately after shipping the previous fix: `milestone_reported`/`milestone_cleared` persist in `status_history.json` from whenever a candidate's checkpoint last fired -- which can be long in the past, while the candidate was still `accepted`. A candidate can (and several real ones did) later degrade to `watch`/`rejected` on a subsequent weekly re-validation without that stale milestone data ever being cleared. The previous version of this fix showed the tag for ANY candidate with milestone history regardless of current status, so `/summary`'s `Rejected` section started showing lines like `shock_extension_breakout -- N=248, p=0.149, MFE/MAE=2.78, not validated` -- reading as if "not validated" were part of today's verdict, when it's really a leftover fact from a checkpoint reached under a completely different (and no longer current) status. `explain_non_acceptance()`'s own "Why:" line already gives the real, current reason directly underneath; the stale tag added nothing but confusion right next to it.

**Type.** Real bug fix, caught immediately after shipping -- yet another instance of this project's own recurring failure mode (a stat or tag shown next to a verdict it doesn't actually describe, implying a relationship that isn't there), same family as the concentration/significance branch-order bug and the group-level `STATUS_PLAIN` gloss bug earlier in this log.

---

## `/replay_summary`'s "still under test" message silently never arrived -- over Telegram's real length limit

**Decision.** `telegram/bot.py::_send()` now splits any message over Telegram's real 4,096-character limit into several messages (`_chunk_message()`), preferring paragraph then line boundaries so no HTML tag is ever split across two messages, falling back to a raw character split only for a single line that's still too long on its own. `reply_markup` attaches only to the last chunk; `pin` applies only to the first. Returns `True` only if every chunk sent.

**Why.** A real, observed failure, reported directly by a human who noticed a response was simply missing: `/replay_summary`'s "still under test" message (Accepted + Watch + Insufficient data, one combined string) reached 6,880 characters once the dynamic registry grew to 96 tracked candidates -- Telegram's `sendMessage` rejects anything over 4,096 outright. `format_trigger_summary()` already splits its output into two SEPARATE messages ("still under test" vs. "already discarded") specifically reasoning about this exact risk -- but that split alone doesn't protect against either HALF growing past the limit on its own as the registry keeps growing, which is exactly what happened here. Worse, nothing at the `/replay_summary`/`/summary` call sites checked `_send()`'s own return value, so the failure was completely silent: no error, no alert, the human just never received that message and had no way to know why.

**Type.** Real bug fix, caught live by a human noticing an entire response section was missing -- fixed at the `_send()` layer (every caller benefits automatically) rather than patched at the two call sites that happened to trigger it, since any sufficiently large, unbounded state (a growing dynamic registry, in this case) could hit the same limit from a different message in the future.

---

## `/replay_details` on a replay-only dynamic candidate showed "trigger definition not found"; `/details`/`/replay_details` never showed the reference TP/SL multipliers

**Decision.** Two real bugs, caught back to back on the same real candidate. (1) Added `_replay_trigger_numeric_description()` (`telegram/bot.py`), mirroring `replay/engine.py::_trigger_description()`'s own lookup against `replay/state.py::load_dynamic_candidates()` -- the `/replay_details` handler now uses it instead of `_trigger_numeric_description()`, which only ever checked production's registry. (2) `format_candidate_details()` now takes `tp_mult`/`sl_mult` and shows them in the "Reference TP/SL backtest" line; both bot.py handlers now read them from the data they already had in hand (production: `run_all()`'s own `live_state` return value, previously discarded as `_live_state`; replay: `replay/state.py::load_battery_status()`, populated by `run_replay_battery()`'s own side effect).

**Why.** (1) Production and the replay track two entirely separate dynamic-candidate registries (see PROJECT_MAP.md's "Historical Replay" section) -- a candidate discovered only during the replay was never going to be found by a lookup that only ever checks production's, exactly what happened: `/replay_details high_efficiency_breakout_with_volume_confirmation` showed "trigger definition not found" for a real, validated, currently-accepted candidate. (2) The "Reference TP/SL backtest" line showed win rate, Sortino, and total expectancy, all of which are meaningless without knowing what TP/SL structure produced them -- the data (`tp_mult`/`sl_mult`, the project's own walk-forward grid search's chosen multipliers against the duration-bucketed anchors) was already being computed and returned by both `run_all()` and `run_replay_battery()`, just never read at the one place a human asks for exactly this level of detail.

**Type.** Real bug fixes, both caught live in immediate succession by a human actually reading the command's output line by line -- same pattern as every other fix in this section of the log: the missing piece was an input never passed in, not a flaw in `format_candidate_details()`'s own logic.

---

## Candidate names bolded consistently everywhere, not just in some messages

**Decision.** Every candidate/trigger name shown in any Telegram message is now wrapped in `<b>...</b>` -- `/summary`/`/replay_summary`'s per-line listing (`_trigger_summary_line()`) and its "no historical occurrences yet" name list (`_insufficient_data_block()`), both "not found" error messages, the live-test-opened/resolved messages, the consecutive-failure alert's closing paragraph, and `weekly_revalidation.py`'s status-change diff line.

**Why.** A real, spot-checked inconsistency: headers ("Checkpoint at 50 occurrences -- X", "Consecutive-failure alert -- X") were already bold, but the exact same name one line below, in the body of the same message or in a different command entirely, often wasn't -- `/summary`'s own per-candidate listing, the single most-read command in this whole system, never bolded a name at all. Nothing here changes what any message says, only how consistently a name reads as a name across every message a human might see it in.

**Type.** Consistency/formatting fix, requested directly -- no effect on any computation or classification.

---

## 2026-08-29 -- Statistical audit: the significance test was not directional, and its bootstrap was badly miscalibrated. Every previously-reported result is superseded.

This is the largest correction in this log. A deep audit of `candidates/methodology.py` found four defects that compounded, and together they were the reason this project appeared to be finding patterns. **After the fix, no candidate in either the production battery or the historical replay is `accepted`, and none is `validated`.** The previously-reported "1 validated out of 98" is withdrawn.

### Defect 1 (critical) -- the test ignored the direction the candidate trades

Two halves of the same root cause:

- **Horizon selection used `abs()`.** `score = abs(float(np.mean(rets)))` picked whichever horizon showed the strongest effect *in either direction* -- so a `long` candidate could have its holding horizon chosen precisely because the effect was strongly **negative** there.
- **The p-value's tail was chosen after seeing the data.** `np.mean(boot >= observed) if observed >= baseline else np.mean(boot <= observed)` is a two-sided procedure priced as one-sided.

Neither `classify_status` nor anything downstream read `excess_return`, so a pattern running *opposite* to its own traded direction was labelled `significant`. Measured on the real static battery: **four of six candidates were "statistically significant" with a negative excess return** (c1_long −1.67% at p=0.010, c1_short −4.92% at p=0.0045, c2_short −7.79% at p=0.0070, c6_short −3.46% at p=0.0020). A synthetic candidate with p=0.001, MFE/MAE=2.4 and excess=−5% returned **`accepted`**. Only coincidence -- all four happened to have MFE/MAE < 1 -- kept them out of production.

**Fix.** `_forward_return` already signs its output by direction, so a positive excess always means "works in the direction actually traded." The horizon is now selected by *signed* mean, and the p-value is a *pre-specified* upper tail. No doubling is needed because the side is fixed in advance rather than read off the data. `classify_status` independently re-checks `excess_return > 0`, and `explain_non_acceptance` names a wrong-direction effect as its own distinct reason rather than collapsing it into "not significant."

### Defect 2 (critical) -- the bootstrap resampled i.i.d. from overlapping windows

The baseline pool is built from overlapping h-day forward-return windows (day 2's 21-day window shares 20 days with day 1's). Resampling them independently destroys that serial dependence and understates the null distribution's variance, biasing every p-value downward. The module's own docstring acknowledged this and did nothing about it.

**This was measured, not argued.** Under a *true null* -- observed sample drawn from the same process as the baseline, so there is no effect to find -- the shipped i.i.d. bootstrap rejected at **43.3%** against a nominal 5%. Nearly 9x over-rejecting. That single fact explains why six of six candidates looked significant.

**Fix.** A moving-block bootstrap (`_block_bootstrap_means`), sampling contiguous blocks within a chunk, never across chunk boundaries. Block length calibrated empirically against that same true-null harness:

| resampling | false-positive rate (target 5%) | power vs. a real +4% effect |
|---|---|---|
| i.i.d. (shipped) | **43.3%** | — |
| block = 1x horizon | 15.3% | 39.0% |
| **block = 3x horizon (chosen)** | **8.7%** | 25.0% |
| block = 4x horizon | 7.7% | 23.5% |

Verified unbiased: block and i.i.d. bootstrap means agree to 0.0015 (the block version is 3.7x wider, which is the entire point). The residual **8.7% is stated rather than rounded to 5%** -- and it is an upper bound, because the calibration harness draws the observed sample as a fully contiguous slice (maximum dependence) while real trigger events are clustered but scattered.

### Defect 3 -- concentration was measured on a different quantity than acceptance

Acceptance is decided by `pattern_significance`'s raw forward returns; `concentration_check` only ever ran on `walk_forward`'s **TP/SL-conditioned** `net_return`. Two different quantities that genuinely disagree -- on identical events, 96.8% concentration on the TP/SL basis versus 50.0% on the forward-return basis. The README's "no single coin or year may carry more than 60% of the positive return" never said which return, and the two answers differed.

**Fix.** `pattern_significance` now returns per-event `oos_events` (group, period, forward_return), and all three callers (`run_battery`, `replay/battery`, `novel_condition_tester`) run concentration on that. `concentration_check` gained a `value_col` parameter; the TP/SL basis is still computed and reported as a diagnostic.

### Defect 4 -- `concentrated: False` when there was nothing to concentrate

When no group had a positive return, `concentration_check` returned `concentrated: False` -- so a candidate losing money on **every single coin** cleared both concentration gates. Harmless while significance was a real gate; not harmless combined with Defect 1. Now returns `concentrated: None` ("cannot assess"), which `classify_status` treats as `watch`, not as a pass.

### Two smaller correctness fixes found in the same pass

- **A flawless candidate was rejected.** `sortino_ratio` returned NaN whenever `downside_dev == 0` -- which happens both for an empty sample *and* for a candidate with no losing trade at all. `classify_status` rejects on a NaN Sortino, so a candidate that never lost was rejected for it. Now returns `+inf` for the no-losses case, NaN only for genuinely unusable input.
- **Silent end-of-series clamping.** `_forward_return` and `path_outcome` clamped `exit_loc` to the last available bar, so an occurrence near the edge of the data returned a *0-bar hold* dressed up as a full-horizon result (measured: `+0.0000%` forward return, and a real-looking `+1.20%` from `path_outcome` with NaN excursions). Unreachable from the battery -- `build_events`'s `entry_loc + max_h >= len(idx)` filter is exactly correct, verified -- but reachable from **live resolution**, precisely where a wrong number becomes recorded evidence. Both now return NaN, and both live-test resolvers leave the test open and retry rather than recording a partial hold.

### What was verified clean, and is worth saying plainly

The audit tried to break the causality layer and could not. Recomputing all six triggers on truncated history (first 2,000 bars) versus full history produced **zero** differing past values in every trigger column; `shock_zscore_series` max |diff| on the overlap was **0.0**. Across 49 real events: zero entries at or before the trigger bar, `entry_loc == trigger_loc + 1` for 100%, full horizon room for 100%. No zero-division artifacts in the funding z-score on real data (0 infinities, max |z| = 5.3). The rolling-then-slice ordering -- the bug caught twice before in `find_backdated_entry` and `_scan_mechanical_triggers` -- is correct throughout this module.

### The result, and why it is reported rather than tuned away

| | before | after |
|---|---|---|
| static battery `accepted` | 2 of 6 | **0 of 6** |
| static battery "significant" | 6 of 6 | 0 of 6 (best: c6_long, p=0.075) |
| replay `accepted` (98 candidates) | 2 | **0** |
| replay `validated` | 1 | **0** |

`high_efficiency_breakout_with_volume_confirmation`, previously the project's one validated candidate, does not clear the corrected bar. Its earlier VALIDATED checkpoint was real in the sense that it genuinely fired -- but it fired on a statistic that was measuring the wrong thing.

This is the outcome README's own Phase 1 "honest finding" predicted and the Dynamic Agent Thesis was built to test against. The correct response is to report it, not to relax a threshold until something passes: a 43% false-positive rate producing candidates is not a discovery, and a project whose entire stated purpose is distinguishing real patterns from flattering noise does not get to keep the flattering noise.

**Type.** Critical statistical bug fix. Supersedes every previously-reported acceptance and validation result in this repository, including this log's own earlier entries on `_effective_milestone_count` and the consecutive-failure alert (both remain correct mechanisms -- they simply have no `accepted` candidate to act on right now). Found by audit, every claim above confirmed by execution against real data before being written down.


---

## 2026-08-29 (later the same day) -- making the actual thesis testable: sequences, the right control group, graded macro events, and multiplicity control

The audit above fixed how a hypothesis is *tested*. This entry is about what could be *expressed* and *asked* at all -- five changes, four of them prompted by a single observation: this project's title promises patterns from **market conditions combined with events**, and the pipeline could not represent that claim.

### The gap, measured

Of the 92 conditions Sonnet actually proposed in the replay: 49 (53%) included `shock_zscore`, 12 (13%) `is_macro_day`, and **31 (34%) contained no event term at all** -- pure chart patterns, indistinguishable from something written directly in Freqtrade with no LLM involved. `high_efficiency_breakout_with_volume_confirmation`, the project's former validated candidate, was one of these.

Worse, of **771 live tests** opened for Sonnet-discovered candidates, **zero** were news-linked. All 771 came from the mechanical hourly scan, which reads price, OHLC and funding only. Haiku's sentiment decides *which* condition gets proposed and then disappears entirely from both the test and the track record. Going live does not fix this: acceptance is always decided by a backtest, and there is no historical news archive to backtest against (verified: CryptoCompare's endpoint is live-only, `lTs` backward paging returns empty, 3,527 days missing).

### 1. Sequenced conditions (`Clause.within_days`)

Every clause was evaluated on the same bar, so the grammar could express *"news AND crash on the same day"* but not *"crash, THEN news"* -- the central case. `within_days=K` means "was true at any point in the last K days" (0 = today, the previous and default behaviour). All three orderings are now writable **and comparable**, so ordering itself becomes a testable hypothesis. Causality holds: the window looks strictly backward, the trigger bar is where the last clause becomes true, entry is still the next bar's open. Rendered explicitly to humans, because "crash then news" and "crash and news together" must never read identically.

### 2. The incremental baseline -- the change that matters most

`pattern_significance` compared every condition against the coin's **unconditional** forward returns. Testing `shock AND negative_news` that way is close to meaningless: the shock *alone* already differs from an ordinary day, so the news clause could be pure decoration and the test would still pass it. `baseline_events` switches the control to the **same condition with its event clause removed**, same period, with the treated events excluded so it is a genuine treatment-vs-control contrast. Demonstrated on a real hypothesis (macro-release day AND RSI<45 → long):

| question | excess | p |
|---|---|---|
| unconditional -- "does anything happen at all?" | **+1.09%** | 0.356 |
| incremental -- "what does the macro day ADD?" | **−0.23%** | 0.483 |

The entire apparent effect belongs to the market state. The unconditional test would have credited it to the event. `baseline_kind` is now reported so the two claims are never worded identically.

### 3. Graded macro surprises, from data already on disk

`is_macro_day` is binary. The ALFRED vintages needed to grade *how far* a print moved were already downloaded, and `latest_release_with_prior` already computed the delta for Sonnet's prompt -- it was simply never a testable indicator. Three added (`cpi_surprise`, `rate_surprise`, `jobless_claims_surprise`), point-in-time correct on publication date with `shift(1)`-ed trailing stats, using each period's first print rather than its revision.

*A real bug caught by reading the output rather than trusting it:* the Fed funds rate sits flat for years, collapsing its rolling std, and a bare `sd > 0` guard produced a surprise of **−2,613,348 sigma**. A scale-free floor now yields NaN when the trailing window is degenerate -- the honest answer when there is no scale to judge against. Deliberately *not* suppressed: jobless claims at +137.94 on 2020-03-26 (281,000 → 3,283,000) is the real COVID spike.

### 4. Multiplicity control (Benjamini-Hochberg)

Flagged as unaddressed in the audit entry above, and no longer optional now that the search space includes orderings and graded events. Testing ~98 candidates at p<0.05 is *expected* to manufacture ~5 significant results with nothing behind them. `apply_fdr_demotion` runs as a family-level second pass and is **demotion-only** -- BH is uniformly at least as strict as raw p<α, so it can remove a candidate from `accepted` but never add one. BH rather than Bonferroni: at n=98 Bonferroni implies a per-test threshold of 0.0005 and no power for the modest real effects being sought. Validated against the canonical Benjamini & Hochberg 1995 worked example (15 hypotheses → exactly 4 discoveries) and cross-checked against `scipy.stats.false_discovery_control`.

### 5. A pre-existing bug found while verifying train/serve agreement -- worse than the one being looked for

Four of twelve indicators were **not distributionally comparable** between the daily backtest and the hourly live scan (BTCUSDT, 1st–99th percentile):

| indicator | daily | hourly@scale=24 | |
|---|---|---|---|
| `rsi_14d` | 22.3 – 85.4 | **42.6 – 58.5** | 0.25× spread |
| `atr_pct_14d` | 0.021 – 0.151 | 0.004 – 0.032 | 0.22× |
| `daily_range_pct` | 0.008 – 0.194 | 0.001 – 0.047 | 0.24× (ignores `scale` entirely) |
| `efficiency_ratio_20d` | 0.003 – 0.760 | 0.001 – 0.185 | 0.24× |

A condition accepted on `rsi_14d < 35` fires **289 times** in the daily backtest and fired **zero** times in the live scan -- a 336-period RSI mean-reverts to ~50 and never reaches the threshold. The candidate looks merely *rare*, not broken. `shock_zscore` already carried a hardcoded exception for exactly this reason; it was never generalised. `DAILY_NATIVE_INDICATORS` now covers all of them and both scanners delegate to one shared `clause_signal_hourly`. After: 289 → 289, zero missed, production and replay byte-identical.

*Also found in the same pass:* all three serializers wrote only indicator/op/threshold, so a sequenced condition would round-trip back as a same-day one -- approved as "crash then news", then silently tested and tracked as "crash and news together". `clause_to_dict`/`clause_from_dict` are now the single pair everywhere, the latter doubling as the sanitiser for model output.

### What is still missing, stated plainly

**Headline sentiment remains untestable**, and therefore the "Market Sentiment" half of this project's title remains unproven. The whitelist has no sentiment term because there is no historical news archive to backtest one against. Closing it requires backfilling news history (GDELT 2.0 is the only free source plausibly covering 2017→present; scoring it with Haiku batched by day costs roughly $11–35, since ~3,500 daily calls is far cheaper than per-article scoring). Until then the honest scope of this system is **market conditions combined with market and macro events** -- which is now genuinely expressible and correctly tested, and was not before.

**Type.** Capability + methodology. Four fixes make the project's own stated hypothesis representable and correctly controlled; one is a real pre-existing bug that silently prevented a whole class of accepted candidates from ever firing live. Tests 49 → 73.
